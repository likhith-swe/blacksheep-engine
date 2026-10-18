"""
Simulation API Routes
Step 2: Zep entity ingestion & filtering, OASIS simulation preparation and execution (fully automated).
"""

import os
import traceback
from contextlib import nullcontext
from flask import request, jsonify, send_file

from . import simulation_bp
from ..config import Config
from ..services.zep_entity_reader import ZepEntityReader
from ..services.oasis_profile_generator import OasisProfileGenerator
from ..services.simulation_manager import SimulationManager, SimulationStatus
from ..services.simulation_runner import (
    SimulationRunner,
    RunnerStatus,
    SimulationStopPending,
)
from ..services.zep_graph_memory_updater import ZepGraphMemoryManager
from ..utils.logger import get_logger
from ..utils.locale import t, get_locale, set_locale
from ..utils.zep_lifecycle import get_graph_readers, graph_lifecycle_lock
from ..models.project import ProjectManager

logger = get_logger('blacksheep.api.simulation')


def _get_default_platform(simulation_id: str) -> str:
    """
    Return default platform based on simulation configuration

    Read enable_twitter / enable_reddit from SimulationState,
    returning the platform actually active rather than hardcoded default.

    Args:
        simulation_id: Simulation identifier

    Returns:
        'twitter' or 'reddit'
    """
    try:
        manager = SimulationManager()
        state = manager._load_simulation_state(simulation_id)
        if state:
            return state.get_default_platform()
    except Exception:
        pass
    return "reddit"


# Interview prompt optimization prefix
# Adding this prefix prevents agents from calling tools and instructs direct English text response
INTERVIEW_PROMPT_PREFIX = "Drawing upon your persona, all your memories and actions, reply directly in plain English text without invoking any tools: "


def optimize_interview_prompt(prompt: str) -> str:
    """
    Optimize interview prompt by prepending direct-answer instruction to prevent agent tool-calling loops
    
    Args:
        prompt: Original prompt
        
    Returns:
        Optimized prompt
    """
    if not prompt:
        return prompt
    # Avoid duplicate prefix
    if prompt.startswith(INTERVIEW_PROMPT_PREFIX):
        return prompt
    return f"{INTERVIEW_PROMPT_PREFIX}{prompt}"


# ============== Entity Ingestion Endpoints ==============

@simulation_bp.route('/entities/<graph_id>', methods=['GET'])
def get_graph_entities(graph_id: str):
    """
    Get all entities from knowledge graph (filtered)
    
    Returns only nodes matching predefined entity types
    
    Query parameters:
        entity_types: Comma-separated entity types (optional)
        enrich: Whether to fetch edge relationships (default true)
    """
    try:
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": t('api.zepApiKeyMissing')
            }), 500
        
        entity_types_str = request.args.get('entity_types', '')
        entity_types = [t.strip() for t in entity_types_str.split(',') if t.strip()] if entity_types_str else None
        enrich = request.args.get('enrich', 'true').lower() == 'true'
        
        logger.info(f"Fetching graph entities: graph_id={graph_id}, entity_types={entity_types}, enrich={enrich}")
        
        reader = ZepEntityReader()
        result = reader.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=entity_types,
            enrich_with_edges=enrich
        )
        
        return jsonify({
            "success": True,
            "data": result.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Failed to fetch graph entities: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/entities/<graph_id>/<entity_uuid>', methods=['GET'])
def get_entity_detail(graph_id: str, entity_uuid: str):
    """Get detailed information for a single entity"""
    try:
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": t('api.zepApiKeyMissing')
            }), 500
        
        reader = ZepEntityReader()
        entity = reader.get_entity_with_context(graph_id, entity_uuid)
        
        if not entity:
            return jsonify({
                "success": False,
                "error": t('api.entityNotFound', id=entity_uuid)
            }), 404
        
        return jsonify({
            "success": True,
            "data": entity.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Failed to fetch entity details: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/entities/<graph_id>/by-type/<entity_type>', methods=['GET'])
def get_entities_by_type(graph_id: str, entity_type: str):
    """Get all entities of a specified type"""
    try:
        if not Config.ZEP_API_KEY:
            return jsonify({
                "success": False,
                "error": t('api.zepApiKeyMissing')
            }), 500
        
        enrich = request.args.get('enrich', 'true').lower() == 'true'
        
        reader = ZepEntityReader()
        entities = reader.get_entities_by_type(
            graph_id=graph_id,
            entity_type=entity_type,
            enrich_with_edges=enrich
        )
        
        return jsonify({
            "success": True,
            "data": {
                "entity_type": entity_type,
                "count": len(entities),
                "entities": [e.to_dict() for e in entities]
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to fetch entities: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Simulation Management Endpoints ==============

@simulation_bp.route('/create', methods=['POST'])
def create_simulation():
    """
    Create a new simulation instance
    
    Note: max_rounds and parameters are intelligently synthesized by LLM
    
    Request (JSON):
        {
            "project_id": "proj_xxxx",      // Required
            "graph_id": "graph_xxxx",       // Optional, inferred from project if omitted
            "enable_twitter": true,          // Optional, default true
            "enable_reddit": true            // Optional, default true
        }
    
    Response:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "project_id": "proj_xxxx",
                "graph_id": "mirofish_xxxx",
                "status": "created",
                "enable_twitter": true,
                "enable_reddit": true,
                "created_at": "2025-12-01T10:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        project_id = data.get('project_id')
        if not project_id:
            return jsonify({
                "success": False,
                "error": t('api.requireProjectId')
            }), 400
        
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": t('api.projectNotFound', id=project_id)
            }), 404
        
        graph_id = data.get('graph_id') or project.graph_id
        if not graph_id:
            return jsonify({
                "success": False,
                "error": t('api.graphNotBuilt')
            }), 400
        
        manager = SimulationManager()
        state = manager.create_simulation(
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=data.get('enable_twitter', True),
            enable_reddit=data.get('enable_reddit', True),
        )
        
        return jsonify({
            "success": True,
            "data": state.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Failed to create simulation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def _check_simulation_prepared(simulation_id: str) -> tuple:
    """
    Check whether simulation environment is already prepared
    
    Verification criteria:
    1. state.json exists with status "ready"
    2. Required files exist: reddit_profiles.json, twitter_profiles.csv, simulation_config.json
    
    Note: runner scripts reside in scripts/ and are not copied to runs directory
    
    Args:
        simulation_id: Simulation identifier
        
    Returns:
        (is_prepared: bool, info: dict)
    """
    import os
    from ..config import Config
    
    simulation_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
    
    # Check if directory exists
    if not os.path.exists(simulation_dir):
        return False, {"reason": "Simulation directory does not exist"}
    
    # Required artifacts (excluding scripts which reside in scripts/)
    required_files = [
        "state.json",
        "simulation_config.json",
        "reddit_profiles.json",
        "twitter_profiles.csv"
    ]
    
    # Check file existence
    existing_files = []
    missing_files = []
    for f in required_files:
        file_path = os.path.join(simulation_dir, f)
        if os.path.exists(file_path):
            existing_files.append(f)
        else:
            missing_files.append(f)
    
    if missing_files:
        return False, {
            "reason": "Missing required simulation files",
            "missing_files": missing_files,
            "existing_files": existing_files
        }
    
    # Check state in state.json
    state_file = os.path.join(simulation_dir, "state.json")
    try:
        import json
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        status = state_data.get("status", "")
        config_generated = state_data.get("config_generated", False)
        
        # Detailed logging
        logger.debug(f"Checking simulation prepare status: {simulation_id}, status={status}, config_generated={config_generated}")
        
        # If config_generated=True and files exist, mark prepared
        # The following states indicate preparation is complete:
        # - ready: preparation complete, ready to run
        # - preparing: complete if config_generated=True
        # - running: active simulation indicates prepared
        # - completed: finished simulation indicates prepared
        # - stopped: stopped simulation indicates prepared
        # - failed: execution failed (preparation completed)
        prepared_statuses = ["ready", "preparing", "running", "completed", "stopped", "failed"]
        if status in prepared_statuses and config_generated:
            # Retrieve file statistics
            profiles_file = os.path.join(simulation_dir, "reddit_profiles.json")
            config_file = os.path.join(simulation_dir, "simulation_config.json")
            
            profiles_count = 0
            if os.path.exists(profiles_file):
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    profiles_count = len(profiles_data) if isinstance(profiles_data, list) else 0
            
            # Auto-transition status preparing -> ready when artifacts exist
            if status == "preparing":
                try:
                    state_data["status"] = "ready"
                    from datetime import datetime
                    state_data["updated_at"] = datetime.now().isoformat()
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump(state_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Auto-transitioning simulation status: {simulation_id} preparing -> ready")
                    status = "ready"
                except Exception as e:
                    logger.warning(f"Failed to auto-update status: {e}")
            
            logger.info(f"Simulation {simulation_id} status: prepared (status={status}, config_generated={config_generated})")
            return True, {
                "status": status,
                "entities_count": state_data.get("entities_count", 0),
                "profiles_count": profiles_count,
                "entity_types": state_data.get("entity_types", []),
                "config_generated": config_generated,
                "created_at": state_data.get("created_at"),
                "updated_at": state_data.get("updated_at"),
                "existing_files": existing_files
            }
        else:
            logger.warning(f"Simulation {simulation_id} status: not prepared (status={status}, config_generated={config_generated})")
            return False, {
                "reason": f"Status not in ready list or config_generated is false: status={status}, config_generated={config_generated}",
                "status": status,
                "config_generated": config_generated
            }
            
    except Exception as e:
        return False, {"reason": f"Failed to read state file: {str(e)}"}


@simulation_bp.route('/prepare', methods=['POST'])
def prepare_simulation():
    """
    Prepare simulation environment (async task, LLM generates all configs)
    
    Long-running asynchronous operation returning task_id immediately,
    poll GET /api/simulation/prepare/status for progress
    
    Features:
    - Automatically detects completed preparation to avoid redundant generation
    - Returns existing artifacts immediately if already prepared
    - Supports force regeneration (force_regenerate=true)
    
    Workflow steps:
    1. Verify whether existing artifacts are valid
    2. Ingest and filter entities from Zep knowledge graph
    3. Generate OASIS agent profiles with retry logic
    4. LLM generates simulation configuration with retry logic
    5. Persist simulation configuration and parameters
    
    Request (JSON):
        {
            "simulation_id": "sim_xxxx",                   // Required simulation ID
            "entity_types": ["Student", "PublicFigure"],  // Optional entity types
            "use_llm_for_profiles": true,                 // Optional: generate rich LLM personas
            "parallel_profile_count": 5,                  // Optional: concurrency level, default 5
            "force_regenerate": false                     // Optional: force regeneration, default false
        }
    
    Response:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",           // Returned when new task is created
                "status": "preparing|ready",
                "message": "Preparation task launched | Artifacts already prepared",
                "already_prepared": true|false    // Whether already prepared
            }
        }
    """
    import threading
    import os
    from ..models.task import TaskManager, TaskStatus
    from ..config import Config
    
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400
        
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": t('api.simulationNotFound', id=simulation_id)
            }), 404
        
        # Check if force regeneration requested
        force_regenerate = data.get('force_regenerate', False)
        logger.info(f"Processing /prepare request: simulation_id={simulation_id}, force_regenerate={force_regenerate}")
        
        # Check if already prepared to avoid duplicate work
        if not force_regenerate:
            logger.debug(f"Checking if simulation {simulation_id} is already prepared...")
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            logger.debug(f"Check result: is_prepared={is_prepared}, prepare_info={prepare_info}")
            if is_prepared:
                logger.info(f"Simulation {simulation_id} is already prepared, skipping duplicate generation")
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "message": t('api.alreadyPrepared'),
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                })
            else:
                logger.info(f"Simulation {simulation_id} is not prepared, launching preparation task")
        
        # Retrieve required project metadata
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": t('api.projectNotFound', id=state.project_id)
            }), 404
        
        # Retrieve simulation requirement scenario
        simulation_requirement = project.simulation_requirement or ""
        if not simulation_requirement:
            return jsonify({
                "success": False,
                "error": t('api.projectMissingRequirement')
            }), 400
        
        # Retrieve source document text
        document_text = ProjectManager.get_extracted_text(state.project_id) or ""
        
        entity_types_list = data.get('entity_types')
        use_llm_for_profiles = data.get('use_llm_for_profiles', True)
        parallel_profile_count = data.get('parallel_profile_count', 5)
        
        # ========== Synchronously fetch entity count before spawning worker ==========
        # Allows UI to immediately display expected agent total
        try:
            logger.info(f"Synchronously fetching entity count: graph_id={state.graph_id}")
            reader = ZepEntityReader()
            # Rapidly count entities without resolving edges
            filtered_preview = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=entity_types_list,
                enrich_with_edges=False  # Skip edge resolution for speed
            )
            # Save entity count to state for immediate UI polling
            state.entities_count = filtered_preview.filtered_count
            state.entity_types = list(filtered_preview.entity_types)
            logger.info(f"Expected entity count: {filtered_preview.filtered_count}, types: {filtered_preview.entity_types}")
        except Exception as e:
            logger.warning(f"Synchronous entity count fetch failed (will retry in worker): {e}")
            # Non-fatal: worker task will re-fetch if needed
        
        # Create background task
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="simulation_prepare",
            metadata={
                "simulation_id": simulation_id,
                "project_id": state.project_id
            }
        )
        
        # Update simulation state with pre-fetched counts
        state.status = SimulationStatus.PREPARING
        manager._save_simulation_state(state)
        
        # Capture locale before spawning background thread
        current_locale = get_locale()

        # Define background worker task
        def run_prepare():
            set_locale(current_locale)
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message=t('progress.startPreparingEnv')
                )
                
                # Execute simulation preparation with progress callback
                # Track stage progress details
                stage_details = {}
                
                def progress_callback(stage, progress, message, **kwargs):
                    # Calculate cumulative progress
                    stage_weights = {
                        "reading": (0, 20),           # 0-20%
                        "generating_profiles": (20, 70),  # 20-70%
                        "generating_config": (70, 90),    # 70-90%
                        "copying_scripts": (90, 100)       # 90-100%
                    }
                    
                    start, end = stage_weights.get(stage, (0, 100))
                    current_progress = int(start + (end - start) * progress / 100)
                    
                    # Build detailed progress mapping
                    stage_names = {
                        "reading": t('progress.readingGraphEntities'),
                        "generating_profiles": t('progress.generatingProfiles'),
                        "generating_config": t('progress.generatingSimConfig'),
                        "copying_scripts": t('progress.preparingScripts')
                    }
                    
                    stage_index = list(stage_weights.keys()).index(stage) + 1 if stage in stage_weights else 1
                    total_stages = len(stage_weights)
                    
                    # Update stage details
                    stage_details[stage] = {
                        "stage_name": stage_names.get(stage, stage),
                        "stage_progress": progress,
                        "current": kwargs.get("current", 0),
                        "total": kwargs.get("total", 0),
                        "item_name": kwargs.get("item_name", "")
                    }
                    
                    # Build detailed progress mapping
                    detail = stage_details[stage]
                    progress_detail_data = {
                        "current_stage": stage,
                        "current_stage_name": stage_names.get(stage, stage),
                        "stage_index": stage_index,
                        "total_stages": total_stages,
                        "stage_progress": progress,
                        "current_item": detail["current"],
                        "total_items": detail["total"],
                        "item_description": message
                    }
                    
                    # Construct concise message
                    if detail["total"] > 0:
                        detailed_message = (
                            f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: "
                            f"{detail['current']}/{detail['total']} - {message}"
                        )
                    else:
                        detailed_message = f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: {message}"
                    
                    task_manager.update_task(
                        task_id,
                        progress=current_progress,
                        message=detailed_message,
                        progress_detail=progress_detail_data
                    )
                
                result_state = manager.prepare_simulation(
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                    document_text=document_text,
                    defined_entity_types=entity_types_list,
                    use_llm_for_profiles=use_llm_for_profiles,
                    progress_callback=progress_callback,
                    parallel_profile_count=parallel_profile_count
                )

                if result_state.status == SimulationStatus.FAILED:
                    task_manager.fail_task(
                        task_id,
                        result_state.error or "Simulation preparation failed"
                    )
                else:
                    task_manager.complete_task(
                        task_id,
                        result=result_state.to_simple_dict()
                    )
                
            except Exception as e:
                logger.error(f"Failed to prepare simulation: {str(e)}")
                task_manager.fail_task(task_id, str(e))
                
                # Set simulation status to failed
                state = manager.get_simulation(simulation_id)
                if state:
                    state.status = SimulationStatus.FAILED
                    state.error = str(e)
                    manager._save_simulation_state(state)
        
        # Launch worker thread
        thread = threading.Thread(target=run_prepare, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "task_id": task_id,
                "status": "preparing",
                "message": t('api.prepareStarted'),
                "already_prepared": False,
                "expected_entities_count": state.entities_count,  # Expected agent total
                "entity_types": state.entity_types  # Entity types list
            }
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
        
    except Exception as e:
        logger.error(f"Failed to start preparation task: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/prepare/status', methods=['POST'])
def get_prepare_status():
    """
    Query preparation task progress
    
    Supports two query modes:
    1. Query active progress via task_id
    2. Verify completed preparation via simulation_id
    
    Request (JSON):
        {
            "task_id": "task_xxxx",          // Optional task_id returned by prepare
            "simulation_id": "sim_xxxx"      // Optional simulation ID to check existing status
        }
    
    Response:
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|ready",
                "progress": 45,
                "message": "...",
                "already_prepared": true|false,  // Whether preparation is complete
                "prepare_info": {...}            // Detailed metadata when prepared
            }
        }
    """
    from ..models.task import TaskManager
    
    try:
        data = request.get_json() or {}
        
        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')
        
        # If simulation_id is provided, check if already prepared
        if simulation_id:
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            if is_prepared:
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "progress": 100,
                        "message": t('api.alreadyPrepared'),
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                })
        
        # If no task_id provided, return error
        if not task_id:
            if simulation_id:
                # simulation_id provided but not yet prepared
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "not_started",
                        "progress": 0,
                        "message": t('api.notStartedPrepare'),
                        "already_prepared": False
                    }
                })
            return jsonify({
                "success": False,
                "error": t('api.requireTaskOrSimId')
            }), 400
        
        task_manager = TaskManager()
        task = task_manager.get_task(task_id)
        
        if not task:
            # Task not found; verify if already prepared by simulation_id
            if simulation_id:
                is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
                if is_prepared:
                    return jsonify({
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "task_id": task_id,
                            "status": "ready",
                            "progress": 100,
                            "message": t('api.taskCompletedPrepared'),
                            "already_prepared": True,
                            "prepare_info": prepare_info
                        }
                    })
            
            return jsonify({
                "success": False,
                "error": t('api.taskNotFound', id=task_id)
            }), 404
        
        task_dict = task.to_dict()
        task_dict["already_prepared"] = False
        
        return jsonify({
            "success": True,
            "data": task_dict
        })
        
    except Exception as e:
        logger.error(f"Failed to query task status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@simulation_bp.route('/<simulation_id>', methods=['GET'])
def get_simulation(simulation_id: str):
    """Get simulation status"""
    try:
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": t('api.simulationNotFound', id=simulation_id)
            }), 404
        
        result = state.to_dict()
        
        # Attach execution instructions if simulation is prepared
        if state.status == SimulationStatus.READY:
            result["run_instructions"] = manager.get_run_instructions(simulation_id)
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Failed to get simulation status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/list', methods=['GET'])
def list_simulations():
    """
    List all simulations
    
    Query parameters:
        project_id: Filter by project ID (optional)
    """
    try:
        project_id = request.args.get('project_id')
        
        manager = SimulationManager()
        simulations = manager.list_simulations(project_id=project_id)
        
        return jsonify({
            "success": True,
            "data": [s.to_dict() for s in simulations],
            "count": len(simulations)
        })
        
    except Exception as e:
        logger.error(f"Failed to list simulations: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def _get_report_id_for_simulation(simulation_id: str) -> str:
    """
    Get latest report_id for simulation
    
    Scans reports directory to match simulation_id,
    returning newest report sorted by creation timestamp.
    
    Args:
        simulation_id: Simulation identifier
        
    Returns:
        report_id or None
    """
    import json
    from datetime import datetime
    
    # Reports directory path: uploads/reports
    # Navigate to uploads directory
    reports_dir = os.path.join(os.path.dirname(__file__), '../../uploads/reports')
    if not os.path.exists(reports_dir):
        return None
    
    matching_reports = []
    
    try:
        for report_folder in os.listdir(reports_dir):
            report_path = os.path.join(reports_dir, report_folder)
            if not os.path.isdir(report_path):
                continue
            
            meta_file = os.path.join(report_path, "meta.json")
            if not os.path.exists(meta_file):
                continue
            
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                if meta.get("simulation_id") == simulation_id:
                    matching_reports.append({
                        "report_id": meta.get("report_id"),
                        "created_at": meta.get("created_at", ""),
                        "status": meta.get("status", "")
                    })
            except Exception:
                continue
        
        if not matching_reports:
            return None
        
        # Sort descending by creation timestamp and return newest
        matching_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return matching_reports[0].get("report_id")
        
    except Exception as e:
        logger.warning(f"Failed to lookup report for simulation {simulation_id}: {e}")
        return None


@simulation_bp.route('/history', methods=['GET'])
def get_simulation_history():
    """
    Get historical simulation list with project details
    
    Returns rich simulation entries including scenario and project details for homepage display
    
    Query parameters:
        limit: Limit count (default 20)
    
    Response:
        {
            "success": true,
            "data": [
                {
                    "simulation_id": "sim_xxxx",
                    "project_id": "proj_xxxx",
                    "project_name": "Simulated Market Analysis",
                    "simulation_requirement": "Simulated product launch scenario...",
                    "status": "completed",
                    "entities_count": 68,
                    "profiles_count": 68,
                    "entity_types": ["Student", "Professor", ...],
                    "created_at": "2024-12-10",
                    "updated_at": "2024-12-10",
                    "total_rounds": 120,
                    "current_round": 120,
                    "report_id": "report_xxxx",
                    "version": "v1.0.2"
                },
                ...
            ],
            "count": 7
        }
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        
        manager = SimulationManager()
        simulations = manager.list_simulations()[:limit]
        
        # Enrich simulation data from run state
        enriched_simulations = []
        for sim in simulations:
            sim_dict = sim.to_dict()
            
            # Read scenario requirement from simulation_config.json
            config = manager.get_simulation_config(sim.simulation_id)
            if config:
                sim_dict["simulation_requirement"] = config.get("simulation_requirement", "")
                time_config = config.get("time_config", {})
                sim_dict["total_simulation_hours"] = time_config.get("total_simulation_hours", 0)
                # Recommended rounds fallback
                recommended_rounds = int(
                    time_config.get("total_simulation_hours", 0) * 60 / 
                    max(time_config.get("minutes_per_round", 60), 1)
                )
            else:
                sim_dict["simulation_requirement"] = ""
                sim_dict["total_simulation_hours"] = 0
                recommended_rounds = 0
            
            # Read total rounds from run_state.json
            run_state = SimulationRunner.get_run_state(sim.simulation_id)
            if run_state:
                sim_dict["current_round"] = run_state.current_round
                sim_dict["runner_status"] = run_state.runner_status.value
                # Use total_rounds from run state, fallback to recommended
                sim_dict["total_rounds"] = run_state.total_rounds if run_state.total_rounds > 0 else recommended_rounds
            else:
                sim_dict["current_round"] = 0
                sim_dict["runner_status"] = "idle"
                sim_dict["total_rounds"] = recommended_rounds
            
            # Get up to 3 associated project files
            project = ProjectManager.get_project(sim.project_id)
            if project and hasattr(project, 'files') and project.files:
                sim_dict["files"] = [
                    {"filename": f.get("filename", "unknown_file")} 
                    for f in project.files[:3]
                ]
            else:
                sim_dict["files"] = []
            
            # Lookup latest associated report ID
            sim_dict["report_id"] = _get_report_id_for_simulation(sim.simulation_id)
            
            # Append version number
            sim_dict["version"] = "v1.0.2"
            
            # Format display date
            try:
                created_date = sim_dict.get("created_at", "")[:10]
                sim_dict["created_date"] = created_date
            except:
                sim_dict["created_date"] = ""
            
            enriched_simulations.append(sim_dict)
        
        return jsonify({
            "success": True,
            "data": enriched_simulations,
            "count": len(enriched_simulations)
        })
        
    except Exception as e:
        logger.error(f"Failed to get simulation history: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/profiles', methods=['GET'])
def get_simulation_profiles(simulation_id: str):
    """
    Get agent profiles for simulation
    
    Query parameters:
        platform: Platform filter (reddit/twitter, default reddit)
    """
    try:
        platform = request.args.get('platform') or _get_default_platform(simulation_id)

        manager = SimulationManager()
        profiles = manager.get_profiles(simulation_id, platform=platform)
        
        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "count": len(profiles),
                "profiles": profiles
            }
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
        
    except Exception as e:
        logger.error(f"Failed to get profiles: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/profiles/realtime', methods=['GET'])
def get_simulation_profiles_realtime(simulation_id: str):
    """
    Real-time agent profile inspection during generation
    
    Difference from standard /profiles:
    - Reads disk file directly without locking SimulationManager
    - Suited for live progress inspection during generation
    - Includes metadata like mtime and is_generating flag
    
    Query parameters:
        platform: Platform filter (reddit/twitter, default reddit)
    
    Response:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "platform": "reddit",
                "count": 15,
                "total_expected": 93,  // Expected total count
                "is_generating": true,  // Whether generation is underway
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "profiles": [...]
            }
        }
    """
    import json
    import csv
    from datetime import datetime
    
    try:
        platform = request.args.get('platform') or _get_default_platform(simulation_id)

        # Get simulation directory
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        
        if not os.path.exists(sim_dir):
            return jsonify({
                "success": False,
                "error": t('api.simulationNotFound', id=simulation_id)
            }), 404
        
        # Determine profile file path
        if platform == "reddit":
            profiles_file = os.path.join(sim_dir, "reddit_profiles.json")
        else:
            profiles_file = os.path.join(sim_dir, "twitter_profiles.csv")
        
        # Check file existence
        file_exists = os.path.exists(profiles_file)
        profiles = []
        file_modified_at = None
        
        if file_exists:
            # Retrieve file modification timestamp
            file_stat = os.stat(profiles_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            
            try:
                if platform == "reddit":
                    with open(profiles_file, 'r', encoding='utf-8') as f:
                        profiles = json.load(f)
                else:
                    with open(profiles_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        profiles = list(reader)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to read profiles file (write in progress): {e}")
                profiles = []
        
        # Check if generation is underway via state.json
        is_generating = False
        total_expected = None
        status = None
        error = None
        
        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    is_generating = status == "preparing"
                    total_expected = state_data.get("entities_count")
                    error = state_data.get("error")
            except Exception:
                pass
        
        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "platform": platform,
                "count": len(profiles),
                "total_expected": total_expected,
                "is_generating": is_generating,
                "status": status,
                "error": error,
                "file_exists": file_exists,
                "file_modified_at": file_modified_at,
                "profiles": profiles
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get profiles in real-time: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config/realtime', methods=['GET'])
def get_simulation_config_realtime(simulation_id: str):
    """
    Real-time configuration inspection during generation
    
    Difference from standard /config:
    - Reads disk file directly without locking SimulationManager
    - Suited for live progress inspection during generation
    - Includes metadata like mtime and is_generating flag
    - Returns partial data even before generation finishes
    
    Response:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "is_generating": true,  // Whether generation is underway
                "generation_stage": "generating_config",  // Current generation stage
                "config": {...}  // Configuration data if available
            }
        }
    """
    import json
    from datetime import datetime
    
    try:
        # Get simulation directory
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        
        if not os.path.exists(sim_dir):
            return jsonify({
                "success": False,
                "error": t('api.simulationNotFound', id=simulation_id)
            }), 404
        
        # Configuration file path
        config_file = os.path.join(sim_dir, "simulation_config.json")
        
        # Check file existence
        file_exists = os.path.exists(config_file)
        config = None
        file_modified_at = None
        
        if file_exists:
            # Retrieve file modification timestamp
            file_stat = os.stat(config_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to read config file (write in progress): {e}")
                config = None
        
        # Check if generation is underway via state.json
        is_generating = False
        generation_stage = None
        status = None
        error = None
        profiles_generated = False
        config_generated = False
        
        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    error = state_data.get("error")
                    is_generating = status == "preparing"
                    profiles_generated = state_data.get("profiles_generated", False)
                    config_generated = state_data.get("config_generated", False)
                    
                    # Determine current stage
                    if is_generating:
                        if profiles_generated:
                            generation_stage = "generating_config"
                        else:
                            generation_stage = "generating_profiles"
                    elif status == "ready":
                        generation_stage = "completed"
                    elif status == "failed":
                        generation_stage = "failed"
            except Exception:
                pass
        
        # Build return payload
        response_data = {
            "simulation_id": simulation_id,
            "file_exists": file_exists,
            "file_modified_at": file_modified_at,
            "is_generating": is_generating,
            "status": status,
            "error": error,
            "generation_stage": generation_stage,
            "profiles_generated": profiles_generated,
            "config_generated": config_generated,
            "config": config
        }
        
        # Extract key statistics if config exists
        if config:
            response_data["summary"] = {
                "total_agents": len(config.get("agent_configs", [])),
                "simulation_hours": config.get("time_config", {}).get("total_simulation_hours"),
                "initial_posts_count": len(config.get("event_config", {}).get("initial_posts", [])),
                "hot_topics_count": len(config.get("event_config", {}).get("hot_topics", [])),
                "has_twitter_config": "twitter_config" in config,
                "has_reddit_config": "reddit_config" in config,
                "generated_at": config.get("generated_at"),
                "llm_model": config.get("llm_model")
            }
        
        return jsonify({
            "success": True,
            "data": response_data
        })
        
    except Exception as e:
        logger.error(f"Failed to get config in real-time: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config', methods=['GET'])
def get_simulation_config(simulation_id: str):
    """
    Get simulation configuration synthesized by LLM
    
    Returns:
        - time_config: Time parameters (hours, rounds, peak intervals)
        - agent_configs: Per-agent activity parameters (frequency, stances)
        - event_config: Initial seed posts and trigger topics
        - platform_configs: Platform settings
        - generation_reasoning: LLM rationale and simulation dynamics rationale
    """
    try:
        manager = SimulationManager()
        config = manager.get_simulation_config(simulation_id)
        
        if not config:
            return jsonify({
                "success": False,
                "error": t('api.configNotFound')
            }), 404
        
        return jsonify({
            "success": True,
            "data": config
        })
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config/download', methods=['GET'])
def download_simulation_config(simulation_id: str):
    """Download simulation configuration JSON"""
    try:
        manager = SimulationManager()
        sim_dir = manager._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            return jsonify({
                "success": False,
                "error": t('api.configFileNotFound')
            }), 404
        
        return send_file(
            config_path,
            as_attachment=True,
            download_name="simulation_config.json"
        )
        
    except Exception as e:
        logger.error(f"Failed to download config: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/script/<script_name>/download', methods=['GET'])
def download_simulation_script(script_name: str):
    """
    Download simulation runner script (from scripts/)
    
    script_name options:
        - run_twitter_simulation.py
        - run_reddit_simulation.py
        - run_parallel_simulation.py
        - action_logger.py
    """
    try:
        # Scripts reside in scripts/ directory
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
        
        # Validate script name
        allowed_scripts = [
            "run_twitter_simulation.py",
            "run_reddit_simulation.py", 
            "run_parallel_simulation.py",
            "action_logger.py"
        ]
        
        if script_name not in allowed_scripts:
            return jsonify({
                "success": False,
                "error": t('api.unknownScript', name=script_name, allowed=allowed_scripts)
            }), 400
        
        script_path = os.path.join(scripts_dir, script_name)
        
        if not os.path.exists(script_path):
            return jsonify({
                "success": False,
                "error": t('api.scriptFileNotFound', name=script_name)
            }), 404
        
        return send_file(
            script_path,
            as_attachment=True,
            download_name=script_name
        )
        
    except Exception as e:
        logger.error(f"Failed to download script: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Standalone Profile Generation Endpoints ==============

@simulation_bp.route('/generate-profiles', methods=['POST'])
def generate_profiles():
    """
    Generate OASIS Agent profiles directly from graph without creating a simulation
    
    Request (JSON):
        {
            "graph_id": "graph_xxxx",        // Required
            "entity_types": ["Persona"],      // Optional
            "use_llm": true,                  // Optional
            "platform": "reddit"              // Optional
        }
    """
    try:
        data = request.get_json() or {}
        
        graph_id = data.get('graph_id')
        if not graph_id:
            return jsonify({
                "success": False,
                "error": t('api.requireGraphId')
            }), 400
        
        entity_types = data.get('entity_types')
        use_llm = data.get('use_llm', True)
        platform = data.get('platform', 'reddit')
        
        reader = ZepEntityReader()
        filtered = reader.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=entity_types,
            enrich_with_edges=True
        )
        
        if filtered.filtered_count == 0:
            return jsonify({
                "success": False,
                "error": t('api.noMatchingEntities')
            }), 400
        
        generator = OasisProfileGenerator()
        profiles = generator.generate_profiles_from_entities(
            entities=filtered.entities,
            use_llm=use_llm
        )
        
        if platform == "reddit":
            profiles_data = [p.to_reddit_format() for p in profiles]
        elif platform == "twitter":
            profiles_data = [p.to_twitter_format() for p in profiles]
        else:
            profiles_data = [p.to_dict() for p in profiles]
        
        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "entity_types": list(filtered.entity_types),
                "count": len(profiles_data),
                "profiles": profiles_data
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to generate profile: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Simulation Execution Controls ==============

@simulation_bp.route('/start', methods=['POST'])
def start_simulation():
    """
    Start simulation run

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",          // Required simulation ID
            "platform": "parallel",                // Optional: twitter / reddit / parallel (default)
            "max_rounds": 100,                     // Optional: max rounds to truncate long runs
            "enable_graph_memory_update": false,   // Optional: stream activities to Zep graph
            "force": false                         // Optional: force restart (stops existing and clears logs)
        }

    Regarding the force parameter:
        - When true, active or completed simulations are terminated and previous logs cleared
        - Cleans up: run_state.json, actions.jsonl, simulation.log, etc.
        - Retains simulation_config.json and profile artifacts
        - Useful when restarting a simulation from scratch

    Regarding enable_graph_memory_update:
        - When true, agent activities (posts, comments, likes) stream to Zep graph memory
        - Allows the knowledge graph to retain simulated events for Q&A and analysis
        - Requires project to have a valid graph_id
        - Uses batched updates to optimize API efficiency

    Response:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "process_pid": 12345,
                "twitter_running": true,
                "reddit_running": true,
                "started_at": "2025-12-01T10:00:00",
                "graph_memory_update_enabled": true,  // Whether graph streaming is active
                "force_restarted": true               // Whether run was force restarted
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400

        platform = data.get('platform', 'parallel')
        max_rounds = data.get('max_rounds')  # Optional: max rounds
        enable_graph_memory_update = data.get('enable_graph_memory_update', False)  # Optional: graph streaming
        force = data.get('force', False)  # Optional: force restart
        if not isinstance(enable_graph_memory_update, bool):
            return jsonify({
                "success": False,
                "error": "enable_graph_memory_update must be a JSON boolean",
            }), 400
        if not isinstance(force, bool):
            return jsonify({
                "success": False,
                "error": "force must be a JSON boolean",
            }), 400

        # Validate max_rounds parameter
        if max_rounds is not None:
            try:
                max_rounds = int(max_rounds)
                if max_rounds <= 0:
                    return jsonify({
                        "success": False,
                        "error": t('api.maxRoundsPositive')
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "error": t('api.maxRoundsInvalid')
                }), 400

        if platform not in ['twitter', 'reddit', 'parallel']:
            return jsonify({
                "success": False,
                "error": t('api.invalidPlatform', platform=platform)
            }), 400

        # Check if simulation is ready
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify({
                "success": False,
                "error": t('api.simulationNotFound', id=simulation_id)
            }), 404

        force_restarted = False
        
        # Verify preparation completeness
        is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
        if not is_prepared:
            return jsonify({
                "success": False,
                "error": t('api.simNotReady', status=state.status.value)
            }), 400

        # Check if an active runner or updater needs finalization/cleanup
        run_state = SimulationRunner.get_run_state(simulation_id)
        updater = ZepGraphMemoryManager.get_updater(simulation_id)
        active_statuses = {
            RunnerStatus.STARTING,
            RunnerStatus.RUNNING,
            RunnerStatus.PAUSED,
            RunnerStatus.STOPPING,
        }
        is_active = bool(
            (run_state is not None and run_state.runner_status in active_statuses)
            or updater is not None
        )

        if is_active or force:
            if is_active and not force:
                return jsonify({
                    "success": False,
                    "error": t('api.simRunningForceHint')
                }), 400

            logger.info(f"Force mode or active run cleanup: terminating {simulation_id}")
            try:
                stopped = SimulationRunner.stop_simulation(simulation_id)
            except SimulationStopPending as error:
                return jsonify({
                    "success": False,
                    "pending": True,
                    "error": str(error),
                }), 409
            except Exception as error:
                logger.warning(f"Stop simulation encountered error during force reset: {error}")
                if updater is not None:
                    try:
                        ZepGraphMemoryManager.stop_updater(simulation_id, force=True)
                    except Exception as e:
                        logger.warning(f"Forced updater stop ignored exception: {e}")

            # Clean up simulation logs and reset in-memory runner state
            logger.info(f"Cleaning logs and run state for {simulation_id}")
            cleanup_result = SimulationRunner.cleanup_simulation_logs(simulation_id)
            if not cleanup_result.get("success"):
                logger.warning(f"Cleanup warnings for {simulation_id}: {cleanup_result.get('errors')}")
            force_restarted = True

        state.status = SimulationStatus.READY
        manager._save_simulation_state(state)
        
        # Retrieve graph_id for graph streaming
        graph_id = None
        if enable_graph_memory_update:
            # The project is authoritative. A graph ID copied into an older
            # simulation can outlive a project reset/rebuild and must not be
            # used to resurrect writes to a deleted graph.
            project = ProjectManager.get_project(state.project_id)
            graph_id = project.graph_id if project else None
            if not graph_id:
                return jsonify({
                    "success": False,
                    "error": t('api.graphIdRequiredForMemory')
                }), 400

        graph_guard = (
            graph_lifecycle_lock(graph_id)
            if enable_graph_memory_update
            else nullcontext()
        )
        with graph_guard:
            if enable_graph_memory_update:
                # Re-read both references under the same per-graph lock used
                # by reset/delete. Keep the lock through updater creation in
                # start_simulation so check -> claim is atomic.
                refreshed_state = manager.get_simulation(simulation_id)
                refreshed_project = (
                    ProjectManager.get_project(refreshed_state.project_id)
                    if refreshed_state
                    else None
                )
                current_graph_id = (
                    refreshed_project.graph_id if refreshed_project else None
                )
                if current_graph_id != graph_id:
                    return jsonify({
                        "success": False,
                        "error": (
                            "The project graph changed while the simulation "
                            "was starting; retry after refreshing the project"
                        ),
                    }), 409
                if (
                    refreshed_state.graph_id
                    and refreshed_state.graph_id != current_graph_id
                ):
                    return jsonify({
                        "success": False,
                        "error": (
                            "The simulation references an older graph; "
                            "prepare it again before enabling graph memory"
                        ),
                    }), 409
                active_reports = get_graph_readers(graph_id)
                if active_reports:
                    return jsonify({
                        "success": False,
                        "error": (
                            "A report is currently reading this graph; wait "
                            "for report generation to finish before enabling "
                            "graph memory updates"
                        ),
                        "active_reports": active_reports,
                    }), 409
                state = refreshed_state
                logger.info(
                    "Enabling graph memory streaming: simulation_id=%s, graph_id=%s",
                    simulation_id,
                    graph_id,
                )

            # Launch simulation. Graph updater hold graph_guard
            # until process resources are published.
            run_state = SimulationRunner.start_simulation(
                simulation_id=simulation_id,
                platform=platform,
                max_rounds=max_rounds,
                enable_graph_memory_update=enable_graph_memory_update,
                graph_id=graph_id
            )
        
        response_data = run_state.to_dict()
        if max_rounds:
            response_data['max_rounds_applied'] = max_rounds
        response_data['graph_memory_update_enabled'] = enable_graph_memory_update
        response_data['force_restarted'] = force_restarted
        if enable_graph_memory_update:
            response_data['graph_id'] = graph_id
        
        return jsonify({
            "success": True,
            "data": response_data
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Failed to start simulation: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/stop', methods=['POST'])
def stop_simulation():
    """
    Stop simulation
    
    Request (JSON):
        {
            "simulation_id": "sim_xxxx"  // Required simulation ID
        }
    
    Response:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "stopped",
                "completed_at": "2025-12-01T12:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400
        
        run_state = SimulationRunner.stop_simulation(simulation_id)
        
        # Update simulation state
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.STOPPED
            state.error = None
            manager._save_simulation_state(state)
        
        return jsonify({
            "success": True,
            "data": run_state.to_dict()
        })

    except SimulationStopPending as e:
        return jsonify({
            "success": False,
            "pending": True,
            "error": str(e),
        }), 202

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Failed to stop simulation: {str(e)}")
        simulation_id = (request.get_json(silent=True) or {}).get('simulation_id')
        if simulation_id:
            manager = SimulationManager()
            state = manager.get_simulation(simulation_id)
            if state:
                state.status = SimulationStatus.FAILED
                state.error = str(e)
                manager._save_simulation_state(state)
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Real-time Status Monitoring Endpoints ==============

@simulation_bp.route('/<simulation_id>/run-status', methods=['GET'])
def get_run_status(simulation_id: str):
    """
    Get real-time simulation run status for UI polling
    
    Response:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "current_round": 5,
                "total_rounds": 144,
                "progress_percent": 3.5,
                "simulated_hours": 2,
                "total_simulation_hours": 72,
                "twitter_running": true,
                "reddit_running": true,
                "twitter_actions_count": 150,
                "reddit_actions_count": 200,
                "total_actions_count": 350,
                "started_at": "2025-12-01T10:00:00",
                "updated_at": "2025-12-01T10:30:00"
            }
        }
    """
    try:
        run_state = SimulationRunner.get_run_state(simulation_id)
        
        if not run_state:
            return jsonify({
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "runner_status": "idle",
                    "current_round": 0,
                    "total_rounds": 0,
                    "progress_percent": 0,
                    "twitter_actions_count": 0,
                    "reddit_actions_count": 0,
                    "total_actions_count": 0,
                }
            })
        
        return jsonify({
            "success": True,
            "data": run_state.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Failed to get run status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/run-status/detail', methods=['GET'])
def get_run_status_detail(simulation_id: str):
    """
    Get detailed simulation run state including recent actions
    
    Used by UI to render live activity feeds
    
    Query parameters:
        platform: Filter by platform (twitter/reddit, optional)
    
    Response:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "current_round": 5,
                ...
                "all_actions": [
                    {
                        "round_num": 5,
                        "timestamp": "2025-12-01T10:30:00",
                        "platform": "twitter",
                        "agent_id": 3,
                        "agent_name": "Agent Name",
                        "action_type": "CREATE_POST",
                        "action_args": {"content": "..."},
                        "result": null,
                        "success": true
                    },
                    ...
                ],
                "twitter_actions": [...],  # Twitter actions
                "reddit_actions": [...]    # Reddit actions
            }
        }
    """
    try:
        run_state = SimulationRunner.get_run_state(simulation_id)
        platform_filter = request.args.get('platform')
        
        if not run_state:
            return jsonify({
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "runner_status": "idle",
                    "all_actions": [],
                    "twitter_actions": [],
                    "reddit_actions": []
                }
            })
        
        # Retrieve complete action list
        all_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform=platform_filter
        )
        
        # Separate actions by platform
        twitter_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform="twitter"
        ) if not platform_filter or platform_filter == "twitter" else []
        
        reddit_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform="reddit"
        ) if not platform_filter or platform_filter == "reddit" else []
        
        # Extract actions for current round
        current_round = run_state.current_round
        recent_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform=platform_filter,
            round_num=current_round
        ) if current_round > 0 else []
        
        # Retrieve base status information
        result = run_state.to_dict()
        result["all_actions"] = [a.to_dict() for a in all_actions]
        result["twitter_actions"] = [a.to_dict() for a in twitter_actions]
        result["reddit_actions"] = [a.to_dict() for a in reddit_actions]
        result["rounds_count"] = len(run_state.rounds)
        # recent_actions displays the latest round actions across both platforms
        result["recent_actions"] = [a.to_dict() for a in recent_actions]
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Failed to get detailed status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/actions', methods=['GET'])
def get_simulation_actions(simulation_id: str):
    """
    Get agent action history with pagination
    
    Query parameters:
        limit: Limit count (default 100)
        offset: Offset (default 0)
        platform: Filter platform (twitter/reddit)
        agent_id: Filter Agent ID
        round_num: Filter round number
    
    Response:
        {
            "success": true,
            "data": {
                "count": 100,
                "actions": [...]
            }
        }
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        platform = request.args.get('platform')
        agent_id = request.args.get('agent_id', type=int)
        round_num = request.args.get('round_num', type=int)
        
        actions = SimulationRunner.get_actions(
            simulation_id=simulation_id,
            limit=limit,
            offset=offset,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num
        )
        
        return jsonify({
            "success": True,
            "data": {
                "count": len(actions),
                "actions": [a.to_dict() for a in actions]
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get action history: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/timeline', methods=['GET'])
def get_simulation_timeline(simulation_id: str):
    """
    Get simulation timeline summarized by round
    
    Used by UI to render timeline progress and round charts
    
    Query parameters:
        start_round: Starting round (default 0)
        end_round: Ending round (default all)
    
    Returns summary for each round
    """
    try:
        start_round = request.args.get('start_round', 0, type=int)
        end_round = request.args.get('end_round', type=int)
        
        timeline = SimulationRunner.get_timeline(
            simulation_id=simulation_id,
            start_round=start_round,
            end_round=end_round
        )
        
        return jsonify({
            "success": True,
            "data": {
                "rounds_count": len(timeline),
                "timeline": timeline
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get timeline: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/agent-stats', methods=['GET'])
def get_agent_stats(simulation_id: str):
    """
    Get statistics for each agent
    
    Used by UI to render agent activity leaderboards and distributions
    """
    try:
        stats = SimulationRunner.get_agent_stats(simulation_id)
        
        return jsonify({
            "success": True,
            "data": {
                "agents_count": len(stats),
                "stats": stats
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get agent stats: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Database Query Endpoints ==============

@simulation_bp.route('/<simulation_id>/posts', methods=['GET'])
def get_simulation_posts(simulation_id: str):
    """
    Get posts created during simulation
    
    Query parameters:
        platform: Platform type (twitter/reddit)
        limit: Return count (default 50)
        offset: Offset
    
    Returns post records read from platform SQLite databases
    """
    try:
        platform = request.args.get('platform') or _get_default_platform(simulation_id)
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )

        db_file = f"{platform}_simulation.db"
        db_path = os.path.join(sim_dir, db_file)
        
        if not os.path.exists(db_path):
            return jsonify({
                "success": True,
                "data": {
                    "platform": platform,
                    "count": 0,
                    "posts": [],
                    "message": t('api.dbNotExist')
                }
            })
        
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM post 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            posts = [dict(row) for row in cursor.fetchall()]
            
            cursor.execute("SELECT COUNT(*) FROM post")
            total = cursor.fetchone()[0]
            
        except sqlite3.OperationalError:
            posts = []
            total = 0
        
        conn.close()
        
        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "total": total,
                "count": len(posts),
                "posts": posts
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get posts: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/comments', methods=['GET'])
def get_simulation_comments(simulation_id: str):
    """
    Get comments created during simulation

    Query parameters:
        platform: Platform (twitter/reddit, auto-selected from configuration)
        post_id: Filter post ID (optional)
        limit: Return count
        offset: Offset
    """
    try:
        platform = request.args.get('platform') or _get_default_platform(simulation_id)
        post_id = request.args.get('post_id')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )
        
        db_path = os.path.join(sim_dir, f"{platform}_simulation.db")
        
        if not os.path.exists(db_path):
            return jsonify({
                "success": True,
                "data": {
                    "count": 0,
                    "comments": []
                }
            })
        
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if post_id:
                cursor.execute("""
                    SELECT * FROM comment 
                    WHERE post_id = ?
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (post_id, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM comment 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            comments = [dict(row) for row in cursor.fetchall()]
            
        except sqlite3.OperationalError:
            comments = []
        
        conn.close()
        
        return jsonify({
            "success": True,
            "data": {
                "count": len(comments),
                "comments": comments
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get comments: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Agent Interview Endpoints ==============

@simulation_bp.route('/interview', methods=['POST'])
def interview_agent():
    """
    Interview a single agent in running environment

    Note: requires simulation environment to be active in command loop mode

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",       // Required simulation ID
            "agent_id": 0,                     // Required agent ID
            "prompt": "What is your stance on this development?",  // Required question
            "platform": "twitter",             // Optional platform (twitter/reddit)
                                               // If omitted in parallel mode, interviews both platforms concurrently
            "timeout": 60                      // Optional timeout in seconds, default 60
        }

    Response (dual-platform mode):
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "What is your stance on this development?",
                "result": {
                    "agent_id": 0,
                    "prompt": "...",
                    "platforms": {
                        "twitter": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit": {"agent_id": 0, "response": "...", "platform": "reddit"}
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }

    Response (single platform):
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "What is your stance on this development?",
                "result": {
                    "agent_id": 0,
                    "response": "I believe that...",
                    "platform": "twitter",
                    "timestamp": "2025-12-08T10:00:00"
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        agent_id = data.get('agent_id')
        prompt = data.get('prompt')
        platform = data.get('platform')  # Optional: twitter/reddit/None
        timeout = data.get('timeout', 60)
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400
        
        if agent_id is None:
            return jsonify({
                "success": False,
                "error": t('api.requireAgentId')
            }), 400
        
        if not prompt:
            return jsonify({
                "success": False,
                "error": t('api.requirePrompt')
            }), 400
        
        # Validate platform parameter
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": t('api.invalidInterviewPlatform')
            }), 400
        
        # Verify environment alive status
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": t('api.envNotRunning')
            }), 400
        
        # Optimize prompt with instruction to prevent tool calls
        optimized_prompt = optimize_interview_prompt(prompt)
        
        result = SimulationRunner.interview_agent(
            simulation_id=simulation_id,
            agent_id=agent_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": t('api.interviewTimeout', error=str(e))
        }), 504
        
    except Exception as e:
        logger.error(f"Interview failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/batch', methods=['POST'])
def interview_agents_batch():
    """
    Batch interview multiple agents

    Note: requires simulation environment to be active

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",       // Required simulation ID
            "interviews": [                    // Required interview list
                {
                    "agent_id": 0,
                    "prompt": "What is your perspective on issue A?",
                    "platform": "twitter"      // Optional platform for this agent
                },
                {
                    "agent_id": 1,
                    "prompt": "What is your perspective on issue B?"  // Uses default if omitted
                }
            ],
            "platform": "reddit",              // Optional default platform override
                                               // When omitted in dual-platform mode, interviews both platforms
            "timeout": 120                     // Optional timeout in seconds, default 120
        }

    Response:
        {
            "success": true,
            "data": {
                "interviews_count": 2,
                "result": {
                    "interviews_count": 4,
                    "results": {
                        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit_0": {"agent_id": 0, "response": "...", "platform": "reddit"},
                        "twitter_1": {"agent_id": 1, "response": "...", "platform": "twitter"},
                        "reddit_1": {"agent_id": 1, "response": "...", "platform": "reddit"}
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        interviews = data.get('interviews')
        platform = data.get('platform')  # Optional: twitter/reddit/None
        timeout = data.get('timeout', 120)

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400

        if not interviews or not isinstance(interviews, list):
            return jsonify({
                "success": False,
                "error": t('api.requireInterviews')
            }), 400

        # Validate platform parameter
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": t('api.invalidInterviewPlatform')
            }), 400

        # Validate each interview entry
        for i, interview in enumerate(interviews):
            if 'agent_id' not in interview:
                return jsonify({
                    "success": False,
                    "error": t('api.interviewListMissingAgentId', index=i+1)
                }), 400
            if 'prompt' not in interview:
                return jsonify({
                    "success": False,
                    "error": t('api.interviewListMissingPrompt', index=i+1)
                }), 400
            # Validate per-item platform override
            item_platform = interview.get('platform')
            if item_platform and item_platform not in ("twitter", "reddit"):
                return jsonify({
                    "success": False,
                    "error": t('api.interviewListInvalidPlatform', index=i+1)
                }), 400

        # Verify environment alive status
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": t('api.envNotRunning')
            }), 400

        # Optimize prompts to prevent tool-calling loops
        optimized_interviews = []
        for interview in interviews:
            optimized_interview = interview.copy()
            optimized_interview['prompt'] = optimize_interview_prompt(interview.get('prompt', ''))
            optimized_interviews.append(optimized_interview)

        result = SimulationRunner.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=optimized_interviews,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": t('api.batchInterviewTimeout', error=str(e))
        }), 504

    except Exception as e:
        logger.error(f"Batch interview failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/all', methods=['POST'])
def interview_all_agents():
    """
    Global interview - interview all agents with the same question

    Note: requires simulation environment to be active

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",            // Required simulation ID
            "prompt": "What is your overall perspective on this event?",  // Required global question
            "platform": "reddit",                   // Optional platform
                                                    // When omitted in dual-platform mode, interviews both platforms
            "timeout": 180                          // Optional timeout in seconds, default 180
        }

    Response:
        {
            "success": true,
            "data": {
                "interviews_count": 50,
                "result": {
                    "interviews_count": 100,
                    "results": {
                        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit_0": {"agent_id": 0, "response": "...", "platform": "reddit"},
                        ...
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        prompt = data.get('prompt')
        platform = data.get('platform')  # Optional: twitter/reddit/None
        timeout = data.get('timeout', 180)

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400

        if not prompt:
            return jsonify({
                "success": False,
                "error": t('api.requirePrompt')
            }), 400

        # Validate platform parameter
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": t('api.invalidInterviewPlatform')
            }), 400

        # Verify environment alive status
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": t('api.envNotRunning')
            }), 400

        # Optimize prompt with instruction to prevent tool calls
        optimized_prompt = optimize_interview_prompt(prompt)

        result = SimulationRunner.interview_all_agents(
            simulation_id=simulation_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": t('api.globalInterviewTimeout', error=str(e))
        }), 504

    except Exception as e:
        logger.error(f"Global interview failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/history', methods=['POST'])
def get_interview_history():
    """
    Get interview history records

    Read interview trace records from platform SQLite databases

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",  // Required simulation ID
            "platform": "reddit",          // Optional platform (reddit/twitter)
                                           // If omitted, returns history from both platforms
            "agent_id": 0,                 // Optional: filter by agent ID
            "limit": 100                   // Optional: limit count, default 100
        }

    Response:
        {
            "success": true,
            "data": {
                "count": 10,
                "history": [
                    {
                        "agent_id": 0,
                        "response": "I believe that...",
                        "prompt": "What is your stance on this development?",
                        "timestamp": "2025-12-08T10:00:00",
                        "platform": "reddit"
                    },
                    ...
                ]
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        platform = data.get('platform')  # Return both if omitted
        agent_id = data.get('agent_id')
        limit = data.get('limit', 100)
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400

        history = SimulationRunner.get_interview_history(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            limit=limit
        )

        return jsonify({
            "success": True,
            "data": {
                "count": len(history),
                "history": history
            }
        })

    except Exception as e:
        logger.error(f"Failed to get interview history: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/env-status', methods=['POST'])
def get_env_status():
    """
    Get simulation environment status

    Check if simulation environment is active and ready for interview commands

    Request (JSON):
        {
            "simulation_id": "sim_xxxx"  // Required simulation ID
        }

    Response:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "env_alive": true,
                "twitter_available": true,
                "reddit_available": true,
                "message": "Environment is active and ready for interview commands"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400

        env_alive = SimulationRunner.check_env_alive(simulation_id)
        
        # Retrieve detailed environment status
        env_status = SimulationRunner.get_env_status_detail(simulation_id)

        if env_alive:
            message = t('api.envRunning')
        else:
            message = t('api.envNotRunningShort')

        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "env_alive": env_alive,
                "twitter_available": env_status.get("twitter_available", False),
                "reddit_available": env_status.get("reddit_available", False),
                "message": message
            }
        })

    except Exception as e:
        logger.error(f"Failed to get environment status: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/close-env', methods=['POST'])
def close_simulation_env():
    """
    Gracefully close simulation environment
    
    Sends command for simulation environment to cleanly exit wait loop.
    
    Note: unlike /stop which terminates processes immediately,
    this endpoint instructs the simulation to gracefully tear down and exit.
    
    Request (JSON):
        {
            "simulation_id": "sim_xxxx",  // Required simulation ID
            "timeout": 30                  // Optional timeout in seconds, default 30
        }
    
    Response:
        {
            "success": true,
            "data": {
                "message": "Environment close command dispatched",
                "result": {...},
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        timeout = data.get('timeout', 30)
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400
        
        result = SimulationRunner.close_simulation_env(
            simulation_id=simulation_id,
            timeout=timeout
        )
        
        # Update simulation state
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.COMPLETED
            manager._save_simulation_state(state)
        
        return jsonify({
            "success": result.get("success", False),
            "data": result
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Failed to close environment: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
