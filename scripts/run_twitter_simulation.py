"""
OASIS Twitter Simulation Runner Script
Automated simulation driver reading configuration parameters.

Features:
- Remains active in command wait mode upon simulation cycle completion
- Supports real-time interview commands via IPC
- Supports individual and batch agent interviews
- Supports remote graceful environment shutdown

Usage:
    python run_twitter_simulation.py --config /path/to/simulation_config.json
    python run_twitter_simulation.py --config /path/to/simulation_config.json --no-wait  # Exit immediately upon completion
"""

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import sys
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

# Global state for signal handling
_shutdown_event = None
_cleanup_done = False

# Add project root to sys.path
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

# Load root .env file containing LLM credentials
from dotenv import load_dotenv
_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
else:
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)


import re


class UnicodeFormatter(logging.Formatter):
    """Custom formatter converting Unicode escape sequences to readable text"""
    
    UNICODE_ESCAPE_PATTERN = re.compile(r'\\u([0-9a-fA-F]{4})')
    
    def format(self, record):
        result = super().format(record)
        
        def replace_unicode(match):
            try:
                return chr(int(match.group(1), 16))
            except (ValueError, OverflowError):
                return match.group(0)
        
        return self.UNICODE_ESCAPE_PATTERN.sub(replace_unicode, result)


class MaxTokensWarningFilter(logging.Filter):
    """Filter out camel-ai max_tokens warnings (omitted deliberately to let model decide)"""
    
    def filter(self, record):
        # Filter out max_tokens log warnings
        if "max_tokens" in record.getMessage() and "Invalid or missing" in record.getMessage():
            return False
        return True


# Install filter early before camel-ai imports execute
logging.getLogger().addFilter(MaxTokensWarningFilter())


def setup_oasis_logging(log_dir: str):
    """Configure OASIS logging with deterministic file paths"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Clean up legacy log files
    for f in os.listdir(log_dir):
        old_log = os.path.join(log_dir, f)
        if os.path.isfile(old_log) and f.endswith('.log'):
            try:
                os.remove(old_log)
            except OSError:
                pass
    
    formatter = UnicodeFormatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
    
    loggers_config = {
        "social.agent": os.path.join(log_dir, "social.agent.log"),
        "social.twitter": os.path.join(log_dir, "social.twitter.log"),
        "social.rec": os.path.join(log_dir, "social.rec.log"),
        "oasis.env": os.path.join(log_dir, "oasis.env.log"),
        "table": os.path.join(log_dir, "table.log"),
    }
    
    for logger_name, log_file in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False


try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph
    )
except ImportError as e:
    print(f"Error: Missing dependency {e}")
    print("Please install dependencies: pip install oasis-ai camel-ai")
    sys.exit(1)


# IPC Constants
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"

class CommandType:
    """IPC Command Types"""
    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class IPCHandler:
    """IPC Command Processor"""
    
    def __init__(self, simulation_dir: str, env, agent_graph):
        self.simulation_dir = simulation_dir
        self.env = env
        self.agent_graph = agent_graph
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        self._running = True
        
        # Ensure directory exists
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def update_status(self, status: str):
        """Update environment status file"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def poll_command(self) -> Optional[Dict[str, Any]]:
        """Poll for pending IPC command files"""
        if not os.path.exists(self.commands_dir):
            return None
        
        # Retrieve command files sorted by timestamp
        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))
        
        command_files.sort(key=lambda x: x[1])
        
        for filepath, _ in command_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
        
        return None
    
    def send_response(self, command_id: str, status: str, result: Dict = None, error: str = None):
        """Send IPC response file"""
        response = {
            "command_id": command_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        # Delete processed command file
        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass
    
    async def handle_interview(self, command_id: str, agent_id: int, prompt: str) -> bool:
        """
        Process single agent interview command.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get target agent
            agent = self.agent_graph.get_agent(agent_id)
            
            # Create interview action
            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": prompt}
            )
            
            # Execute interview action
            actions = {agent: interview_action}
            await self.env.step(actions)
            
            # Retrieve interview record from database
            result = self._get_interview_result(agent_id)
            
            self.send_response(command_id, "completed", result=result)
            print(f"  Interview complete: agent_id={agent_id}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"  Interview failed: agent_id={agent_id}, error={error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False
    
    async def handle_batch_interview(self, command_id: str, interviews: List[Dict]) -> bool:
        """
        Process batch interview command
        
        Args:
            interviews: [{"agent_id": int, "prompt": str}, ...]
        """
        try:
            # Build actions mapping
            actions = {}
            agent_prompts = {}  # Track prompt for each agent
            
            for interview in interviews:
                agent_id = interview.get("agent_id")
                prompt = interview.get("prompt", "")
                
                try:
                    agent = self.agent_graph.get_agent(agent_id)
                    actions[agent] = ManualAction(
                        action_type=ActionType.INTERVIEW,
                        action_args={"prompt": prompt}
                    )
                    agent_prompts[agent_id] = prompt
                except Exception as e:
                    print(f"  Warning: Could not get agent {agent_id}: {e}")
            
            if not actions:
                self.send_response(command_id, "failed", error="No valid agents available")
                return False
            
            # Execute batch interview actions
            await self.env.step(actions)
            
            # Gather all interview results
            results = {}
            for agent_id in agent_prompts.keys():
                result = self._get_interview_result(agent_id)
                results[agent_id] = result
            
            self.send_response(command_id, "completed", result={
                "interviews_count": len(results),
                "results": results
            })
            print(f"  Batch interview complete: {len(results)} agents")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"  Batch interview failed: {error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False
    
    def _get_interview_result(self, agent_id: int) -> Dict[str, Any]:
        """Retrieve latest interview results from SQLite database"""
        db_path = os.path.join(self.simulation_dir, "twitter_simulation.db")
        
        result = {
            "agent_id": agent_id,
            "response": None,
            "timestamp": None
        }
        
        if not os.path.exists(db_path):
            return result
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query latest interview record
            cursor.execute("""
                SELECT user_id, info, created_at
                FROM trace
                WHERE action = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (ActionType.INTERVIEW.value, agent_id))
            
            row = cursor.fetchone()
            if row:
                user_id, info_json, created_at = row
                try:
                    info = json.loads(info_json) if info_json else {}
                    result["response"] = info.get("response", info)
                    result["timestamp"] = created_at
                except json.JSONDecodeError:
                    result["response"] = info_json
            
            conn.close()
            
        except Exception as e:
            print(f"  Failed to read interview result: {e}")
        
        return result
    
    async def process_commands(self) -> bool:
        """
        Process pending IPC commands.

        Returns:
            True to continue running, False if shutdown commanded
        """
        command = self.poll_command()
        if not command:
            return True
        
        command_id = command.get("command_id")
        command_type = command.get("command_type")
        args = command.get("args", {})
        
        print(f"\nReceived IPC command: {command_type}, id={command_id}")
        
        if command_type == CommandType.INTERVIEW:
            await self.handle_interview(
                command_id,
                args.get("agent_id", 0),
                args.get("prompt", "")
            )
            return True
            
        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(
                command_id,
                args.get("interviews", [])
            )
            return True
            
        elif command_type == CommandType.CLOSE_ENV:
            print("Received close environment command")
            self.send_response(command_id, "completed", result={"message": "Environment shutting down"})
            return False
        
        else:
            self.send_response(command_id, "failed", error=f"Unknown command type: {command_type}")
            return True


class TwitterSimulationRunner:
    """Twitter Simulation Runner"""
    
    # Allowed Twitter actions (excluding INTERVIEW which triggers via ManualAction)
    AVAILABLE_ACTIONS = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.QUOTE_POST,
    ]
    
    def __init__(self, config_path: str, wait_for_commands: bool = True):
        """
        Initialize simulation runner.

        Args:
            config_path: Configuration file path (simulation_config.json)
            wait_for_commands: Whether to await IPC commands upon cycle completion
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.simulation_dir = os.path.dirname(config_path)
        self.wait_for_commands = wait_for_commands
        self.env = None
        self.agent_graph = None
        self.ipc_handler = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load simulation configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_profile_path(self) -> str:
        """Get profile CSV path (OASIS Twitter uses CSV format)"""
        return os.path.join(self.simulation_dir, "twitter_profiles.csv")
    
    def _get_db_path(self) -> str:
        """Get SQLite database path"""
        return os.path.join(self.simulation_dir, "twitter_simulation.db")
    
    def _create_model(self):
        """
        Initialize LLM model.

        Resolves credentials from project root .env file:
        - LLM_API_KEY: API Key
        - LLM_BASE_URL: Base URL
        - LLM_MODEL_NAME: Model name
        """
        # Read configuration from .env first
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        
        # Fallback to config file if not in .env
        if not llm_model:
            llm_model = self.config.get("llm_model", "gpt-4o-mini")
        
        # Set environment variables required by camel-ai
        if llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Missing API Key configuration: set LLM_API_KEY in root .env file")
        
        if llm_base_url:
            os.environ["OPENAI_API_BASE_URL"] = llm_base_url
        
        print(f"LLM config: model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else 'default'}...")
        
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=llm_model,
        )
    
    def _get_active_agents_for_round(
        self, 
        env, 
        current_hour: int,
        round_num: int
    ) -> List:
        """
        Select agents to activate for current round based on time and profiles.

        Args:
            env: OASIS environment
            current_hour: Simulated hour of day (0-23)
            round_num: Current round number

        Returns:
            List of activated agent instances
        """
        time_config = self.config.get("time_config", {})
        agent_configs = self.config.get("agent_configs", [])
        
        # Base activation count
        base_min = time_config.get("agents_per_hour_min", 5)
        base_max = time_config.get("agents_per_hour_max", 20)
        
        # Adjust by time of day
        peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
        off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
        
        if current_hour in peak_hours:
            multiplier = time_config.get("peak_activity_multiplier", 1.5)
        elif current_hour in off_peak_hours:
            multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
        else:
            multiplier = 1.0
        
        target_count = int(random.uniform(base_min, base_max) * multiplier)
        
        # Compute activation probability per agent
        candidates = []
        for cfg in agent_configs:
            agent_id = cfg.get("agent_id", 0)
            active_hours = cfg.get("active_hours", list(range(8, 23)))
            activity_level = cfg.get("activity_level", 0.5)
            
            # Check if within agent active hours
            if current_hour not in active_hours:
                continue
            
            # Compute probability based on activity level
            if random.random() < activity_level:
                candidates.append(agent_id)
        
        # Probabilistic selection
        selected_ids = random.sample(
            candidates, 
            min(target_count, len(candidates))
        ) if candidates else []
        
        # Convert to agent instances
        active_agents = []
        for agent_id in selected_ids:
            try:
                agent = env.agent_graph.get_agent(agent_id)
                active_agents.append((agent_id, agent))
            except Exception:
                pass
        
        return active_agents
    
    async def run(self, max_rounds: int = None):
        """Run Twitter simulation.

        Args:
            max_rounds: Optional max rounds to truncate run
        """
        print("=" * 60)
        print("OASIS Twitter Simulation")
        print(f"Configuration: {self.config_path}")
        print(f"Simulation ID: {self.config.get('simulation_id', 'unknown')}")
        print(f"Command Wait Mode: {'enabled' if self.wait_for_commands else 'disabled'}")
        print("=" * 60)
        
        # Load time parameters
        time_config = self.config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        
        # Compute total rounds
        total_rounds = (total_hours * 60) // minutes_per_round
        
        # Truncate if max_rounds specified
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                print(f"\nRounds truncated: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
        
        print(f"\nSimulation Parameters:")
        print(f"  - Total Simulated Duration: {total_hours} hours")
        print(f"  - Per-round Duration: {minutes_per_round} minutes")
        print(f"  - Total Rounds: {total_rounds}")
        if max_rounds:
            print(f"  - Max Rounds Limit: {max_rounds}")
        print(f"  - Agent Count: {len(self.config.get('agent_configs', []))}")
        
        # Initialize LLM model
        print("\nInitializing LLM model...")
        model = self._create_model()
        
        # Load agent profiles
        print("Loading Agent Profiles...")
        profile_path = self._get_profile_path()
        if not os.path.exists(profile_path):
            print(f"Error: Profile file not found: {profile_path}")
            return
        
        self.agent_graph = await generate_twitter_agent_graph(
            profile_path=profile_path,
            model=model,
            available_actions=self.AVAILABLE_ACTIONS,
        )
        
        # Database path
        db_path = self._get_db_path()
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Deleted existing database: {db_path}")
        
        # Create environment
        print("Creating OASIS environment...")
        self.env = oasis.make(
            agent_graph=self.agent_graph,
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=db_path,
            semaphore=30,  # Concurrency limit to prevent rate limits
        )
        
        await self.env.reset()
        print("Environment initialized successfully\n")
        
        # Initialize IPC processor
        self.ipc_handler = IPCHandler(self.simulation_dir, self.env, self.agent_graph)
        self.ipc_handler.update_status("running")
        
        # Execute seed events
        event_config = self.config.get("event_config", {})
        initial_posts = event_config.get("initial_posts", [])
        
        if initial_posts:
            print(f"Executing seed events ({len(initial_posts)} initial posts)...")
            initial_actions = {}
            for post in initial_posts:
                agent_id = post.get("poster_agent_id", 0)
                content = post.get("content", "")
                try:
                    agent = self.env.agent_graph.get_agent(agent_id)
                    initial_actions[agent] = ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    )
                except Exception as e:
                    print(f"  Warning: Could not create initial post for agent {agent_id}: {e}")
            
            if initial_actions:
                await self.env.step(initial_actions)
                print(f"  Published {len(initial_actions)} seed posts")
        
        # Main simulation loop
        print("\nStarting simulation loop...")
        start_time = datetime.now()
        
        for round_num in range(total_rounds):
            # Calculate current simulated time
            simulated_minutes = round_num * minutes_per_round
            simulated_hour = (simulated_minutes // 60) % 24
            simulated_day = simulated_minutes // (60 * 24) + 1
            
            # Get activated agents for round
            active_agents = self._get_active_agents_for_round(
                self.env, simulated_hour, round_num
            )
            
            if not active_agents:
                continue
            
            # Construct agent actions
            actions = {
                agent: LLMAction()
                for _, agent in active_agents
            }
            
            # Execute actions
            await self.env.step(actions)
            
            # Display progress
            if (round_num + 1) % 10 == 0 or round_num == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = (round_num + 1) / total_rounds * 100
                print(f"  [Day {simulated_day}, {simulated_hour:02d}:00] "
                      f"Round {round_num + 1}/{total_rounds} ({progress:.1f}%) "
                      f"- {len(active_agents)} agents active "
                      f"- elapsed: {elapsed:.1f}s")
        
        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\nSimulation loop completed!")
        print(f"  - Total Elapsed: {total_elapsed:.1f}s")
        print(f"  - Database: {db_path}")
        
        # Check if entering command wait loop
        if self.wait_for_commands:
            print("\n" + "=" * 60)
            print("Entering command wait mode - environment active")
            print("Supported commands: interview, batch_interview, close_env")
            print("=" * 60)
            
            self.ipc_handler.update_status("alive")
            
            # Wait loop using global shutdown event
            try:
                while not _shutdown_event.is_set():
                    should_continue = await self.ipc_handler.process_commands()
                    if not should_continue:
                        break
                    try:
                        await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                        break  # Received shutdown signal
                    except asyncio.TimeoutError:
                        pass
            except KeyboardInterrupt:
                print("\nReceived interrupt signal")
            except asyncio.CancelledError:
                print("\nTask cancelled")
            except Exception as e:
                print(f"\nCommand processing error: {e}")
            
            print("\nTearing down environment...")
        
        # Teardown environment
        self.ipc_handler.update_status("stopped")
        await self.env.close()
        
        print("Environment shut down")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='OASIS Twitter Simulation')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Configuration file path (simulation_config.json)'
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=None,
        help='Optional max simulation rounds to truncate run'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        default=False,
        help='Immediately exit on completion without entering command wait mode'
    )
    
    args = parser.parse_args()
    
    # Initialize shutdown event at main start
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Configure OASIS logging with clean file state
    simulation_dir = os.path.dirname(args.config) or "."
    setup_oasis_logging(os.path.join(simulation_dir, "log"))
    
    runner = TwitterSimulationRunner(
        config_path=args.config,
        wait_for_commands=not args.no_wait
    )
    await runner.run(max_rounds=args.max_rounds)


def setup_signal_handlers():
    """
    Configure signal handlers to cleanly handle SIGTERM and SIGINT,
    allowing resources (databases, environments) to tear down gracefully.
    """
    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\nReceived  {sig_name} signal, exiting gracefully...")
        if not _cleanup_done:
            _cleanup_done = True
            if _shutdown_event:
                _shutdown_event.set()
        else:
            # Multiple signals force immediate termination
            print("Forcing exit...")
            sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    setup_signal_handlers()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecution interrupted")
    except SystemExit:
        pass
    finally:
        print("Simulation process terminated")
