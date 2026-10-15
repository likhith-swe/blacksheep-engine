"""
Simulation Configuration Intelligence Generator.
Uses LLM reasoning over simulation requirements, document context, and knowledge graph entities
to automatically synthesize granular simulation parameters.

Adopts a staged generation strategy to guarantee reliability:
1. Formulate temporal schedule configuration.
2. Formulate event and seed post configuration.
3. Batch synthesize individual agent activity profiles.
4. Formulate social platform algorithm weights.
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, t
from ..utils.openai_chat_compat import create_chat_completion, extract_chat_completion_text
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('blacksheep.simulation_config')

# Standard activity schedule configuration
GLOBAL_TIMEZONE_CONFIG = {
    # Off-peak hours (very low activity)
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # Morning hours (gradually awakening)
    "morning_hours": [6, 7, 8],
    # Working hours
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # Evening peak (most active)
    "peak_hours": [19, 20, 21, 22],
    # Late night hours
    "night_hours": [23],
    # Activity multipliers
    "activity_multipliers": {
        "dead": 0.05,
        "morning": 0.4,
        "work": 0.7,
        "peak": 1.5,
        "night": 0.5
    }
}
CHINA_TIMEZONE_CONFIG = GLOBAL_TIMEZONE_CONFIG


@dataclass
class AgentActivityConfig:
    """Individual Agent Activity and Behavioral Configuration."""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str
    
    # Activity level (0.0 - 1.0)
    activity_level: float = 0.5
    
    # Expected action frequency per hour
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0
    
    # Active hours (24h format, 0-23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))
    
    # Response latency to trending events (in simulation minutes)
    response_delay_min: int = 5
    response_delay_max: int = 60
    
    # Sentiment bias (-1.0 to 1.0, negative to positive)
    sentiment_bias: float = 0.0
    
    # Stance on target topic
    stance: str = "neutral"  # supportive, opposing, neutral, observer
    
    # Influence multiplier for network visibility
    influence_weight: float = 1.0


@dataclass  
class TimeSimulationConfig:
    """Temporal Simulation Schedule Configuration."""
    # Total duration in simulation hours
    total_simulation_hours: int = 72  # Default 72 hours (3 days)
    
    # Simulated minutes elapsed per round (default 60 mins)
    minutes_per_round: int = 60
    
    # Range of activated agents per hour
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20
    
    # Peak activity hours
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5
    
    # Off-peak hours
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05
    
    # Morning hours
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4
    
    # Business hours
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Event and Seed Discourse Configuration."""
    # Seed posts published at the inception of simulation
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scheduled environmental interventions
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Trending topic keywords
    hot_topics: List[str] = field(default_factory=list)
    
    # Narrative evolution direction
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """Platform-specific Recommendation and Dynamics Configuration."""
    platform: str  # twitter or reddit
    
    # Feed algorithm weights
    recency_weight: float = 0.4
    popularity_weight: float = 0.3
    relevance_weight: float = 0.3
    
    # Virality threshold
    viral_threshold: int = 10
    
    # Echo chamber polarization coefficient
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Complete Simulation Parameters Manifest."""
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str
    
    # Schedule configuration
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)
    
    # Agent configurations
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)
    
    # Event configuration
    event_config: EventConfig = field(default_factory=EventConfig)
    
    # Platform configurations
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None
    
    # LLM configuration
    llm_model: str = ""
    llm_base_url: str = ""
    
    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert manifest to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    Simulation Configuration Intelligence Generator.
    
    Analyzes scenario requirements, document context, and knowledge graph entities
    to synthesize optimal simulation parameters.
    """
    
    MAX_CONTEXT_LENGTH = 50000
    AGENTS_PER_BATCH = 15
    
    TIME_CONFIG_CONTEXT_LENGTH = 10000
    EVENT_CONFIG_CONTEXT_LENGTH = 8000
    ENTITY_SUMMARY_LENGTH = 300
    AGENT_SUMMARY_LENGTH = 300
    ENTITIES_PER_TYPE_DISPLAY = 20
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        Generate complete simulation configuration across multiple staged steps.
        
        Args:
            simulation_id: Unique simulation ID
            project_id: Project identifier
            graph_id: Zep graph identifier
            simulation_requirement: Scenario description
            document_text: Raw context document
            entities: Filtered entity list
            enable_twitter: Enable Twitter platform
            enable_reddit: Enable Reddit platform
            progress_callback: Callback callable(step, total_steps, message)
            
        Returns:
            SimulationParameters manifest
        """
        logger.info(f"Starting simulation configuration generation: simulation_id={simulation_id}, entities={len(entities)}")
        
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches
        current_step = 0
        
        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")
        
        # 1. Build foundational context
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )
        
        reasoning_parts = []
        
        # ========== Step 1: Generate Time Configuration ==========
        report_progress(1, t('progress.generatingTimeConfig'))
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"{t('progress.timeConfigLabel')}: {time_config_result.get('reasoning', t('common.success'))}")
        
        # ========== Step 2: Generate Event Configuration ==========
        report_progress(2, t('progress.generatingEventConfig'))
        event_config_result = self._generate_event_config(context, simulation_requirement, entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"{t('progress.eventConfigLabel')}: {event_config_result.get('reasoning', t('common.success'))}")
        
        # ========== Step 3-N: Batch Generate Agent Configurations ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]
            
            report_progress(
                3 + batch_idx,
                t('progress.generatingAgentConfig', start=start_idx + 1, end=end_idx, total=len(entities))
            )
            
            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)
        
        reasoning_parts.append(t('progress.agentConfigResult', count=len(all_agent_configs)))
        
        # ========== Assign seed posts to matched poster agents ==========
        logger.info("Assigning seed posts to matched poster agents...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(t('progress.postAssignResult', count=assigned_count))
        
        # ========== Final Step: Generate Platform Configurations ==========
        report_progress(total_steps, t('progress.generatingPlatformConfig'))
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )
        
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        
        logger.info(f"Simulation configuration generated: {len(params.agent_configs)} agent profiles configured")
        
        return params
    
    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """Construct grounding context truncated to maximum length."""
        entity_summary = self._summarize_entities(entities)
        
        context_parts = [
            f"## Simulation Requirement\n{simulation_requirement}",
            f"\n## Entity Information ({len(entities)} entities)\n{entity_summary}",
        ]
        
        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500
        
        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(document truncated)"
            context_parts.append(f"\n## Original Document Context\n{doc_text}")
        
        return "\n".join(context_parts)
    
    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Generate structured entity summary."""
        lines = []
        
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)} entities)")
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... and {len(type_entities) - display_count} more")
        
        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Execute LLM invocation with exponential retry and JSON sanitization."""
        import re
        
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = create_chat_completion(
                    self.client,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1),
                )
                
                content = extract_chat_completion_text(response)
                finish_reason = response.choices[0].finish_reason
                
                if finish_reason == 'length':
                    logger.warning(f"LLM output truncated (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode failed (attempt {attempt+1}): {str(e)[:80]}")
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    last_error = e
                    
            except Exception as e:
                logger.warning(f"LLM invocation failed (attempt {attempt+1}): {str(e)[:80]}")
                import time
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "quota" in err_str or "rate_limit" in err_str or "resource_exhausted" in err_str
                is_transient = "503" in err_str or "500" in err_str or "unavailable" in err_str or "high demand" in err_str
                wait_time = 15 if is_rate_limit else (5 * (attempt + 1) if is_transient else 2 * (attempt + 1))
                time.sleep(wait_time)
        
        raise last_error or Exception("LLM call failed")
    
    def _fix_truncated_json(self, content: str) -> str:
        """Fix truncated JSON strings by balancing brackets and quotes."""
        content = content.strip()
        
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        if content and content[-1] not in '",}]':
            content += '"'
        
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Attempt regex and sanitation repair on corrupted JSON output."""
        import re
        
        content = self._fix_truncated_json(content)
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s
            
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)
            
            try:
                return json.loads(json_str)
            except Exception:
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except Exception:
                    pass
        
        return None
    
    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Generate simulation time schedule configuration."""
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]
        max_agents_allowed = max(1, int(num_entities * 0.9))
        
        prompt = f"""Based on the following simulation requirements, generate the simulation time configuration.

{context_truncated}

## Task
Generate a time configuration in JSON format.

### General Principles (adapt to the specific event and target population):
- Infer target user group timezones and daily routines based on the simulation scenario
- 0:00-5:00: Off-peak, minimal activity (multiplier 0.05)
- 6:00-8:00: Morning ramp-up (multiplier 0.4)
- 9:00-18:00: Business/working hours, steady activity (multiplier 0.7)
- 19:00-22:00: Evening peak hours, highest activity (multiplier 1.5)
- 23:00+: Late night wind-down (multiplier 0.5)
- Adapt active hours to participant characteristics:
  - Students or young demographics may peak between 21:00-24:00
  - Media accounts are active throughout the day
  - Official and corporate accounts primarily operate during standard business hours

### Return JSON Format (no markdown):
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Rationale for time schedule design in English"
}}

Field Definitions:
- total_simulation_hours (int): Total simulation hours, 24-168 hours
- minutes_per_round (int): Round duration in minutes (30-120 min, recommended 60)
- agents_per_hour_min (int): Min agents activated per hour (range: 1-{max_agents_allowed})
- agents_per_hour_max (int): Max agents activated per hour (range: 1-{max_agents_allowed})
- peak_hours (int array): Peak activity hours
- off_peak_hours (int array): Off-peak activity hours
- morning_hours (int array): Morning hours
- work_hours (int array): Working hours
- reasoning (string): Brief explanation for the schedule configuration in English"""

        system_prompt = "You are a social media simulation expert. Return pure JSON format. The time configuration must conform to the daily habits and active hours of the target user population in the simulation scenario."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Time config LLM generation failed: {e}, falling back to defaults")
            return self._get_default_time_config(num_entities)
    
    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Get default time schedule configuration."""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Using standard default activity schedule (60 mins per round)"
        }
    
    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """Parse time schedule configuration and validate agent activation boundaries."""
        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))
        
        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) exceeds total agents ({num_entities}), corrected")
            agents_per_hour_min = max(1, num_entities // 10)
        
        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) exceeds total agents ({num_entities}), corrected")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)
        
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max, corrected to {agents_per_hour_min}")
        
        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )
    
    def _generate_event_config(
        self, 
        context: str, 
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """Generate event configuration and initial seed posts."""
        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))
        
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)
        
        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}" 
            for t, examples in type_examples.items()
        ])
        
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]
        
        prompt = f"""Based on the following simulation requirement, generate the event configuration.

Simulation Requirement: {simulation_requirement}

{context_truncated}

## Available Entity Types and Examples
{type_info}

## Task
Generate the event configuration in JSON format:
- Extract trending topic keywords
- Describe public narrative direction
- Design initial seed posts, **each post must specify poster_type (matching available entity types)**

**Important**: poster_type must be selected from the available entity types above, so seed posts are assigned to appropriate agents.
Example: Official statements from Official/University, news from MediaOutlet, user perspectives from individual personas.

Return JSON format (no markdown):
{{
    "hot_topics": ["keyword1", "keyword2"],
    "narrative_direction": "<description of public discourse trajectory in English>",
    "initial_posts": [
        {{"content": "Post content in English", "poster_type": "Entity type exactly matching available types"}},
        ...
    ],
    "reasoning": "<brief explanation in English>"
}}"""

        system_prompt = "You are a public sentiment and discourse analysis expert. Return pure JSON format. Note that poster_type MUST exactly match available entity types."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'poster_type' field value MUST be in English PascalCase exactly matching the available entity types. Only 'content', 'narrative_direction', 'hot_topics' and 'reasoning' fields should use the specified language."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Event config LLM generation failed: {e}, falling back to defaults")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Using default configuration"
            }
    
    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Parse event configuration."""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )
    
    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        Assign appropriate poster agent IDs to initial seed posts.
        """
        if not event_config.initial_posts:
            return event_config
        
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)
        
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government", "enterprise"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media", "press"],
            "student": ["student", "person", "user"],
            "professor": ["professor", "expert", "teacher", "researcher"],
            "alumni": ["alumni", "person", "founder"],
            "organization": ["organization", "ngo", "company", "group", "enterprise"],
            "person": ["person", "student", "alumni", "founder", "developer"],
        }
        
        used_indices: Dict[str, int] = {}
        updated_posts = []
        
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")
            matched_agent_id = None
            
            # 1. Direct match
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. Alias match
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break
            
            # 3. Fallback: highest influence agent
            if matched_agent_id is None:
                logger.warning(f"No matching agent found for type '{poster_type}', assigning highest influence agent")
                if agent_configs:
                    sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0
            
            updated_posts.append({
                "content": content,
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_agent_id
            })
            
            logger.info(f"Initial post assignment: poster_type='{poster_type}' -> agent_id={matched_agent_id}")
        
        event_config.initial_posts = updated_posts
        return event_config
    
    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """Batch synthesize agent activity profiles."""
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })
        
        prompt = f"""Based on the following information, generate social media activity configurations for each entity.

Simulation Requirement: {simulation_requirement}

## Entity List
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Task
Generate activity configuration for each entity:
- Time habits matching target user demographic
- Official organizations (University/GovernmentAgency/Enterprise): Low activity (0.1-0.3), business hours (9-17), slower response (60-240 min), high influence (2.5-3.0)
- Media outlets (MediaOutlet): Medium activity (0.4-0.6), broad day hours (8-23), fast response (5-30 min), high influence (2.0-2.5)
- Individuals (Student/Person/User/Founder): High activity (0.6-0.9), primarily evening (18-23), rapid response (1-15 min), lower baseline influence (0.8-1.2)
- Public figures/Experts: Medium activity (0.4-0.6), medium-high influence (1.5-2.0)

Return JSON format (no markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <must match input agent_id>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <posting frequency>,
            "comments_per_hour": <commenting frequency>,
            "active_hours": [<list of active hours 0-23>],
            "response_delay_min": <min response delay in minutes>,
            "response_delay_max": <max response delay in minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <influence multiplier>
        }},
        ...
    ]
}}"""

        system_prompt = "You are a social media behavior analysis expert. Return pure JSON format. Configurations must align with the target audience's routine and habits in the simulation scenario."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'stance' field value MUST be one of the English strings: 'supportive', 'opposing', 'neutral', 'observer'. All JSON field names and numeric values must remain unchanged. Only natural language text fields should use the specified language."

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"Agent config batch LLM generation failed: {e}, falling back to rule-based generation")
            llm_configs = {}
        
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})
            
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)
            
            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)
        
        return configs
    
    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """Generate individual agent activity configuration using rule heuristics."""
        entity_type = (entity.get_entity_type() or "Unknown").lower()
        
        if entity_type in ["university", "governmentagency", "ngo", "enterprise"]:
            # Official organization: business hours, low frequency, high influence
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet", "media"]:
            # Media outlet: all day, medium frequency, high influence
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            # Professor/expert: business + evening, medium frequency
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            # Student: evening focus, high frequency
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni", "founder", "executive"]:
            # Alumni / executive: lunch + evening
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            # General individual: evening peak
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
