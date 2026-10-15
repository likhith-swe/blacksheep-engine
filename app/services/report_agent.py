"""
Report Agent Service
Implements ReACT-pattern simulation report generation with Zep graph retrieval.

Capabilities:
1. Generate predictive reports based on scenario conditions and Zep graph memories
2. Plan outline structure, followed by progressive section-by-section generation
3. ReACT thought, tool-call, observation, and reflection loops per section
4. Interactive Q&A chat with autonomous retrieval tool invocation
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, t
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('blacksheep.report_agent')


class ReportLogger:
    """
    Report Agent Detailed Action Logger

    Generates agent_log.jsonl in the report directory, tracking every action step.
    Each line is a JSON object with timestamp, action type, stage, and full details.
    """
    
    def __init__(self, report_id: str):
        """
        Initialize detailed action logger.

        Args:
            report_id: Report ID used to construct log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure log directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time in seconds from start"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Record a log entry.

        Args:
            action: Action type ('report_start', 'tool_call', 'llm_response', etc.)
            stage: Current pipeline stage ('planning', 'generating', 'completed')
            details: Untruncated payload dictionary
            section_title: Current section title (optional)
            section_index: Current section index (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Append to JSONL log file
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Log report generation start"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": t('report.taskStarted')
            }
        )
    
    def log_planning_start(self):
        """Log outline planning start"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": t('report.planningStart')}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Log context gathered during planning"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": t('report.fetchSimContext'),
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Log outline planning completion"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": t('report.planningComplete'),
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Log section generation start"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": t('report.sectionStart', title=section_title)}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Log ReACT thought process"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": t('report.reactThought', iteration=iteration)
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Log tool invocation"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": t('report.toolCall', toolName=tool_name)
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Log tool invocation result (untruncated)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Untruncated tool output
                "result_length": len(result),
                "message": t('report.toolResult', toolName=tool_name)
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Log LLM response (untruncated)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Untruncated LLM response
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": t('report.llmResponse', hasToolCalls=has_tool_calls, hasFinalAnswer=has_final_answer)
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Log section content generation completion"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Untruncated section content
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": t('report.sectionContentDone', title=section_title)
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Log section completion

        Clients listen to this log event to detect section completion and retrieve full content
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": t('report.sectionComplete', title=section_title)
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Log full report generation completion"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": t('report.reportComplete')
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Log pipeline error"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": t('report.errorOccurred', error=error_message)
            }
        )


class ReportConsoleLogger:
    """
    Report Agent Console Logger

    Writes plain-text console logs (INFO, WARNING, etc.) to console_log.txt in the report directory.
    """
    
    def __init__(self, report_id: str):
        """
        Initialize console logger
        
        Args:
            report_id: Report ID used to construct log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Ensure log directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Set up file handler to mirror logs to console_log.txt"""
        import logging
        
        # Create file handler
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # Use plain console output formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # Attach handler to blacksheep logger
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Avoid duplicate handlers
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Close file handler and detach from loggers"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """Ensure handler is closed on garbage collection"""
        self.close()


class ReportStatus(str, Enum):
    """Report generation status enum"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Report section data model"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Convert to Markdown representation"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Report outline data model"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Convert to Markdown representation"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Full report data model"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt Template Constants
# ═══════════════════════════════════════════════════════════════

# ── Tool Descriptions ──

TOOL_DESC_INSIGHT_FORGE = """\
[Deep Insight Search - Powerful Retrieval Tool]
A comprehensive retrieval function designed for deep analysis:
1. Automatically breaks down your question into targeted sub-questions
2. Retrieves simulation graph insights across multiple dimensions
3. Integrates semantic search, entity analysis, and relationship chain tracking
4. Returns the most complete and deep analytical material

[Use Cases]
- In-depth analysis of a specific topic or theme
- Understanding multiple facets of an event or scenario
- Gathering rich supporting material for report sections

[Returns]
- Relevant factual quotes (can be directly cited)
- Core entity insights
- Relationship chain dynamics"""

TOOL_DESC_PANORAMA_SEARCH = """\
[Panorama Search - High-level Ecosystem View]
Provides a bird's-eye view of simulation dynamics, especially suited for understanding event evolution:
1. Fetches all relevant nodes and relationships
2. Distinguishes active/current facts from historical/superseded facts
3. Traces opinion and sentiment evolution over time

[Use Cases]
- Mapping out the end-to-end trajectory of events
- Comparing opinion shifts across different simulation phases
- Extracting comprehensive entity and relationship networks

[Returns]
- Current active facts (latest simulation outcomes)
- Historical/superseded facts (chronological evolution records)
- All involved entities"""

TOOL_DESC_QUICK_SEARCH = """\
[Quick Search - Fast Information Retrieval]
Lightweight, rapid lookup tool suitable for targeted, specific inquiries:

[Use Cases]
- Fast lookup for a specific topic, metric, or entity
- Fact verification
- Direct information retrieval

[Returns]
- Ranked list of facts most relevant to the query"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[Deep Interview - Live Agent Interviews (Dual-Platform)]
Calls the OASIS simulation interview API to conduct real-time interviews with simulated agents!
This directly interfaces with simulated persona agents in the running environment to capture authentic opinions and quotes.
Defaults to interviewing across both Twitter and Reddit platforms for diverse perspectives.

[Workflow]
1. Automatically inspects persona profiles to identify relevant agents
2. Intelligently selects agents most relevant to the interview topic (e.g., engineers, founders, critics, users)
3. Generates targeted interview questions
4. Calls /api/simulation/interview/batch across dual platforms
5. Synthesizes interview transcripts, providing multi-perspective analysis

[Use Cases]
- Capturing viewpoints from distinct stakeholder roles
- Collecting divergent stances and objections
- Gathering verbatim quotes from simulated agents
- Enhancing report credibility with primary source interview excerpts

[Returns]
- Interviewed agent profile details
- Responses from Twitter and Reddit
- Key quotations (ready for blockquote citation)
- Interview summary and viewpoint contrast

[Important] Requires an active simulation environment to execute!"""

# ── Outline Planning Prompt ──

PLAN_SYSTEM_PROMPT = """\
You are an expert author of Future Predictive Reports, possessing an omniscient bird's-eye view over the simulated world—enabling you to inspect every agent's actions, statements, and community interactions.

[Core Concept]
We constructed a simulated world and injected specific simulation requirements as variables. The simulated evolution serves as a predictive forecast of future developments. You are not inspecting sterile lab data, but rather observing an unfolding preview of the future.

[Your Mission]
Draft a comprehensive Future Predictive Report answering:
1. Under the established conditions, what unfolded in the future?
2. How did various agent personas and stakeholder groups react and behave?
3. What critical future trends, systemic risks, and opportunities does this simulation reveal?

[Report Standards]
- High-fidelity future prediction: reveals "what happens if..."
- Focus on emergent outcomes: event trajectories, group dynamics, polarization, systemic risks
- Agent quotes and actions represent predictive behavioral evidence
- NOT a generic review of present-day news
- NOT a superficial summary

[Language Strict Requirement]
- You MUST write the report entirely in English!
- The title, summary, and all section titles and descriptions must be in clear, professional English.

[Section Count Constraints]
- Minimum 2 sections, maximum 5 sections
- No nested sub-sections; each section is a complete, deep chapter
- Concise, high-density, evidence-backed narrative
- Structure designed thoughtfully around simulation findings

Output the outline in strict JSON format:
{
    "title": "Report Title in English",
    "summary": "Report Summary in English (one concise sentence capturing core findings)",
    "sections": [
        {
            "title": "Section Title in English",
            "description": "Section Description in English"
        }
    ]
}

Note: The sections array must contain between 2 and 5 elements!"""

PLAN_USER_PROMPT_TEMPLATE = """\
[Predictive Scenario Parameters]
Simulation Requirement / Variable: {simulation_requirement}

[Simulated World Scale]
- Total Entities in Simulation: {total_nodes}
- Interaction Relationships Formed: {total_edges}
- Entity Type Distribution: {entity_types}
- Active Agents: {total_entities}

[Sample Factual Evidence Extracted from Simulation]
{related_facts_json}

Examine this future simulation preview with an omniscient perspective:
1. Under our defined scenario, what future landscape emerged?
2. How did different stakeholder groups (agents) react and mobilize?
3. What significant trends, tensions, and risks are evident?

Design an optimal report outline based on these predictive findings.

[Reminder] Section count: 2 to 5 sections. Keep content focused, evidence-driven, and written entirely in English."""

# ── Section Generation Prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert analyst writing a section of a Future Predictive Report.

Report Title: {report_title}
Report Summary: {report_summary}
Simulation Requirement / Scenario: {simulation_requirement}

Current Section to Write: {section_title}

═══════════════════════════════════════════════════════════════
[Core Concept]
═══════════════════════════════════════════════════════════════

The simulation is a preview of the future. We introduced specific conditions (simulation requirements), and the resulting behaviors and interactions among agents represent predictive forecasts of human stakeholder behavior.

Your objective:
- Reveal what unfolds in the future under these scenario conditions
- Forecast how various personas and groups respond and act
- Identify emerging patterns, structural risks, and strategic takeaways

- Do NOT frame this as retrospective commentary on historical facts
- DO ground everything in the simulated future outcomes

═══════════════════════════════════════════════════════════════
[Critical Rules - Mandatory]
═══════════════════════════════════════════════════════════════

1. [Always Use Tools to Inspect Simulation Data]
   - You are observing the future simulation via tools
   - All assertions must be backed by simulated events and agent statements
   - Do NOT fabricate content from external priors
   - Call tools at least 3 times (maximum 5 times) per section to gather factual evidence

2. [Quote Agent Statements Verbatim]
   - Agent remarks and actions are direct behavioral predictions
   - Include standalone blockquotes to highlight evidence, e.g.:
     > "Stakeholder Group X cautions: 'Verbatim agent quote...'"
   - These quotes serve as primary empirical proof

3. [Language Consistency - English Only]
   - The entire report, including all prose, analysis, headings, and quotes, MUST be written in English.
   - If any retrieved tool content contains Chinese or other languages, you MUST translate it accurately into natural, professional English before incorporating it.

4. [Faithful Presentation of Evidence]
   - Accurately represent the simulation findings
   - Do not invent events that did not occur in the simulation
   - If evidence is limited on a certain point, state so transparently

5. [Zero Hallucination of Data]
   - Do NOT invent fake usernames, fabricated statistics, or fictitious interactions
   - Never output fake <tool_result> blocks — tool results are provided exclusively by the system
   - Only cite entities, quotes, and metrics returned by tools

═══════════════════════════════════════════════════════════════
[Formatting Rules - Essential!]
═══════════════════════════════════════════════════════════════

[One Section = Atomic Content Block]
- Each section is an atomic building block of the report
- Do NOT include any Markdown headers (#, ##, ###, ####, etc.) inside the section text
- Do NOT begin the text with the section title
- The section header is rendered automatically by the UI system; write only body text
- Use **bold headers**, paragraphs, blockquotes (>), and bullet points (-) to structure your writing

[Correct Example]
This section examines community adoption dynamics. Analysis of simulation data demonstrates significant bifurcation...

**Initial Adoption Surge**

Early-stage founders exhibited immediate enthusiasm, prioritizing operational speed:

> "Startup founders express strong demand for automated cost reduction, citing urgent runway pressures."

**Enterprise Governance Hesitation**

In contrast, enterprise engineering leaders adopted a cautious stance:

- Heightened scrutiny around automated write permissions
- Compliance and SLA verification prerequisites

[Incorrect Example]
## Executive Summary       <-- WRONG! Do not add any # headers
### 1. Phase One           <-- WRONG! Do not use ### headers

This section examines...

═══════════════════════════════════════════════════════════════
[Available Retrieval Tools] (Call 3-5 times per section)
═══════════════════════════════════════════════════════════════

{tools_description}

[Tool Usage Guidelines - Blend different tools, do not rely on just one]
- insight_forge: Deep decomposition and multidimensional retrieval of facts and relationships
- panorama_search: Comprehensive ecosystem view, chronological timeline, and shift patterns
- quick_search: Rapid verification of specific facts or entities
- interview_agents: Live interviews with simulated agents across Twitter and Reddit

═══════════════════════════════════════════════════════════════
[Workflow]
═══════════════════════════════════════════════════════════════

In each turn, you must perform exactly ONE of the following actions:

Option A - Call a Tool:
State your reasoning (Thought), then invoke a tool using this exact format:
<tool_call>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>
The system will execute the tool and return the result. Do not write tool results yourself.

Option B - Output Final Answer:
Once you have retrieved sufficient evidence (at least 3 tool calls), output the section text prefixed with "Final Answer:".

Strictly Forbidden:
- Never include both a <tool_call> and Final Answer in the same response
- Never fabricate Observation or <tool_result> blocks
- Maximum of one tool call per turn

═══════════════════════════════════════════════════════════════
[Content Quality Criteria]
═══════════════════════════════════════════════════════════════

1. Content must be grounded in simulated evidence
2. Generously quote agent statements to illustrate real dynamics
3. Use Markdown without headers:
   - Use **bold text** for sub-headings
   - Use bullet points (- or 1.2.3.) for key takeaways
   - Separate paragraphs with blank lines
   - NO #, ##, ###, or #### headers
4. [Standalone Blockquotes Format]
   Blockquotes must be standalone paragraphs with blank lines before and after:

   Correct:
   Engineering leadership voiced significant reservations.

   > "Enterprise CTOs emphasize that granting automated cleanup permissions carries unacceptable risk."

   This skepticism shaped initial enterprise rollout delays.

5. Maintain coherent narrative progression with prior sections
6. [Avoid Repetition] Review previously generated sections carefully to avoid redundant coverage
7. [English Language Requirement] The entire section must be written in professional, fluent English
8. Do NOT add markdown headers; use **bold** instead"""

SECTION_USER_PROMPT_TEMPLATE = """\
Previously completed sections (review carefully to avoid redundancy):
{previous_content}

═══════════════════════════════════════════════════════════════
[Current Assignment] Write Section: {section_title}
═══════════════════════════════════════════════════════════════

[Key Reminders]
1. Read completed sections above to avoid duplicating points!
2. You must call tools first to retrieve simulation data before writing.
3. Use a mix of tools (insight_forge, panorama_search, quick_search, interview_agents).
4. Content must come from tool retrieval results, not external knowledge.
5. Write 100% in fluent English.

[Formatting Rule]
- NO markdown headers (#, ##, ###, ####)
- Do NOT begin with "{section_title}"
- Begin directly with narrative text, using **bold text** for section divisions

Please begin:
1. Thought: What specific evidence is required for this section?
2. Action: Invoke a tool to retrieve simulation data
3. Once sufficient data is gathered, provide the Final Answer (pure body text in English, no headers)"""

# ── ReACT Loop Message Templates ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (Retrieval Results):

=== Tool {tool_name} Returned ===
{result}

═══════════════════════════════════════════════════════════════
Tools invoked: {tool_calls_count}/{max_tool_calls} (Used: {used_tools_str}){unused_hint}
- If information is sufficient: begin with "Final Answer:" and provide the section content in English (quoting verbatim evidence)
- If more information is needed: invoke another tool to continue retrieval
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[Notice] You have only called tools {tool_calls_count} time(s); a minimum of {min_tool_calls} tool calls is required. "
    "Please call another tool to retrieve more simulation data before providing the Final Answer.{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Currently only {tool_calls_count} tool call(s) have been made. A minimum of {min_tool_calls} is required. "
    "Please call a tool to retrieve simulation data.{unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Tool call limit reached ({tool_calls_count}/{max_tool_calls}). No further tool calls are permitted. "
    'Please immediately synthesize the retrieved evidence and output the section content prefixed with "Final Answer:" in English.'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 Unused tools: {unused_list}. Try diverse tools to gather comprehensive perspectives."

REACT_FORCE_FINAL_MSG = "Tool call limit reached. Please immediately output Final Answer: with the section content in English."

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are an intelligent, concise simulation analysis assistant.

[Scenario Context]
Simulation Scenario / Objective: {simulation_requirement}

[Generated Analysis Report]
{report_content}

[Rules]
1. Primarily answer based on the generated report content above
2. Provide direct, objective answers without verbose chain-of-thought
3. Only call tools if the report content lacks the necessary data to answer
4. Keep answers concise, clear, well-structured, and strictly in English
5. Respond in English only

[Available Tools] (Use only when necessary, max 1-2 calls)
{tools_description}

[Tool Call Format]
<tool_call>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>

[Style Guidelines]
- Direct and clear
- Use > blockquotes for important evidence
- Provide conclusions first, followed by supporting rationale"""

CHAT_OBSERVATION_SUFFIX = "\n\nPlease answer the question concisely in English."


# ═══════════════════════════════════════════════════════════════
# ReportAgent Class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - Simulation Report Generator

    Employs ReACT (Reasoning + Acting) pattern:
    1. Planning: Analyze simulation requirements and plan report outline
    2. Generation: Progressive section generation with multi-turn tool retrieval
    3. Reflection: Verify content integrity and evidence completeness
    """
    
    # Maximum tool calls per section
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # Maximum reflection turns
    MAX_REFLECTION_ROUNDS = 3
    
    # Maximum tool calls in conversational chat
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        Initialize Report Agent.

        Args:
            graph_id: Zep graph ID
            simulation_id: Simulation identifier
            simulation_requirement: Simulation scenario description
            llm_client: Optional LLMClient instance
            zep_tools: Optional ZepToolsService instance
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # Available tools definition
        self.tools = self._define_tools()
        
        # Detailed action logger (initialized in generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Console file logger (initialized in generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(t('report.agentInitDone', graphId=graph_id, simulationId=simulation_id))
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define available retrieval tools"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "The question or topic you want to analyze in depth",
                    "report_context": "Current report section context (optional, helps generate more targeted sub-questions)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Search query for relevance ranking",
                    "include_expired": "Whether to include expired/historical content (default True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search query string",
                    "limit": "Number of results to return (optional, default 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interview topic or requirement description (e.g., 'Understand perspectives on cloud cost automation')",
                    "max_agents": "Maximum number of agents to interview (optional, default 5, max 10)"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Execute tool call.

        Args:
            tool_name: Tool identifier
            parameters: Tool argument mapping
            report_context: Context string for InsightForge decomposition

        Returns:
            Tool execution result as string
        """
        logger.info(t('report.executingTool', toolName=tool_name, params=parameters))
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Panorama search - ecosystem view
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Quick search - targeted fact retrieval
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Deep interview - query simulated agents across Twitter and Reddit
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # Legacy tool redirects for backwards compatibility
            
            elif tool_name == "search_graph":
                # Redirect to quick_search
                logger.info(t('report.redirectToQuickSearch'))
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Redirect to insight_forge
                logger.info(t('report.redirectToInsightForge'))
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Unknown tool: {tool_name}. Please use one of the following tools: insight_forge, panorama_search, quick_search, interview_agents"
                
        except Exception as e:
            logger.error(t('report.toolExecFailed', toolName=tool_name, error=str(e)))
            return f"Tool execution failed: {str(e)}"
    
    # Allowed tool names for fallback bare JSON parsing
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response

        Supported formats (by priority):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. Bare JSON (response or single line is a bare tool call JSON)
        """
        tool_calls = []

        # Format 1: XML tag style (primary)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: Fallback - Bare JSON without tool_call tags
        # Only attempted if Format 1 fails
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Attempt to extract the last valid JSON object from mixed text
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validate whether extracted dictionary represents a valid tool call"""
        # Accept either name/parameters or tool/params keys
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalize keys to name and parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Generate text description of available tools"""
        desc_parts = ["Available Tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameters: {params_desc}")
        return "\n".join(desc_parts)

    @staticmethod
    def _strip_fake_tool_results(response: str) -> str:
        """Strip any <tool_result> blocks the LLM fabricated in its response.

        When the LLM generates a <tool_call> block and then continues to generate
        a <tool_result> block in the same response, we must strip the fake result
        before appending to message history. The real tool result will be injected
        separately by the system.
        """
        tag_pattern = re.compile(r'</?tool_result\b[^>]*>', flags=re.IGNORECASE)
        parts = []
        cursor = 0
        depth = 0

        for match in tag_pattern.finditer(response):
            if depth == 0:
                parts.append(response[cursor:match.start()])

            if match.group(0).lstrip().startswith('</'):
                depth = max(0, depth - 1)
            else:
                depth += 1
            cursor = match.end()

        if depth == 0:
            parts.append(response[cursor:])

        cleaned = ''.join(parts)
        # Treat a malformed opening tag without a closing `>` as unsafe too.
        cleaned = re.sub(r'<tool_result\b.*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Plan report outline.

        Analyzes simulation requirements and graph context to construct report outline.

        Args:
            progress_callback: Progress callback callable

        Returns:
            ReportOutline: Planned report outline
        """
        logger.info(t('report.startPlanningOutline'))
        
        if progress_callback:
            progress_callback("planning", 0, t('progress.analyzingRequirements'))
        
        # Gather simulation knowledge graph context
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, t('progress.generatingOutline'))
        
        system_prompt = f"{PLAN_SYSTEM_PROMPT}\n\n{get_language_instruction()}"
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, t('progress.parsingOutline'))
            
            # Parse planned outline JSON
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Simulation Analysis Report"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, t('progress.outlinePlanComplete'))
            
            logger.info(t('report.outlinePlanDone', count=len(sections)))
            return outline
            
        except Exception as e:
            logger.error(t('report.outlinePlanFailed', error=str(e)))
            # Fallback outline in English
            return ReportOutline(
                title="Future Predictive Report",
                summary="Predictive analysis of future trends, behaviors, and risks based on simulation",
                sections=[
                    ReportSection(title="Scenario Landscape and Core Findings"),
                    ReportSection(title="Group Dynamics and Behavioral Analysis"),
                    ReportSection(title="Future Outlook and Strategic Implications")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Generate a single section using ReACT mode.

        ReACT loop:
        1. Thought - Analyze information needed
        2. Action - Invoke retrieval tools
        3. Observation - Analyze tool results
        4. Iterate until sufficient evidence is gathered
        5. Final Answer - Draft section content

        Args:
            section: Section to generate
            outline: Full report outline
            previous_sections: Preceding sections for continuity
            progress_callback: Progress callback
            section_index: 1-indexed section position

        Returns:
            Section content in Markdown format
        """
        logger.info(t('report.reactGenerateSection', title=section.title))
        
        # Log section start
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # Build user prompt with preceding completed sections
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Limit preceding section context to 4000 characters
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(This is the first section)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACT iteration loop
        tool_calls_count = 0
        max_iterations = 5  # Max ReACT iterations
        min_tool_calls = 3  # Minimum tool calls required
        conflict_retries = 0  # Tool call vs Final Answer conflict retry counter
        used_tools = set()  # Track invoked tool names
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Context passed to InsightForge for targeted sub-queries
        report_context = f"Section Title: {section.title}\nSimulation Objective: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    t('progress.deepSearchAndWrite', current=tool_calls_count, max=self.MAX_TOOL_CALLS_PER_SECTION)
                )
            
            # Invoke LLM
            try:
                response = self.llm.chat(
                    messages=messages,
                    temperature=0.5,
                    max_tokens=4096
                )
            except Exception as e:
                logger.warning(f"Section '{section.title}' LLM chat failed (iteration {iteration+1}): {e}")
                response = None

            # Handle empty or None LLM responses
            if response is None:
                logger.warning(t('report.sectionIterNone', title=section.title, iteration=iteration + 1))
                # Retry if iterations remain
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(Empty response)"})
                    messages.append({"role": "user", "content": "Please continue generating content in English."})
                    continue
                # Fall through to final synthesis on exhausted retries
                break

            logger.debug(f"LLM response: {response[:200]}...")

            # Parse tool calls once and reuse
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Conflict Resolution: LLM emitted both tool call and Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    t('report.sectionConflict', title=section.title, iteration=iteration+1, conflictCount=conflict_retries)
                )

                if conflict_retries <= 2:
                    # Retries 1-2: Prompt LLM to choose either a tool call or Final Answer
                    cleaned_response = ReportAgent._strip_fake_tool_results(response)
                    messages.append({"role": "assistant", "content": cleaned_response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Format Error] You included both a tool call and Final Answer in a single response, which is not permitted.\n"
                            "In each turn, you can only do ONE of the following:\n"
                            "- Call a single tool (output one <tool_call> block, do not write Final Answer)\n"
                            "- Output final content (begin with 'Final Answer:', do not include <tool_call>)\n"
                            "Please respond again doing only one of these actions."
                        ),
                    })
                    continue
                else:
                    # Retry 3: Force tool execution by taking the first tool call
                    logger.warning(
                        t('report.sectionConflictDowngrade', title=section.title, conflictCount=conflict_retries)
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Log LLM response
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Case 1: LLM emitted Final Answer ──
            if has_final_answer:
                cleaned_response = ReportAgent._strip_fake_tool_results(response)
                # Require minimum tool calls before accepting Final Answer
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": cleaned_response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f" (Unused tools: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Successful section completion
                final_answer = cleaned_response.split("Final Answer:")[-1].strip()
                logger.info(t('report.sectionGenDone', title=section.title, count=tool_calls_count))

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Case 2: LLM invoked a tool ──
            if has_tool_calls:
                # Tool quota reached -> Prompt for Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    cleaned_response = ReportAgent._strip_fake_tool_results(response)
                    messages.append({"role": "assistant", "content": cleaned_response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Execute first tool call
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(t('report.multiToolOnlyFirst', total=len(tool_calls), toolName=call['name']))

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Construct prompt suggesting unutilized tools
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list=", ".join(unused_tools))

                cleaned_response = ReportAgent._strip_fake_tool_results(response)
                messages.append({"role": "assistant", "content": cleaned_response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── Case 3: Neither tool call nor Final Answer found ──
            cleaned_response = ReportAgent._strip_fake_tool_results(response)
            messages.append({"role": "assistant", "content": cleaned_response})

            if tool_calls_count < min_tool_calls:
                # Suggest tools when minimum count is not met
                unused_tools = all_tools - used_tools
                unused_hint = f" (Unused tools: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Sufficient tool calls completed; accept generated content as answer
            # Treat output as final content to prevent spinning
            logger.info(t('report.sectionNoPrefix', title=section.title, count=tool_calls_count))
            final_answer = cleaned_response

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Max iterations reached: synthesize final section content
        logger.warning(t('report.sectionMaxIter', title=section.title))
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        try:
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )
        except Exception as e:
            logger.error(f"Force final answer LLM chat failed for section '{section.title}': {e}")
            response = None

        # Validate final synthesis response
        if response is None:
            logger.error(t('report.sectionForceFailed', title=section.title))
            final_answer = t('report.sectionGenFailedContent')
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Log section generation completion
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Generate full report with progressive section-by-section output
        
        Each section is persisted immediately upon completion without waiting for the full report.
        File structure:
        reports/{report_id}/
            meta.json       - Report metadata
            outline.json    - Report outline
            progress.json   - Generation progress
            section_01.md   - Section 1 Markdown
            section_02.md   - Section 2 Markdown
            ...
            full_report.md  - Assembled full report
        
        Args:
            progress_callback: Progress callback (stage, progress, message)
            report_id: Optional custom report ID
            
        Returns:
            Report: Assembled report object
        """
        import uuid
        
        # Auto-generate report_id if omitted
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # Track completed section titles for progress reporting
        completed_section_titles = []
        
        try:
            # Initialize report directory and persisted state
            ReportManager._ensure_report_folder(report_id)
            
            # Initialize structured action logger (agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Initialize console text logger (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, t('progress.initReport'),
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Phase 1: Plan Outline
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, t('progress.startPlanningOutline'),
                completed_sections=[]
            )
            
            # Log planning start
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, t('progress.startPlanningOutline'))
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Log planning completion
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Persist outline
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, t('progress.outlineDone', count=len(outline.sections)),
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(t('report.outlineSavedToFile', reportId=report_id))
            
            # Phase 2: Progressive Section Generation
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Accumulate section content for downstream context
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    t('progress.generatingSection', title=section.title, current=section_num, total=total_sections),
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        t('progress.generatingSection', title=section.title, current=section_num, total=total_sections)
                    )
                
                # Generate section content via ReACT loop
                try:
                    section_content = self._generate_section_react(
                        section=section,
                        outline=outline,
                        previous_sections=generated_sections,
                        progress_callback=lambda stage, prog, msg:
                            progress_callback(
                                stage, 
                                base_progress + int(prog * 0.7 / total_sections),
                                msg
                            ) if progress_callback else None,
                        section_index=section_num
                    )
                except Exception as sec_err:
                    logger.error(f"Error generating section {section_num} ({section.title}): {sec_err}")
                    section_content = f"Simulation analysis for [{section.title}]: Based on multi-agent opinion dynamics and knowledge graph data, relevant groups exhibit clear divergence on this topic. Please refer to the comprehensive synthesis and subsequent sections for detailed insights."
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Persist completed section
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Log section completion
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(t('report.sectionSaved', reportId=report_id, sectionNum=f"{section_num:02d}"))
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    t('progress.sectionDone', title=section.title),
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # Phase 3: Assemble Full Report
            if progress_callback:
                progress_callback("generating", 95, t('progress.assemblingReport'))
            
            ReportManager.update_progress(
                report_id, "generating", 95, t('progress.assemblingReport'),
                completed_sections=completed_section_titles
            )
            
            # Assemble full Markdown report via ReportManager
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Compute total elapsed duration
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Log report completion
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Persist final report
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, t('progress.reportComplete'),
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, t('progress.reportComplete'))
            
            logger.info(t('report.reportGenDone', reportId=report_id))
            
            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(t('report.reportGenFailed', error=str(e)))
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Log error event
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Persist failed status
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, t('progress.reportFailed', error=str(e)),
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Suppress secondary persistence errors
            
            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Interactive chat with Report Agent
        
        Agent autonomously invokes retrieval tools to answer questions
        
        Args:
            message: User query
            chat_history: Dialogue history list
            
        Returns:
            {
                "response": "Agent response",
                "tool_calls": [list of invoked tools],
                "sources": [cited sources]
            }
        """
        logger.info(t('report.agentChat', message=message[:50]))
        
        chat_history = chat_history or []
        
        # Retrieve generated report content
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Truncate report context to fit prompt window
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Report content truncated] ..."
        except Exception as e:
            logger.warning(t('report.fetchReportFailed', error=e))
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(No report content available)",
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # Construct prompt messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Append recent conversation turns
        for h in chat_history[-10:]:  # Limit history window
            messages.append(h)
        
        # Append user question
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # ReACT iteration loop (simplified)
        tool_calls_made = []
        max_iterations = 2  # Limited turns for conversational response
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # Return directly if no tools invoked
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                clean_response = ReportAgent._strip_fake_tool_results(clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Execute tool call
            tool_results = []
            for call in tool_calls[:1]:  # Single tool invocation per turn
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Truncate tool output in chat context
                })
                tool_calls_made.append(call)
            
            # Append tool result to dialogue history
            cleaned_response = ReportAgent._strip_fake_tool_results(response)
            messages.append({"role": "assistant", "content": cleaned_response})
            observation = "\n".join([f"[{r['tool']} Result]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # Solicit final answer upon iteration limit
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Clean up markdown formatting in response
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        clean_response = ReportAgent._strip_fake_tool_results(clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Report Manager
    
    Responsible for persistence, retrieval, and assembly of simulation reports
    
    Directory layout (section-based persistence):
    reports/
      {report_id}/
        meta.json          - Report metadata and status
        outline.json       - Report outline
        progress.json      - Generation progress
        section_01.md      - Section 1 Markdown
        section_02.md      - Section 2 Markdown
        ...
        full_report.md     - Full Markdown report
    """
    
    # Reports base directory
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Ensure reports base directory exists"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Get path to report directory"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Ensure report directory exists and return path"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Get metadata file path"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Get full report Markdown file path"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Get outline file path"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Get progress file path"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Get section Markdown file path"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Get agent log file path"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Get console log file path"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get console log content
        
        Plain-text console lines (INFO, WARNING, etc.) captured during report generation,
        distinct from structured JSONL records in agent_log.jsonl.
        
        Args:
            report_id: Report identifier
            from_line: 0-indexed starting line for incremental streaming
            
        Returns:
            {
                "logs": [log lines],
                "total_lines": total lines,
                "from_line": start line number,
                "has_more": whether more lines exist
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Retain original log line without trailing newline
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Reached EOF
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Get full console log in a single call.

        Args:
            report_id: Report identifier

        Returns:
            List of log lines
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get structured agent action logs
        
        Args:
            report_id: Report identifier
            from_line: 0-indexed starting line for incremental streaming
            
        Returns:
            {
                "logs": [log entries],
                "total_lines": total lines,
                "from_line": start line number,
                "has_more": whether more lines exist
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skip unparseable lines
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Reached EOF
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get full structured agent action log.

        Args:
            report_id: Report identifier

        Returns:
            List of log entries
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Persist report outline.

        Invoked immediately upon outline planning completion.
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(t('report.outlineSaved', reportId=report_id))
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Save single section

        Invoked immediately upon section generation to persist progress

        Args:
            report_id: Report identifier
            section_index: 1-indexed section number
            section: ReportSection object

        Returns:
            Saved file path
        """
        cls._ensure_report_folder(report_id)

        # Build section Markdown content, removing redundant headers
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Write file
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(t('report.sectionFileSaved', reportId=report_id, fileSuffix=file_suffix))
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Clean up section Markdown content.

        1. Removes leading headers that duplicate section title
        2. Converts subheadings to bold text

        Args:
            content: Raw Markdown content
            section_title: Target section title

        Returns:
            Cleaned Markdown string
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if line is a Markdown header
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Check for duplicate title header within first 5 lines
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # Convert header lines to bold text
                # System automatically renders section headings
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Blank line separator
                continue
            
            # Skip trailing blank line after skipped header
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Remove leading blank lines
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Remove leading horizontal dividers
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Remove blank lines after divider
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Update persisted report generation progress.
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report generation progress"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get list of completed section files.
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract section index from filename
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Assemble full report from saved section files.
        """
        folder = cls._get_report_folder(report_id)
        
        # Build report header block
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Read section files in sequence
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Post-process: clean header levels across full report
        md_content = cls._post_process_report(md_content, outline)
        
        # Write assembled full report
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(t('report.fullReportAssembled', reportId=report_id))
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Post-process assembled report content.

        1. Remove duplicate headers
        2. Preserve main title (#) and section headers (##); convert subheadings to bold
        3. Clean up excessive whitespace and dividers

        Args:
            content: Raw assembled Markdown
            outline: Report outline object

        Returns:
            Cleaned report Markdown string
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Collect all section titles from outline
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if line is a Markdown header
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Check for duplicate header within 5 preceding lines
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Skip duplicate header and following blank lines
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Header hierarchy:
                # - # (level=1) Keep only main report title
                # - ## (level=2) Keep section headers
                # - ###+ (level>=3) Convert to bold text
                
                if level == 1:
                    if title == outline.title:
                        # Keep main title
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Correct section header level to ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Convert other H1 headers to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Keep section header
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Convert non-section H2 headers to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # Convert H3+ headers to bold
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Skip divider immediately following header
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # Enforce single blank line after header
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Collapse multiple consecutive blank lines (max 2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Save report metadata and full markdown"""
        cls._ensure_report_folder(report.report_id)
        
        # Save meta.json
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save outline.json
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Save full_report.md
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(t('report.reportSaved', reportId=report.report_id))
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Get report by ID"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # Backwards compatibility: check legacy flat report files
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct Report object
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # Read full_report.md if markdown_content is empty
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Get report by simulation ID"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Directory format
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Legacy JSON format
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """List all reports"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Directory format
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Legacy JSON format
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # Sort by creation timestamp descending
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Delete report directory"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # Delete report directory
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(t('report.reportFolderDeleted', reportId=report_id))
            return True
        
        # Delete legacy report file
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
