"""
Zep Retrieval Tools Service.
Encapsulates graph semantic search, node retrieval, and interview tools for the Report Agent.

Core retrieval tools:
1. InsightForge - Multi-dimensional hybrid retrieval with automated sub-query generation.
2. PanoramaSearch - Comprehensive breadth-first search across active and historical graph facts.
3. QuickSearch - Fast semantic edge/node lookup.
4. InterviewAgents - Targeted live Q&A with running simulated agent personas.
"""

import time
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from zep_cloud import NotFoundError

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.locale import get_locale, t
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges
from ..utils.zep import (
    call_zep_read_with_retry,
    get_zep_client,
    normalize_zep_search_limit,
    normalize_zep_search_query,
)

logger = get_logger('blacksheep.zep_tools')


@dataclass
class SearchResult:
    """Semantic Search Result."""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """Format as readable text for LLM ingestion."""
        text_parts = [f"Search Query: {self.query}", f"Found {self.total_count} relevant facts"]
        
        if self.facts:
            text_parts.append("\n### Relevant Facts:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Entity Node Record."""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "Unknown Type")
        return f"Entity: {self.name} (Type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    """Graph Relationship Edge Record."""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Temporal metadata
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Relationship: {source} --[{self.name}]--> {target}\nFact: {self.fact}"
        
        if include_temporal:
            valid_at = self.valid_at or "Unknown"
            invalid_at = self.invalid_at or "Present"
            base_text += f"\nValidity: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Expired: {self.expired_at})"
        
        return base_text
    
    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None
    
    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    InsightForge Multi-dimensional Deep Inquiry Result.
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)
    
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        text_parts = [
            f"## In-Depth Simulation Forecast Analysis",
            f"Analytical Query: {self.query}",
            f"Simulation Scenario: {self.simulation_requirement}",
            f"\n### Simulation Data Metrics",
            f"- Relevant Predicted Facts: {self.total_facts}",
            f"- Entities Involved: {self.total_entities}",
            f"- Relationship Chains: {self.total_relationships}"
        ]
        
        if self.sub_queries:
            text_parts.append(f"\n### Analyzed Sub-Queries")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        
        if self.semantic_facts:
            text_parts.append(f'\n### [Key Facts] (Quote these verbatim in the report)')
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        
        if self.entity_insights:
            text_parts.append(f"\n### [Core Entities]")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(f"  Summary: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  Related Facts: {len(entity.get('related_facts', []))}")
        
        if self.relationship_chains:
            text_parts.append(f"\n### [Relationship Chains]")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Panorama Comprehensive Graph State Result.
    """
    query: str
    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)
    
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        text_parts = [
            f"## Panoramic Search Results (Future Overview)",
            f"Query: {self.query}",
            f"\n### Statistics",
            f"- Total Nodes: {self.total_nodes}",
            f"- Total Edges: {self.total_edges}",
            f"- Active Facts: {self.active_count}",
            f"- Historical / Superseded Facts: {self.historical_count}"
        ]
        
        if self.active_facts:
            text_parts.append(f"\n### [Active Facts] (Latest simulation outcomes)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        
        if self.historical_facts:
            text_parts.append(f"\n### [Historical / Superseded Facts] (Evolutionary records)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        
        if self.all_nodes:
            text_parts.append(f"\n### [Involved Entities]")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Individual Agent Interview Response."""
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Key Quotes:**\n"
            for quote in self.key_quotes:
                clean_quote = quote.replace('"', '').strip()
                while clean_quote and clean_quote[0] in ',;:.\n\r\t ':
                    clean_quote = clean_quote[1:]
                
                skip = False
                for d in '123456789':
                    if f'Question {d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('.', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """Collective Agent Interview Result."""
    interview_topic: str
    interview_questions: List[str]
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)
    selection_reasoning: str = ""
    summary: str = ""
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        text_parts = [
            "## In-Depth Interview Transcript",
            f"**Interview Topic:** {self.interview_topic}",
            f"**Agents Interviewed:** {self.interviewed_count} / {self.total_agents} Simulated Agents",
            "\n### Agent Selection Rationale",
            self.selection_reasoning or "(Automated selection)",
            "\n---",
            "\n### Interview Transcripts",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(No interview records)\n\n---")

        text_parts.append("\n### Interview Summary and Key Takeaways")
        text_parts.append(self.summary or "(No summary available)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Zep Knowledge Graph Retrieval and Agent Interview Service.
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY is not configured")
        
        self.client = get_zep_client(self.api_key)
        self._llm_client = llm_client
        logger.info(t("console.zepToolsInitialized"))
    
    @property
    def llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        return call_zep_read_with_retry(
            func,
            operation_name=operation_name,
            max_attempts=max_retries or self.MAX_RETRIES,
            initial_delay=self.RETRY_DELAY,
        )
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """Execute semantic search over Zep graph."""
        logger.info(t("console.graphSearch", graphId=graph_id, query=query[:50]))
        
        zep_query = normalize_zep_search_query(query)
        zep_limit = normalize_zep_search_limit(limit)

        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=zep_query,
                    limit=zep_limit,
                    scope=scope,
                    reranker="cross_encoder"
                ),
                operation_name=t("console.graphSearchOp", graphId=graph_id)
            )
            
            facts = []
            edges = []
            nodes = []
            
            if hasattr(search_results, 'edges') and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })
            
            if hasattr(search_results, 'nodes') and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append({
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                        "name": getattr(node, 'name', ''),
                        "labels": getattr(node, 'labels', []),
                        "summary": getattr(node, 'summary', ''),
                    })
                    if hasattr(node, 'summary') and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(t("console.searchComplete", count=len(facts)))
            
            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )
            
        except Exception as e:
            logger.error(t("console.zepSearchApiFallback", error=str(e)))
            raise
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """Local fallback keyword search."""
        logger.info(t("console.usingLocalSearch", query=query[:30]))
        
        facts = []
        edges_result = []
        nodes_result = []
        
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            if not text:
                return 0
            text_lower = text.lower()
            if query_lower in text_lower:
                return 100
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score
        
        try:
            if scope in ["edges", "both"]:
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))
                
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(t("console.localSearchComplete", count=len(facts)))
            
        except Exception as e:
            logger.error(t("console.localSearchFailed", error=str(e)))
        
        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """Fetch all graph nodes with pagination."""
        logger.info(t("console.fetchingAllNodes", graphId=graph_id))

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
            result.append(NodeInfo(
                uuid=str(node_uuid) if node_uuid else "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            ))

        logger.info(t("console.fetchedNodes", count=len(result)))
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """Fetch all graph edges with pagination."""
        logger.info(t("console.fetchingAllEdges", graphId=graph_id))

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or ""
            )

            if include_temporal:
                edge_info.created_at = getattr(edge, 'created_at', None)
                edge_info.valid_at = getattr(edge, 'valid_at', None)
                edge_info.invalid_at = getattr(edge, 'invalid_at', None)
                edge_info.expired_at = getattr(edge, 'expired_at', None)

            result.append(edge_info)

        logger.info(t("console.fetchedEdges", count=len(result)))
        return result
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """Fetch detail for a single node UUID."""
        logger.info(t("console.fetchingNodeDetail", uuid=node_uuid[:8]))
        
        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=t("console.fetchNodeDetailOp", uuid=node_uuid[:8])
            )
            
            if not node:
                return None
            
            return NodeInfo(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            )
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(t("console.fetchNodeDetailFailed", error=str(e)))
            raise
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """Fetch all incident edges for a node."""
        logger.info(t("console.fetchingNodeEdges", uuid=node_uuid[:8]))
        
        try:
            all_edges = self.get_all_edges(graph_id)
            result = [
                edge for edge in all_edges 
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid
            ]
            logger.info(t("console.foundNodeEdges", count=len(result)))
            return result
        except Exception as e:
            logger.error(t("console.fetchNodeEdgesFailed", error=str(e)))
            raise
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """Fetch all entities bearing a specific label."""
        logger.info(t("console.fetchingEntitiesByType", type=entity_type))
        all_nodes = self.get_all_nodes(graph_id)
        filtered = [node for node in all_nodes if entity_type in node.labels]
        logger.info(t("console.foundEntitiesByType", count=len(filtered), type=entity_type))
        return filtered
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """Fetch relationship context and facts for an entity name."""
        logger.info(t("console.fetchingEntitySummary", name=entity_name))
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )
        
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break
        
        related_edges = []
        if entity_node:
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Retrieve node, edge, and label count distributions."""
        logger.info(t("console.fetchingGraphStats", graphId=graph_id))
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1
        
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_simulation_context(
        self, 
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Retrieve overall simulation context and entity baseline."""
        logger.info(t("console.fetchingSimContext", requirement=simulation_requirement[:50]))
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )
        
        stats = self.get_graph_statistics(graph_id)
        all_nodes = self.get_all_nodes(graph_id)
        
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })
        
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities)
        }
    
    # ========== Core Strategic Retrieval Tools ==========
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        InsightForge: Decomposes inquiries into multidimensional sub-queries and aggregates insights.
        """
        logger.info(t("console.insightForgeStart", query=query[:50]))
        
        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )
        
        # Step 1: Synthesize sub-queries with LLM
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(t("console.generatedSubQueries", count=len(sub_queries)))
        
        # Step 2: Semantic search per sub-query
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            
            all_edges.extend(search_result.edges)
        
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        # Step 3: Extract entity UUIDs and retrieve detailed insights
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        entity_insights = []
        node_map = {}
        
        for uuid in list(entity_uuids):
            if not uuid:
                continue
            try:
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                    
                    related_facts = [
                        f for f in all_facts 
                        if node.name.lower() in f.lower()
                    ]
                    
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts
                    })
            except Exception as e:
                logger.debug(f"Failed to fetch node {uuid}: {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        # Step 4: Construct relationship chains
        relationship_chains = []
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        logger.info(t("console.insightForgeComplete", facts=result.total_facts, entities=result.total_entities, relationships=result.total_relationships))
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """Decompose inquiry into targeted sub-queries via LLM."""
        system_prompt = """You are an expert question analyst. Your task is to break down a complex inquiry into multiple targeted sub-queries that can be observed and retrieved independently within a simulated world.

Requirements:
1. Each sub-query should be specific and measurable through agent actions, stances, or statements
2. Cover multiple dimensions (who, what, why, how, sentiment, risks)
3. Maintain relevance to the simulation scenario
4. Write all sub-queries in English
5. Return JSON format: {"sub_queries": ["sub-query 1", "sub-query 2", ...]}"""

        user_prompt = f"""Simulation Scenario Background:
{simulation_requirement}

{f"Report Context: {report_context[:500]}" if report_context else ""}

Please decompose the following inquiry into {max_queries} targeted sub-queries in English:
{query}

Return the list in strict JSON format."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            sub_queries = response.get("sub_queries", [])
            return [str(sq) for sq in sub_queries[:max_queries]]
            
        except Exception as e:
            logger.warning(t("console.generateSubQueriesFailed", error=str(e)))
            return [
                query,
                f"Key stakeholders in {query}",
                f"Causes and repercussions of {query}",
                f"Progression and sentiment evolution of {query}"
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """Panorama search across all active and historical facts."""
        logger.info(t("console.panoramaSearchStart", query=query[:50]))
        
        result = PanoramaResult(query=query)
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)
        
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)
        
        active_facts = []
        historical_facts = []
        
        for edge in all_edges:
            if not edge.fact:
                continue
            
            is_historical = edge.is_expired or edge.is_invalid
            if is_historical:
                valid_at = edge.valid_at or "Unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "Unknown"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                active_facts.append(edge.fact)
        
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        logger.info(t("console.panoramaSearchComplete", active=result.active_count, historical=result.historical_count))
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """Fast semantic lookup over edges."""
        logger.info(t("console.quickSearchStart", query=query[:50]))
        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
        logger.info(t("console.quickSearchComplete", count=result.total_count))
        return result
    
    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        Interview live simulated agents across social platforms.
        """
        from .simulation_runner import SimulationRunner
        import os
        from .simulation_ipc import SimulationIPCClient
        
        logger.info(t("console.interviewAgentsStart", requirement=interview_requirement[:50]))
        
        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )
        
        sim_dir = os.path.join(SimulationRunner.RUN_STATE_DIR, simulation_id)
        ipc_client = SimulationIPCClient(sim_dir)
        if not ipc_client.check_env_alive():
            logger.info(f"Simulation environment is not active or closed ({simulation_id}), skipping live interview and using graph data")
            result.summary = "The simulation environment has concluded and is closed. Real-time interview is unavailable. Generating report from existing graph and discourse facts."
            return result
        
        profiles = self._load_agent_profiles(simulation_id)
        if not profiles:
            logger.warning(t("console.profilesNotFound", simId=simulation_id))
            result.summary = "No interviewable agent profiles found"
            return result
        
        result.total_agents = len(profiles)
        logger.info(t("console.loadedProfiles", count=len(profiles)))
        
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )
        
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(t("console.selectedAgentsForInterview", count=len(selected_agents), indices=selected_indices))
        
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(t("console.generatedInterviewQuestions", count=len(result.interview_questions)))
        
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        
        INTERVIEW_PROMPT_PREFIX = (
            "You are being interviewed. Drawing upon your persona, memories, and actions, "
            "directly answer the following questions in natural English plain text.\n"
            "Response guidelines:\n"
            "1. Answer in natural conversational English; do NOT invoke any tools\n"
            "2. Do NOT output JSON format or tool call blocks\n"
            "3. Do NOT use Markdown headings (#, ##, ###)\n"
            "4. Answer each question in order, prefixed with 'Question X:' (where X is the question number)\n"
            "5. Separate answers to each question with a blank line\n"
            "6. Provide thoughtful, substantive answers in English (at least 2-3 sentences per question)\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"
        
        try:
            interviews_request = [
                {"agent_id": agent_idx, "prompt": optimized_prompt}
                for agent_idx in selected_indices
            ]
            
            logger.info(t("console.callingBatchInterviewApi", count=len(interviews_request)))
            
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0
            )
            
            logger.info(t("console.interviewApiReturned", count=api_result.get('interviews_count', 0), success=api_result.get('success')))
            
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Unknown error")
                logger.warning(t("console.interviewApiReturnedFailure", error=error_msg))
                result.summary = f"Interview API call failed: {error_msg}. Please check simulation environment status."
                return result
            
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}
            
            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Unknown")
                agent_bio = agent.get("bio", "")
                
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})
                
                twitter_response = self._clean_tool_call_response(twitter_result.get("response", ""))
                reddit_response = self._clean_tool_call_response(reddit_result.get("response", ""))

                twitter_text = twitter_response if twitter_response else "(No response on this platform)"
                reddit_text = reddit_response if reddit_response else "(No response on this platform)"
                response_text = f"[Twitter Platform Response]\n{twitter_text}\n\n[Reddit Platform Response]\n{reddit_text}"

                combined_responses = f"{twitter_response} {reddit_response}"
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'Question\s*\d+[：:]\s*', '', clean_text)

                sentences = re.split(r'[.!?\n]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W,;:]+', s.strip())
                    and not s.strip().startswith(('{', 'Question'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "." for s in meaningful[:3]]

                if not key_quotes:
                    paired = re.findall(r'"([^"]{15,100})"', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[,;:]', q)][:3]
                
                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)
            
            result.interviewed_count = len(result.interviews)
            
        except ValueError as e:
            logger.warning(t("console.interviewApiCallFailed", error=e))
            result.summary = f"Interview failed: {str(e)}. The simulation environment may have closed; please ensure it is running."
            return result
        except Exception as e:
            logger.error(t("console.interviewApiCallException", error=e))
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"Interview error occurred: {str(e)}"
            return result
        
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )
        
        logger.info(t("console.interviewAgentsComplete", count=result.interviewed_count))
        return result
    
    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Extract inner message text if LLM outputted raw tool JSON payload."""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Load stored simulation agent personas."""
        import os
        import csv
        
        sim_dir = os.path.join(
            os.path.dirname(__file__), 
            f'../../uploads/simulations/{simulation_id}'
        )
        
        profiles = []
        
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(t("console.loadedRedditProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readRedditProfilesFailed", error=e))
        
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Unknown"
                        })
                logger.info(t("console.loadedTwitterProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readTwitterProfilesFailed", error=e))
        
        return profiles
    
    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """Select representative agents to interview based on theme and role."""
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "Unknown"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)
        
        system_prompt = """You are an expert interview producer. Your task is to select the most suitable agents to interview from a list of simulated agent personas based on the interview objective.

Selection Criteria:
1. Agent background/profession is directly relevant to the interview theme
2. Agent holds distinct, insightful, or polarized perspectives
3. Select diverse viewpoints (e.g., proponents, opponents, neutral observers, practitioners, experts)
4. Prioritize agents directly affected by the events

Return JSON format:
{
    "selected_indices": [0, 1, 2],
    "reasoning": "Explanation of selection rationale in English"
}"""

        user_prompt = f"""Interview Theme / Requirement:
{interview_requirement}

Simulation Context:
{simulation_requirement if simulation_requirement else "Not provided"}

Available Agents (Total {len(agent_summaries)}):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Please select up to {max_agents} most suitable agents to interview and explain your rationale in English."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Automatically selected based on relevance")
            
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            
            return selected_agents, valid_indices, reasoning
            
        except Exception as e:
            logger.warning(t("console.llmSelectAgentFailed", error=e))
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Default selection fallback"
    
    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """Synthesize interview questions via LLM."""
        agent_roles = [a.get("profession", "Unknown") for a in selected_agents]
        
        system_prompt = """You are an expert investigative journalist. Generate 3-5 in-depth interview questions based on the interview requirements.

Question Guidelines:
1. Open-ended questions that provoke substantive, authentic answers
2. Tailored to illuminate differences in perspectives across roles
3. Cover facts, emotional reactions, motivations, and perceived systemic risks
4. Natural, conversational tone
5. Clear and concise (under 25 words per question)
6. Direct questions only, without meta-explanations or prefixes
7. All questions must be written in English

Return JSON format: {"questions": ["Question 1", "Question 2", ...]}"""

        user_prompt = f"""Interview Objective: {interview_requirement}

Simulation Scenario: {simulation_requirement if simulation_requirement else "Not provided"}

Interviewee Roles: {', '.join(agent_roles)}

Please generate 3-5 sharp, in-depth interview questions in English."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            return response.get("questions", [f"What is your perspective on {interview_requirement}?"])
            
        except Exception as e:
            logger.warning(t("console.generateInterviewQuestionsFailed", error=e))
            return [
                f"What is your perspective on {interview_requirement}?",
                "How does this development impact you and your immediate stakeholders?",
                "What actions or resolutions do you advocate for moving forward?"
            ]
    
    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """Synthesize interview responses into an analytical summary."""
        if not interviews:
            return "No interviews were completed."
        
        interview_texts = [
            f"[{interview.agent_name} ({interview.agent_role})]\n{interview.response[:500]}"
            for interview in interviews
        ]
        
        system_prompt = """You are an expert news editor. Synthesize the responses from multiple interviewees into a comprehensive interview summary in English.

Summary Requirements:
1. Distill primary perspectives and core themes across participants
2. Highlight areas of consensus versus sharp divergence
3. Emphasize notable, high-impact quotations
4. Maintain an objective, balanced tone without taking sides
5. Write 100% in professional English
6. Keep under 500 words

Formatting Constraints:
- Use clear paragraphs separated by blank lines
- Do NOT use markdown headers (#, ##, ###)
- Do NOT use horizontal dividers (---, ***)
- Use standard English quotation marks "" for verbatim quotes
- You may use **bold text** to highlight key points, but avoid other markdown syntax"""

        user_prompt = f"""Interview Topic: {interview_requirement}

Interview Transcripts:
{"".join(interview_texts)}

Please generate the synthesized interview summary in English."""

        try:
            return self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
        except Exception as e:
            logger.warning(t("console.generateInterviewSummaryFailed", error=e))
            return f"Interviewed {len(interviews)} respondents: " + ", ".join([i.agent_name for i in interviews])
