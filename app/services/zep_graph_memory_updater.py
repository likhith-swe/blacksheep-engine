"""
Zep Graph Memory Updater Service.
Dynamically ingests running simulation actions and agent interactions into the Zep knowledge graph.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_locale, set_locale
from ..utils.zep import (
    ZEP_INGESTION_WAIT_TIMEOUT_SECONDS,
    call_zep_read_with_retry,
    get_zep_client,
)

logger = get_logger('blacksheep.zep_graph_memory_updater')


@dataclass
class AgentActivity:
    """Agent Activity Event Record."""
    platform: str           # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str        # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str
    
    def to_episode_text(self) -> str:
        """
        Convert activity into natural language episode text for Zep semantic extraction.
        Keeps contextual details intact to allow Zep to derive relations and facts.
        """
        action_descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }
        
        describe_func = action_descriptions.get(self.action_type, self._describe_generic)
        description = describe_func()
        
        return (
            f"[{self.timestamp}] [{self.platform} round {self.round_num}] "
            f"{self.agent_name}: {description}"
        )
    
    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f'posted: "{content}"'
        return "published a post"
    
    def _describe_like_post(self) -> str:
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if post_content and post_author:
            return f'liked {post_author}\'s post: "{post_content}"'
        elif post_content:
            return f'liked a post: "{post_content}"'
        elif post_author:
            return f'liked a post by {post_author}'
        return "liked a post"
    
    def _describe_dislike_post(self) -> str:
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if post_content and post_author:
            return f'disliked {post_author}\'s post: "{post_content}"'
        elif post_content:
            return f'disliked a post: "{post_content}"'
        elif post_author:
            return f'disliked a post by {post_author}'
        return "disliked a post"
    
    def _describe_repost(self) -> str:
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        
        if original_content and original_author:
            return f'reposted {original_author}\'s post: "{original_content}"'
        elif original_content:
            return f'reposted a post: "{original_content}"'
        elif original_author:
            return f'reposted a post by {original_author}'
        return "reposted a post"
    
    def _describe_quote_post(self) -> str:
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        quote_content = self.action_args.get("quote_content", "") or self.action_args.get("content", "")
        
        base = ""
        if original_content and original_author:
            base = f'quoted {original_author}\'s post: "{original_content}"'
        elif original_content:
            base = f'quoted a post: "{original_content}"'
        elif original_author:
            base = f'quoted a post by {original_author}'
        else:
            base = "quoted a post"
        
        if quote_content:
            base += f' with comment: "{quote_content}"'
        return base
    
    def _describe_follow(self) -> str:
        target_user_name = self.action_args.get("target_user_name", "")
        if target_user_name:
            return f'followed user "{target_user_name}"'
        return "followed a user"
    
    def _describe_create_comment(self) -> str:
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if content:
            if post_content and post_author:
                return f'commented on {post_author}\'s post "{post_content}": "{content}"'
            elif post_content:
                return f'commented on post "{post_content}": "{content}"'
            elif post_author:
                return f'commented on {post_author}\'s post: "{content}"'
            return f'commented: "{content}"'
        return "posted a comment"
    
    def _describe_like_comment(self) -> str:
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        
        if comment_content and comment_author:
            return f'liked {comment_author}\'s comment: "{comment_content}"'
        elif comment_content:
            return f'liked a comment: "{comment_content}"'
        elif comment_author:
            return f'liked a comment by {comment_author}'
        return "liked a comment"
    
    def _describe_dislike_comment(self) -> str:
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        
        if comment_content and comment_author:
            return f'disliked {comment_author}\'s comment: "{comment_content}"'
        elif comment_content:
            return f'disliked a comment: "{comment_content}"'
        elif comment_author:
            return f'disliked a comment by {comment_author}'
        return "disliked a comment"
    
    def _describe_search(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f'searched for "{query}"' if query else "performed a search"
    
    def _describe_search_user(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f'searched for user "{query}"' if query else "searched for users"
    
    def _describe_mute(self) -> str:
        target_user_name = self.action_args.get("target_user_name", "")
        if target_user_name:
            return f'muted user "{target_user_name}"'
        return "muted a user"
    
    def _describe_generic(self) -> str:
        return f"performed {self.action_type} action"


class _DrainDeadlineExceeded(TimeoutError):
    def __init__(self, processed_count: int):
        super().__init__("Zep updater drain deadline elapsed")
        self.processed_count = processed_count


class ZepGraphMemoryUpdater:
    """
    Zep Graph Memory Updater.
    
    Monitors simulation actions and streams formatted agent activity episodes into the Zep knowledge graph.
    Buffers events by platform and sends them in batches to optimize rate limits.
    """
    
    BATCH_SIZE = 5
    
    PLATFORM_DISPLAY_NAMES = {
        'twitter': 'Twitter',
        'reddit': 'Reddit',
    }
    
    SEND_INTERVAL = 0.5
    MAX_EPISODE_CHARS = 9_500
    
    def __init__(
        self,
        graph_id: str,
        api_key: Optional[str] = None,
        simulation_id: Optional[str] = None,
    ):
        """
        Initialize memory updater.
        
        Args:
            graph_id: Zep Graph ID
            api_key: Optional Zep API key override
            simulation_id: Simulation identifier
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id or "unknown"
        self.api_key = api_key or Config.ZEP_API_KEY
        
        if not self.api_key:
            raise ValueError("ZEP_API_KEY is not configured")
        
        self.client = get_zep_client(self.api_key)
        
        self._activity_queue: Queue = Queue()
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            'twitter': [],
            'reddit': [],
        }
        self._buffer_lock = threading.Lock()
        self._acceptance_lock = threading.Lock()
        
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        self._total_activities = 0
        self._total_sent = 0
        self._total_items_sent = 0
        self._failed_count = 0
        self._skipped_count = 0
        self._failed_batches: List[Dict[str, Any]] = []
        self._pending_episode_uuids: List[str] = []
        
        logger.info(f"ZepGraphMemoryUpdater initialized: graph_id={graph_id}, batch_size={self.BATCH_SIZE}")
    
    def _get_platform_display_name(self, platform: str) -> str:
        return self.PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform)
    
    def start(self):
        """Start background worker thread."""
        if self._running:
            return

        current_locale = get_locale()

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(current_locale,),
            daemon=True,
            name=f"ZepMemoryUpdater-{self.graph_id[:8]}"
        )
        self._worker_thread.start()
        logger.info(f"ZepGraphMemoryUpdater started: graph_id={self.graph_id}")
    
    def stop(self):
        """Drain worker, flush remaining events, and wait for Cloud graph ingestion."""
        deadline = time.time() + ZEP_INGESTION_WAIT_TIMEOUT_SECONDS
        with self._acceptance_lock:
            self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            join_timeout = max(0.0, deadline - time.time())
            self._worker_thread.join(timeout=join_timeout)
            if self._worker_thread.is_alive():
                raise TimeoutError(
                    f"Zep updater worker did not stop within {join_timeout:.0f}s"
                )

        self._flush_remaining(deadline=deadline)

        if self._failed_batches:
            raise RuntimeError(
                f"{len(self._failed_batches)} Zep activity batch(es) failed; "
                "simulation graph ingestion is incomplete"
            )

        self._wait_for_pending_episodes(deadline=deadline)
        
        logger.info(f"ZepGraphMemoryUpdater stopped: graph_id={self.graph_id}, "
                    f"total_activities={self._total_activities}, "
                    f"batches_sent={self._total_sent}, "
                    f"items_sent={self._total_items_sent}, "
                    f"failed={self._failed_count}, "
                    f"skipped={self._skipped_count}")
    
    def add_activity(self, activity: AgentActivity):
        """Enqueue agent activity for ingestion."""
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return

        with self._acceptance_lock:
            if not self._running:
                raise RuntimeError("Zep graph updater is not running")
            self._activity_queue.put(activity)
            self._total_activities += 1
        logger.debug(f"Queued activity for Zep: {activity.agent_name} - {activity.action_type}")
    
    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """Parse dictionary item from actions.jsonl and enqueue activity."""
        if "event_type" in data:
            return
        if data.get("success") is False:
            self._skipped_count += 1
            return
        
        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )
        
        self.add_activity(activity)
    
    def _worker_loop(self, locale: str = 'en'):
        """Background worker loop - batches activities and flushes to Zep."""
        set_locale(locale)
        while self._running or not self._activity_queue.empty():
            try:
                try:
                    activity = self._activity_queue.get(timeout=1)
                    platform = activity.platform.lower()
                    batch = None
                    with self._buffer_lock:
                        if platform not in self._platform_buffers:
                            self._platform_buffers[platform] = []
                        self._platform_buffers[platform].append(activity)
                        
                        if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                            batch = self._platform_buffers[platform][:self.BATCH_SIZE]
                            self._platform_buffers[platform] = self._platform_buffers[platform][self.BATCH_SIZE:]

                    if batch:
                        self._send_batch_activities(batch, platform)
                        time.sleep(self.SEND_INTERVAL)
                    
                except Empty:
                    pass
                    
            except Exception as e:
                logger.error(f"Worker loop exception: {e}")
                time.sleep(1)
    
    def _build_episode_payloads(
        self,
        activities: List[AgentActivity],
    ) -> List[tuple[List[AgentActivity], str]]:
        payloads: List[tuple[List[AgentActivity], str]] = []
        current_activities: List[AgentActivity] = []
        current_lines: List[str] = []
        current_length = 0

        for activity in activities:
            text = activity.to_episode_text()
            if len(text) > self.MAX_EPISODE_CHARS:
                marker = "... [truncated by Black Sheep Engine]"
                text = text[: self.MAX_EPISODE_CHARS - len(marker)] + marker
            projected_length = current_length + (1 if current_lines else 0) + len(text)
            if current_lines and projected_length > self.MAX_EPISODE_CHARS:
                payloads.append((current_activities, "\n".join(current_lines)))
                current_activities = []
                current_lines = []
                current_length = 0
            current_activities.append(activity)
            current_lines.append(text)
            current_length += (1 if len(current_lines) > 1 else 0) + len(text)

        if current_lines:
            payloads.append((current_activities, "\n".join(current_lines)))
        return payloads

    def _send_batch_activities(
        self,
        activities: List[AgentActivity],
        platform: str,
        *,
        deadline: float | None = None,
    ) -> int:
        """Batch ingest activity payloads into Zep graph."""
        if not activities:
            return 0

        processed_count = 0
        for payload_activities, combined_text in self._build_episode_payloads(activities):
            if deadline is not None and time.time() >= deadline:
                raise _DrainDeadlineExceeded(processed_count)
            try:
                episode = self.client.graph.add(
                    graph_id=self.graph_id,
                    type="text",
                    data=combined_text,
                    created_at=self._to_rfc3339(payload_activities[-1].timestamp),
                    source_description="Black Sheep simulation activity batch",
                    metadata={
                        "source": "blacksheep_simulation",
                        "simulation_id": self.simulation_id,
                        "platform": platform,
                        "activity_count": len(payload_activities),
                        "first_round": min(a.round_num for a in payload_activities),
                        "last_round": max(a.round_num for a in payload_activities),
                        "agent_ids": ",".join(
                            str(value)
                            for value in sorted({a.agent_id for a in payload_activities})
                        ),
                        "action_types": ",".join(
                            value
                            for value in sorted({a.action_type for a in payload_activities})
                            if value
                        ) or "unknown",
                    },
                )

                episode_uuid = (
                    getattr(episode, "uuid_", None)
                    or getattr(episode, "uuid", None)
                )
                if not episode_uuid:
                    raise RuntimeError("Zep graph.add returned no episode UUID")
                self._pending_episode_uuids.append(str(episode_uuid))
                self._total_sent += 1
                self._total_items_sent += len(payload_activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(f"Successfully dispatched {len(payload_activities)} {display_name} activities to graph {self.graph_id}")
                logger.debug(f"Batch content preview: {combined_text[:200]}...")

            except Exception as e:
                logger.error(f"Failed to batch dispatch activities to Zep, did not auto-replay non-idempotent write: {e}")
                self._failed_count += 1
                self._failed_batches.append({
                    "platform": platform,
                    "activities": payload_activities,
                    "error": str(e),
                })
            finally:
                processed_count += len(payload_activities)
        return processed_count

    @staticmethod
    def _to_rfc3339(value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.astimezone()
            return parsed.isoformat()
        except (AttributeError, TypeError, ValueError):
            return datetime.now().astimezone().isoformat()

    def _flush_remaining(self, *, deadline: float | None = None):
        """Flush remaining buffered events to Zep."""
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                platform = activity.platform.lower()
                with self._buffer_lock:
                    if platform not in self._platform_buffers:
                        self._platform_buffers[platform] = []
                    self._platform_buffers[platform].append(activity)
            except Empty:
                break
        
        for platform in list(self._platform_buffers):
            with self._buffer_lock:
                buffer = list(self._platform_buffers.get(platform, []))
            if not buffer:
                continue
            display_name = self._get_platform_display_name(platform)
            logger.info(f"Flushing {len(buffer)} remaining activities for {display_name}")
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(
                    "Zep updater drain deadline elapsed before flushing all activities"
                )
            try:
                processed_count = self._send_batch_activities(
                    buffer,
                    platform,
                    deadline=deadline,
                )
            except _DrainDeadlineExceeded as error:
                with self._buffer_lock:
                    del self._platform_buffers[platform][:error.processed_count]
                raise TimeoutError(str(error)) from error
            else:
                with self._buffer_lock:
                    del self._platform_buffers[platform][:processed_count]

    def _wait_for_pending_episodes(self, *, deadline: float | None = None) -> None:
        pending = set(self._pending_episode_uuids)
        if not pending:
            return

        if deadline is None:
            deadline = time.time() + ZEP_INGESTION_WAIT_TIMEOUT_SECONDS
        while pending:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Zep simulation ingestion timed out with {len(pending)} "
                    "episode(s) pending"
                )
            for episode_uuid in list(pending):
                episode = call_zep_read_with_retry(
                    lambda: self.client.graph.episode.get(uuid_=episode_uuid),
                    operation_name=f"poll simulation episode {episode_uuid}",
                )
                if getattr(episode, "processed", False):
                    pending.remove(episode_uuid)
            if pending:
                time.sleep(3)
        self._pending_episode_uuids = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Retrieve telemetry and operational stats."""
        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}
        
        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,
            "batches_sent": self._total_sent,
            "items_sent": self._total_items_sent,
            "failed_count": self._failed_count,
            "pending_episode_count": len(self._pending_episode_uuids),
            "skipped_count": self._skipped_count,
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,
            "running": self._running,
        }


class ZepGraphMemoryManager:
    """
    Registry and lifecycle manager for Zep Graph Memory Updaters across simulation runs.
    """
    
    _updaters: Dict[str, ZepGraphMemoryUpdater] = {}
    _lock = threading.Lock()
    
    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> ZepGraphMemoryUpdater:
        """Create and start memory updater for a simulation."""
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
            
            updater = ZepGraphMemoryUpdater(
                graph_id,
                simulation_id=simulation_id,
            )
            updater.start()
            cls._updaters[simulation_id] = updater
            cls._stop_all_done = False
            
            logger.info(f"Created graph memory updater: simulation_id={simulation_id}, graph_id={graph_id}")
            return updater
    
    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[ZepGraphMemoryUpdater]:
        with cls._lock:
            return cls._updaters.get(simulation_id)

    @classmethod
    def get_simulation_ids_for_graph(cls, graph_id: str) -> List[str]:
        """Return simulations whose updater still owns or drains this graph."""
        with cls._lock:
            return sorted(
                simulation_id
                for simulation_id, updater in cls._updaters.items()
                if updater.graph_id == graph_id
            )

    @classmethod
    def get_simulation_ids(cls) -> List[str]:
        with cls._lock:
            return sorted(cls._updaters)

    @classmethod
    def discard_inactive_updater(cls, simulation_id: str) -> bool:
        """Discard an inactive updater during explicit graph deletion."""
        with cls._lock:
            updater = cls._updaters.get(simulation_id)
            if updater is None:
                return False
            worker_alive = bool(
                updater._worker_thread and updater._worker_thread.is_alive()
            )
            if updater._running or worker_alive:
                raise RuntimeError(
                    f"Zep updater for {simulation_id} is still active"
                )
            cls._updaters.pop(simulation_id, None)
        logger.warning(
            "Discarded inactive Zep updater during explicit graph deletion: "
            "simulation_id=%s, graph_id=%s",
            simulation_id,
            updater.graph_id,
        )
        return True
    
    @classmethod
    def stop_updater(cls, simulation_id: str):
        """Stop and unregister updater for a simulation."""
        with cls._lock:
            updater = cls._updaters.get(simulation_id)
        if updater is None:
            return

        updater.stop()
        with cls._lock:
            if cls._updaters.get(simulation_id) is updater:
                cls._updaters.pop(simulation_id, None)
        logger.info(f"Stopped graph memory updater: simulation_id={simulation_id}")
    
    _stop_all_done = False
    
    @classmethod
    def stop_all(cls):
        """Stop all active updaters."""
        if cls._stop_all_done:
            return

        with cls._lock:
            simulation_ids = list(cls._updaters)

        errors = []
        for simulation_id in simulation_ids:
            try:
                cls.stop_updater(simulation_id)
            except Exception as error:
                logger.error(
                    "Failed to stop updater: simulation_id=%s, error=%s",
                    simulation_id,
                    error,
                )
                errors.append((simulation_id, error))

        with cls._lock:
            cls._stop_all_done = not cls._updaters

        if errors:
            details = "; ".join(
                f"{simulation_id}: {error}"
                for simulation_id, error in errors
            )
            raise RuntimeError(f"Some graph updaters failed to stop cleanly: {details}")
        logger.info("Stopped all graph memory updaters")
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        return {
            sim_id: updater.get_stats() 
            for sim_id, updater in cls._updaters.items()
        }
