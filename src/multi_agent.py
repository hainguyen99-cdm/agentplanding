"""Multi-agent management: AgentManager, GroupManager, Orchestrator"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import date, timedelta

from config import get_config
from agent import AIAgent
from vector_db import VectorDB
from composite_vector_db import CompositeVectorDB


@dataclass
class AgentMeta:
    agent_id: str
    name: str
    role: str = "general"
    language: str = "vi"
    personality: str = "friendly"
    speaking_style: str = "natural"


class AgentManager:
    """CRUD agents and load their instances on demand"""

    def __init__(self, root: Optional[str] = None):
        cfg = get_config()
        self.root = Path(root or cfg.multi_agent.agents_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "agents.json"
        self.shared_db_path = Path(cfg.multi_agent.shared_db_path)
        self._agents: Dict[str, AgentMeta] = {}
        self._load()

    def _load(self):
        if self.meta_path.exists():
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
            for a in data.get("agents", []):
                meta = AgentMeta(**a)
                self._agents[meta.agent_id] = meta
        else:
            self._save()

    def _save(self):
        payload = {"agents": [asdict(a) for a in self._agents.values()]}
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_agents(self) -> List[AgentMeta]:
        return list(self._agents.values())

    def get(self, agent_id: str) -> Optional[AgentMeta]:
        return self._agents.get(agent_id)

    def create(self, meta: AgentMeta) -> bool:
        if meta.agent_id in self._agents:
            return False
        self._agents[meta.agent_id] = meta
        # create private db folder
        agent_db = self.root / meta.agent_id / "vector_db"
        agent_db.mkdir(parents=True, exist_ok=True)
        self._save()
        return True

    def delete(self, agent_id: str) -> bool:
        if agent_id not in self._agents:
            return False
        del self._agents[agent_id]
        self._save()
        return True

    def instantiate(self, agent_id: str) -> AIAgent:
        meta = self.get(agent_id)
        if not meta:
            raise ValueError(f"Agent {agent_id} not found")
        # Build AIAgent with private + shared vector stores
        cfg = get_config()
        private_path = str(self.root / agent_id / "vector_db")
        private_db = VectorDB(db_path=private_path)
        shared_db = None
        composite = None
        if cfg.multi_agent.shared_enabled:
            self.shared_db_path.mkdir(parents=True, exist_ok=True)
            shared_db = VectorDB(db_path=str(self.shared_db_path))
            composite = CompositeVectorDB([private_db, shared_db], store_policy=cfg.multi_agent.share_policy)
        agent = AIAgent(agent_id=agent_id, role=meta.role, vector_db=composite or private_db)
        # override agent visible profile
        agent.update_config(
            name=meta.name,
            language=meta.language,
            personality=meta.personality,
            speaking_style=meta.speaking_style
        )
        return agent


@dataclass
class Group:
    group_id: str
    name: str
    members: List[str]  # agent_ids
    roles: Dict[str, str] = field(default_factory=dict)  # agent_id -> role in team (e.g., market_analyst, content_planner)
    executor_id: Optional[str] = None  # which agent will produce final output
    action: Optional[Dict[str, Any]] = None  # e.g., {"type": "mongodb"|"postgresql"|"none", "execute": False, "target": {"collection": "content_plans"}}


class GroupManager:
    def __init__(self, root: Optional[str] = None):
        cfg = get_config()
        self.root = Path(root or cfg.multi_agent.agents_root)
        self.meta_path = self.root / "groups.json"
        self._groups: Dict[str, Group] = {}
        self._load()

    def _load(self):
        if self.meta_path.exists():
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
            for g in data.get("groups", []):
                grp = Group(**g)
                self._groups[grp.group_id] = grp
        else:
            self._save()

    def _save(self):
        payload = {"groups": [asdict(g) for g in self._groups.values()]}
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_groups(self) -> List[Group]:
        return list(self._groups.values())

    def create(self, grp: Group) -> bool:
        if grp.group_id in self._groups:
            return False
        self._groups[grp.group_id] = grp
        self._save()
        return True

    def delete(self, group_id: str) -> bool:
        if group_id not in self._groups:
            return False
        del self._groups[group_id]
        self._save()
        return True


class MultiAgentOrchestrator:
    """Run multi-agent sessions with shared knowledge via shared DB"""

    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager

    def run_session(self, group: Group, task_prompt: str, rounds: int = 2, final_only: bool = False, auto_finalize: bool = False) -> Dict:
        # If only final output is needed, skip dialogue rounds
        if final_only:
            rounds = 0
        cfg = get_config()
        if auto_finalize:
            # Ignore rounds; use configured max_auto_rounds
            rounds = getattr(cfg.multi_agent, 'max_auto_rounds', 6) # Increased from 20 to a more reasonable default
        # instantiate all members
        agents = [self.agent_manager.instantiate(aid) for aid in group.members]
        transcript: List[Tuple[str, str]] = []  # (speaker, message)

        # seed
        current_msg = task_prompt
        speaker_idx = 0

        # config
        cfg = get_config()
        history_window = getattr(cfg.multi_agent, 'turn_history_window', 6)
        max_reply_chars = getattr(cfg.multi_agent, 'max_reply_chars', 2000)

        def _sanitize(x):
            if isinstance(x, str):
                return x
            if x is None:
                return ""
            try:
                return str(x)
            except Exception:
                return ""

        r = 0
        while r < rounds:
            speaker = agents[speaker_idx]
            speaker_name = speaker.config.agent.name
            speaker_role = speaker.config.agent.role
            speaker_style = speaker.config.agent.speaking_style

            prev_speaker = transcript[-1][0] if transcript else "Người dùng"
            next_agent = agents[(speaker_idx + 1) % len(agents)]
            next_name = next_agent.config.agent.name

            # Format recent transcript (last N messages)
            recent = transcript[-history_window:]
            transcript_str = "\n".join([f"- {who}: {_sanitize(msg)}" for who, msg in recent]) if recent else "(chưa có)"

            # Compose turn prompt to stimulate real exchange
            team_roles = []
            for a in agents:
                aid = a.agent_id or a.config.agent.name
                role = group.roles.get(aid, a.config.agent.role) if hasattr(group, 'roles') and group.roles else a.config.agent.role
                team_roles.append(f"{a.config.agent.name}=>{role}")
            prompt_parts = [
                f"Bạn đang tham gia một phiên thảo luận nhiều agent về chủ đề sau (trả lời bằng tiếng Việt):\n",
                f"Nhiệm vụ: {task_prompt}\n\n",
                f"Thành viên: {', '.join([a.config.agent.name for a in agents])}\n",
                f"Vai trò nhóm: {', '.join(team_roles)}\n\n",
                f"Bối cảnh gần đây:\n{transcript_str}\n\n",
                f"Bạn là {speaker_name} (vai trò: {speaker_role}, phong cách: {speaker_style}).\n",
                f"Hãy phản hồi trực tiếp {prev_speaker} bằng 3–5 câu, tuân thủ:\n",
                f"- Thêm ÍT NHẤT 1 ý mới chưa được nêu, hạn chế lặp lại;\n",
                f"- Nếu có thể, viện dẫn kiến thức liên quan từ bộ nhớ (shared/riêng);\n"
            ]

            is_executor = group.executor_id and speaker.agent_id == group.executor_id

            if is_executor:
                prompt_parts.append("- Nếu bạn là executor của nhóm, hãy tổng hợp và chốt kế hoạch cuối cùng nếu cảm thấy đã đủ dữ kiện;\n")
            else:
                prompt_parts.append(f"- Kết thúc bằng một câu hỏi ngắn gửi tới {next_name} để tiếp tục;\n")

            prompt_parts.append("- Ngắn gọn, rõ ràng, không liệt kê trùng lặp.\n")

            if auto_finalize:
                if is_executor:
                    prompt_parts.append("\nNếu bạn cho rằng nhóm đã sẵn sàng chốt, hãy thêm token [READY_TO_FINALIZE] ở cuối câu trả lời.")
                else:
                    prompt_parts.append("\nLưu ý: Chỉ executor mới có quyền chốt kế hoạch.")

            turn_prompt = "".join(prompt_parts)

            # Temporarily increase max_tokens for multi-agent sessions to get complete responses
            original_max_tokens = speaker.config.openai.max_tokens
            speaker.config.openai.max_tokens = 4096

            result = speaker.process_message(turn_prompt, extract_knowledge=False)
            if result.get("error"):
                reply = f"(Lỗi khi gọi model: {result.get('error')})"
            else:
                reply = result.get("response", "")
            reply = _sanitize(reply)
            if max_reply_chars and len(reply) > max_reply_chars:
                reply = reply[:max_reply_chars] + "…"

            # Restore original max_tokens
            speaker.config.openai.max_tokens = original_max_tokens

            transcript.append((speaker_name, reply))
            # Prepare next turn
            current_msg = reply
            speaker_idx = (speaker_idx + 1) % len(agents)
            r += 1

            # Auto-finalize attempt: only the designated executor can trigger it
            is_executor = group.executor_id and speaker.agent_id == group.executor_id
            ready_signal = "[READY_TO_FINALIZE]" in reply
            if auto_finalize and is_executor and ready_signal:
                # Try to synthesize final only when team signals readiness
                synth_ok, final_obj = self._synthesize_final(group, agents, transcript, history_window, task_prompt)
                if synth_ok:
                    result_dict = {
                        "transcript": transcript,
                        "agents": [a.config.agent.name for a in agents],
                        "final": final_obj
                    }
                    # Optional action execution
                    if getattr(group, 'action', None) and isinstance(group.action, dict):
                        action_res = self._execute_action(group.action, final_obj)
                        result_dict["action_result"] = action_res
                    return result_dict

        result: Dict[str, Any] = {
            "transcript": transcript,
            "agents": [a.config.agent.name for a in agents]
        }

        # Finalization policy:
        # - final_only: always synthesize now
        # - auto_finalize: handled in-loop via [READY_TO_FINALIZE]; as fallback, we can attempt once here
        # - otherwise: ALWAYS synthesize after rounds are complete.
        # This ensures a plan is always generated.
        if not result.get("final"): # Don't re-synthesize if auto-finalize already succeeded
            synth_ok, final_obj = self._synthesize_final(group, agents, transcript, history_window, task_prompt)
            if synth_ok:
                result["final"] = final_obj
                if getattr(group, 'action', None) and isinstance(group.action, dict):
                    action_res = self._execute_action(group.action, final_obj)
                    result["action_result"] = action_res


        return result

    def _synthesize_final(self, group: Group, agents: List[AIAgent], transcript: List[Tuple[str, str]], history_window: int, task_prompt: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        def _sanitize(x):
            if isinstance(x, str):
                return x
            if x is None:
                return ""
            try:
                return str(x)
            except Exception:
                return ""
        # Find executor
        exec_agent = None
        if getattr(group, 'executor_id', None):
            for a in agents:
                if a.agent_id == group.executor_id:
                    exec_agent = a
                    break
        if exec_agent is None and agents:
            exec_agent = agents[0]
        if exec_agent is None:
            return False, None
        # Schema and timeframe hints
        schema = {
            "task": "web3_market_content_plan",
            "timeframe": "string (ví dụ: 2025-12-23..2025-12-29)",
            "network": "string (ví dụ: ethereum)",
            "summary": "string tóm tắt phân tích thị trường",
            "posts": [
                {
                    "date": "YYYY-MM-DD",
                    "title": "string",
                    "summary": "string",
                    "outline": ["bullet 1", "bullet 2"],
                    "channels": ["twitter", "telegram", "blog"],
                    "target_audience": "string",
                    "cta": "string"
                }
            ]
        }
        cfg = get_config()
        recent = "\n".join([f"- {who}: {_sanitize(msg)}" for who, msg in transcript[-history_window:]]) if transcript else "(chưa có)"
        
        # Build prompt with a CORRECT timeframe hint for the future
        prompt_lines = [
            f"Bạn là {exec_agent.config.agent.name}, executor của nhóm.",
            f"Nhiệm vụ: Sinh ra JSON kế hoạch nội dung dựa trên chủ đề: {task_prompt}.",
            f"Yêu cầu: Chỉ trả về JSON hợp lệ (không kèm giải thích).",
            f"Schema mẫu:\n{json.dumps(schema, ensure_ascii=False, indent=2)}",
            f"\nBối cảnh gần đây:\n{recent}\n"
        ]

        # Generate a correct timeframe hint for the next 7 days
        today = date.today()
        next_week_start = today + timedelta(days=1)
        next_week_end   = today + timedelta(days=7)
        timeframe_hint  = f"{next_week_start.isoformat()}..{next_week_end.isoformat()}"
        prompt_lines.append(
            f"Gợi ý: Thời gian cho kế hoạch nên là 'tuần tới' (ví dụ: {timeframe_hint}). Nếu đề bài yêu cầu khác, hãy tuân theo đề bài."
        )

        prompt_lines.extend([
            "Hãy điền network, summary ngắn, và danh sách posts chi tiết.",
            "QUAN TRỌNG: Kết quả PHẢI chứa một danh sách 'posts' không rỗng, mỗi post là một object chứa đầy đủ các trường theo schema."
        ])
        synth_prompt = "\n".join(prompt_lines)
        
        # Call model
        orig = exec_agent.config.openai.max_tokens
        exec_agent.config.openai.max_tokens = 4096
        synth_res = exec_agent.process_message(synth_prompt, extract_knowledge=False)
        exec_agent.config.openai.max_tokens = orig
        if synth_res.get("error"):
            return False, None
        raw = synth_res.get("response", "")
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
        if not isinstance(parsed, dict):
            return False, None
        
        # Basic validation & ID injection
        if "task" not in parsed or "posts" not in parsed or not isinstance(parsed.get("posts"), list) or len(parsed["posts"]) == 0:
            return False, None

        # Inject a simple, unique post_id for linking articles later
        for i, post in enumerate(parsed["posts"]):
            if not isinstance(post, dict):
                continue # Skip malformed entries
            post["post_id"] = f"post_{i}"
        
        # If the model didn't generate a timeframe, fill it with our future-dated hint
        if timeframe_hint and not parsed.get("timeframe"):
            parsed["timeframe"] = timeframe_hint
            
        return True, parsed

    def _execute_action(self, action: Dict[str, Any], data: Any) -> Dict[str, Any]:
        cfg = get_config()
        if not getattr(cfg, 'database', None) or not cfg.database.allow_actions:
            return {"status": "skipped", "reason": "actions disabled by config"}
        if not action or not isinstance(action, dict):
            return {"status": "error", "error": "invalid action config"}
        a_type = action.get("type", cfg.database.default_action)
        execute = action.get("execute", False)
        target = action.get("target", {}) or {}
        dry_run = not execute
        try:
            if a_type == "mongodb":
                try:
                    from pymongo import MongoClient
                except Exception as e:
                    return {"status": "error", "error": f"pymongo not available: {e}"}
                uri = cfg.database.mongo_uri
                dbname = target.get("db") or cfg.database.mongo_db
                coll = target.get("collection") or cfg.database.mongo_collection
                if dry_run:
                    return {"status": "dry_run", "driver": "mongodb", "db": dbname, "collection": coll, "sample": data}
                client = MongoClient(uri)
                db = client[dbname]
                col = db[coll]
                if isinstance(data, dict):
                    res = col.insert_one(data)
                    return {"status": "ok", "driver": "mongodb", "inserted_id": str(res.inserted_id)}
                elif isinstance(data, list):
                    res = col.insert_many(data)
                    return {"status": "ok", "driver": "mongodb", "inserted_ids": [str(x) for x in res.inserted_ids]}
                else:
                    return {"status": "error", "error": "unsupported data type"}
            elif a_type == "postgresql":
                try:
                    import psycopg2
                    import json as _json
                except Exception as e:
                    return {"status": "error", "error": f"psycopg2 not available: {e}"}
                dsn = cfg.database.postgres_dsn
                table = target.get("table", "content_plans")
                if dry_run:
                    return {"status": "dry_run", "driver": "postgresql", "table": table, "sample": data}
                conn = psycopg2.connect(dsn)
                cur = conn.cursor()
                # store as JSONB in a generic table (id serial, payload jsonb)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id SERIAL PRIMARY KEY,
                        payload JSONB
                    )
                """)
                cur.execute(f"INSERT INTO {table} (payload) VALUES (%s) RETURNING id", (_json.dumps(data),))
                new_id = cur.fetchone()[0]
                conn.commit()
                cur.close()
                conn.close()
                return {"status": "ok", "driver": "postgresql", "id": new_id}
            else:
                return {"status": "skipped", "reason": f"unknown action type: {a_type}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

