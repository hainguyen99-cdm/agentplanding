"""Gradio UI for AI Agent with Multi-Agent Dashboard"""
"""Gradio UI for AI Agent with Multi-Agent Dashboard"""
import gradio as gr
from typing import Tuple, List, Optional, Dict, Any
import json
import os
from datetime import datetime
import uuid

from file_extractors import extract_text_from_file
from pymongo import MongoClient
from bson.objectid import ObjectId

from agent import AIAgent
from tools import ToolManager
from config import get_config
from vector_db import VectorDB
from multi_agent import AgentManager, GroupManager, MultiAgentOrchestrator, AgentMeta, Group


class AgentUI:
    """Gradio UI for AI Agent with optional multi-agent management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Managers
        self.cfg = get_config()
        self.agent_manager = AgentManager()
        self.group_manager = GroupManager()
        self.orchestrator = MultiAgentOrchestrator(self.agent_manager)
        
        # Ensure there is at least one agent
        existing = self.agent_manager.list_agents()
        if not existing:
            default_meta = AgentMeta(
                agent_id="agent_default",
                name=self.cfg.agent.name,
                role=self.cfg.agent.role,
                language=self.cfg.agent.language,
                personality=self.cfg.agent.personality,
                speaking_style=self.cfg.agent.speaking_style,
            )
            self.agent_manager.create(default_meta)
            existing = self.agent_manager.list_agents()
        
        # Active agent
        self.active_agent_id = existing[0].agent_id
        self.agent = self.agent_manager.instantiate(self.active_agent_id)
        
        # Tool manager bound to active agent
        self.tool_manager = ToolManager(self.agent.rag_pipeline)
        self.chat_history = []
        self.current_plan_state = []

        # MongoDB connection (optional)
        self.mongo_client = None
        self.mongo_db = None
        self.content_plans_col = None
        self.articles_col = None
        try:
            cfg = self.cfg
            if getattr(cfg, "database", None) and cfg.database.mongo_uri:
                # cfg.database.mongo_uri is loaded from config.yaml or .env (.env: MONGODB_URL)
                print(f"[DB] Connecting MongoDB: uri={cfg.database.mongo_uri}")
                self.mongo_client = MongoClient(cfg.database.mongo_uri)
                # Use configured DB name if provided, otherwise default
                # User expects db=agentsocial
                db_name = cfg.database.mongo_db or "agentsocial"
                self.mongo_db = self.mongo_client[db_name]
                # Option A: store plan + embedded posts in one collection
                self.content_plans_col = self.mongo_db["content_plans"]
                # Store articles separately (linked by plan_id + post_id)
                self.articles_col = self.mongo_db["articles"]
        except Exception as e:
            print(f"[WARN] MongoDB not available: {e}")
    
    # ------------- Chat -------------
    def chat(self, message: str, show_context: bool = True) -> Tuple[str, str]:
        if not message.strip():
            return "", "Please enter a message"
        
        result = self.agent.process_message(message)
        
        if result["error"]:
            return f"Error: {result['error']}", ""
        
        response = result["response"]
        
        # Build context info
        context_info = ""
        if show_context and result["rag_context"]:
            context = result["rag_context"]
            context_info = f"📚 Retrieved {context['retrieved_count']} knowledge entries:\n\n"
            for i, entry in enumerate(context["entries"], 1):
                context_info += f"{i}. (relevance: {entry['similarity']:.2f}) {entry['content'][:100]}...\n"
        
        # Add knowledge extraction info
        if result["knowledge_extraction"]:
            ke = result["knowledge_extraction"]
            if ke["stored"]:
                context_info += f"\n✅ Stored {len(ke['stored'])} new knowledge entries"
            if ke["rejected"]:
                context_info += f"\n⚠️ Rejected {len(ke['rejected'])} entries (duplicates/low confidence)"
        
        return response, context_info
    
    # ------------- Config ops -------------
    def update_agent_config(
        self,
        name: str,
        age: int,
        gender: str,
        language: str,
        personality: str,
        speaking_style: str,
        role: str,
    ) -> str:
        try:
            self.agent.update_config(
                name=name,
                age=age,
                gender=gender,
                language=language,
                personality=personality,
                speaking_style=speaking_style,
                role=role,
            )
            # Also persist meta if matches active agent in manager
            meta = self.agent_manager.get(self.active_agent_id)
            if meta:
                meta.name = name
                meta.role = role
                meta.language = language
                meta.personality = personality
                meta.speaking_style = speaking_style
                # Save back
                self.agent_manager._save()
            return f"✅ Configuration updated for {name}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def get_knowledge_stats(self) -> str:
        stats = self.agent.get_knowledge_stats()
        return f"""📊 Knowledge Base Statistics (Agent: {self.active_agent_id}):
- Total entries: {stats['total_entries']}
- Active entries: {stats['active_entries']}
- Index size: {stats['index_size']}
- Database path: {stats['db_path']}"""
    
    def add_manual_knowledge(self, content: str, source: str = "manual") -> str:
        # Backwards-compatible: adds to active agent
        if not content.strip():
            return "❌ Please enter knowledge content"
        success, result = self.agent.add_knowledge(content, source=source)
        if success:
            return f"✅ Knowledge added with ID: {result}"
        else:
            return f"❌ Failed to add knowledge: {result}"

    def add_manual_knowledge_to_agent(self, agent_id: str, content: str, source: str = "manual") -> str:
        """Add manual knowledge to selected agent (agent_id)."""
        if not agent_id:
            return "❌ agent_id is required"
        if not content or not content.strip():
            return "❌ Please enter knowledge content"
        try:
            agent = self.agent_manager.instantiate(agent_id)
            ok, entry_id = agent.add_knowledge(content, source=source)
            return f"✅ Added to {agent_id} (id={entry_id})" if ok else f"❌ Failed: {entry_id}"
        except Exception as e:
            return f"❌ Error: {e}"

    def add_file_knowledge_to_agent(self, agent_id: str, file_obj, source: str = "upload") -> str:
        """Extract text from uploaded file and add to selected agent."""
        if not agent_id:
            return "❌ agent_id is required"
        if file_obj is None:
            return "❌ Please upload a file"

        file_path = getattr(file_obj, "name", None) or str(file_obj)
        ok, text_or_err = extract_text_from_file(file_path)
        if not ok:
            return f"❌ {text_or_err}"

        text = (text_or_err or "").strip()
        if not text:
            return "❌ Could not extract any text from file"

        try:
            agent = self.agent_manager.instantiate(agent_id)
            ok2, entry_id = agent.add_knowledge(text, source=f"{source}:{os.path.basename(file_path)}")
            return f"✅ Added file to {agent_id} (id={entry_id}, chars={len(text)})" if ok2 else f"❌ Failed: {entry_id}"
        except Exception as e:
            return f"❌ Error: {e}"
    
    def clear_knowledge(self) -> str:
        self.agent.clear_knowledge()
        return "✅ Knowledge base cleared"
    
    def clear_conversation(self) -> str:
        self.agent.clear_conversation()
        self.chat_history = []
        return "✅ Conversation cleared"
    
    def get_agent_config(self) -> str:
        config = self.agent.get_config()
        return json.dumps(config, ensure_ascii=False, indent=2)
    
    # ------------- Multi-agent ops -------------
    def list_agents_json(self) -> str:
        agents = [a.__dict__ for a in self.agent_manager.list_agents()]
        return json.dumps(agents, ensure_ascii=False, indent=2)
    
    def create_agent(self, agent_id: str, name: str, role: str, language: str, personality: str, speaking_style: str) -> str:
        if not agent_id.strip() or not name.strip():
            return "❌ agent_id and name are required"
        ok = self.agent_manager.create(AgentMeta(
            agent_id=agent_id.strip(),
            name=name.strip(),
            role=role.strip() or "general",
            language=language.strip() or "vi",
            personality=personality.strip() or "friendly",
            speaking_style=speaking_style.strip() or "natural",
        ))
        return "✅ Agent created" if ok else "❌ Agent already exists"
    
    def set_active_agent(self, agent_id: str) -> str:
        try:
            self.active_agent_id = agent_id
            self.agent = self.agent_manager.instantiate(agent_id)
            # reset tools to new agent RAG pipeline
            self.tool_manager = ToolManager(self.agent.rag_pipeline)
            return f"✅ Active agent set to {agent_id}"
        except Exception as e:
            return f"❌ {e}"
    
    def delete_agent(self, agent_id: str) -> str:
        ok = self.agent_manager.delete(agent_id)
        return "✅ Agent deleted" if ok else "❌ Agent not found"
    
    def list_groups_json(self) -> str:
        groups = [g.__dict__ for g in self.group_manager.list_groups()]
        return json.dumps(groups, ensure_ascii=False, indent=2)
    
    def create_group(self, group_id: str, name: str, members_csv: str, roles_json: str = "{}", executor_id: str = "", action_type: str = "none", execute_action: bool = False, target_json: str = "{}") -> str:
        members = [m.strip() for m in members_csv.split(',') if m.strip()]
        if not group_id or not name:
            return "❌ group_id and name are required"
        # parse roles and target
        try:
            roles = json.loads(roles_json) if roles_json.strip() else {}
            if not isinstance(roles, dict):
                return "❌ roles_json must be a JSON object mapping agent_id -> role"
        except Exception as e:
            return f"❌ Invalid roles_json: {e}"
        try:
            target = json.loads(target_json) if target_json.strip() else {}
            if not isinstance(target, dict):
                return "❌ target_json must be a JSON object"
        except Exception as e:
            return f"❌ Invalid target_json: {e}"
        action = None
        if action_type and action_type != "none":
            action = {"type": action_type, "execute": bool(execute_action), "target": target}
        grp = Group(group_id=group_id, name=name, members=members, roles=roles, executor_id=executor_id or None, action=action)
        ok = self.group_manager.create(grp)
        return "✅ Group created" if ok else "❌ Group already exists"
    
    def delete_group(self, group_id: str) -> str:
        ok = self.group_manager.delete(group_id)
        return "✅ Group deleted" if ok else "❌ Group not found"
    
    def run_multi_agent(self, group_id: str, prompt: str, rounds: int, final_only: bool = False, auto_finalize: bool = False) -> Tuple[str, Optional[List[Dict[str, Any]]], Any, Any]:
        try:
            cfg = self.cfg
            try:
                rounds_int = int(rounds)
            except Exception:
                rounds_int = 1
            rounds_int = max(1, min(rounds_int, getattr(cfg.multi_agent, 'max_rounds', 12)))

            grp = self.group_manager._groups.get(group_id)
            if not grp:
                return "❌ Group not found", None, gr.update(visible=False)

            result = self.orchestrator.run_session(grp, prompt, rounds=rounds_int, final_only=final_only, auto_finalize=auto_finalize)

            def _sanitize(x):
                return str(x) if x is not None else ""

            agent_names = ", ".join([_sanitize(a) for a in result.get('agents', [])])
            lines = [f"Agents: {agent_names}", "Transcript:"]
            for speaker, msg in result.get("transcript", []):
                lines.append(f"\n{'='*60}\n🗣️ {_sanitize(speaker)}:\n{'='*60}\n{_sanitize(msg)}")

            final_plan = result.get("final")
            plan_data_for_df = []

            if final_plan and isinstance(final_plan.get("posts"), list):
                lines.append("\n\nFinal Plan Summary:")
                lines.append(f"- Task: {final_plan.get('task', 'N/A')}")
                lines.append(f"- Timeframe: {final_plan.get('timeframe', 'N/A')}")
                lines.append(f"- Summary: {final_plan.get('summary', 'N/A')}")
                self.current_plan_state = final_plan.get("posts", [])
                # Convert for DataFrame and ensure posts have post_id
                for i, post in enumerate(self.current_plan_state):
                    if not isinstance(post, dict):
                        continue
                    post["post_id"] = post.get("post_id", f"post_{i}")
                    plan_data_for_df.append([
                        i,
                        post.get("date", ""),
                        post.get("title", ""),
                        post.get("summary", ""),
                    ])



            if result.get("action_result") is not None:
                lines.append("\n\nAction Result:")
                lines.append(json.dumps(result["action_result"], ensure_ascii=False, indent=2))

            text = "\n".join(lines)
            max_chars = getattr(cfg.multi_agent, 'max_transcript_chars', 150000)
            if len(text) > max_chars:
                text = text[-max_chars:]
            
            print("[DEBUG] plan_data_for_df:", plan_data_for_df)
            print("[DEBUG] plan_data_for_df length:", len(plan_data_for_df))
            print("[DEBUG] Returning to Gradio...")
            # Return transcript and plan data
            return text, plan_data_for_df, final_plan
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"❌ Error running session: {e}", [], None
    
    def get_shared_stats(self) -> str:
        path = self.cfg.multi_agent.shared_db_path
        stats = VectorDB(db_path=path).get_stats()
        return f"""📚 Shared Knowledge Stats:
- Total entries: {stats['total_entries']}
- Active entries: {stats['active_entries']}
- Index size: {stats['index_size']}
- Path: {stats['db_path']}"""
    
    def add_shared_knowledge(self, content: str) -> str:
        if not content.strip():
            return "❌ Please enter knowledge content"
        path = self.cfg.multi_agent.shared_db_path
        ok, msg = VectorDB(db_path=path).add_entry(content=content, source="shared_manual", confidence=0.95)
        return "✅ Added to shared knowledge" if ok else f"❌ {msg}"

    # ------------- DB ops (MongoDB) -------------
    def _list_saved_plans(self):
        """Return dropdown choices of saved plans (plan_id)."""
        if self.content_plans_col is None:
            return gr.update(choices=[], value=None), "❌ MongoDB chưa sẵn sàng"
        try:
            docs = list(self.content_plans_col.find({}, {"plan_id": 1, "task": 1, "timeframe": 1, "created_at": 1}).sort("created_at", -1).limit(50))
            choices = []
            for d in docs:
                pid = d.get("plan_id")
                if not pid:
                    continue
                label = f"{pid} | {d.get('timeframe','')} | {d.get('task','')}"
                choices.append(label)
            return gr.update(choices=choices, value=(choices[0] if choices else None)), f"✅ Đã tải {len(choices)} kế hoạch"
        except Exception as e:
            return gr.update(choices=[], value=None), f"❌ Lỗi load danh sách: {e}"

    def _load_plan_from_db(self, selected_label: str):
        """Load a plan by selected dropdown label -> update plan_df/plan_state/full_plan_state/plan_id_state."""
        if not selected_label:
            return "❌ Chưa chọn plan", [], [], None, None, gr.update(choices=[], value=None)
        if self.content_plans_col is None:
            return "❌ MongoDB chưa sẵn sàng", [], [], None, None, gr.update(choices=[], value=None)
        plan_id = str(selected_label).split("|")[0].strip()
        try:
            doc = self.content_plans_col.find_one({"plan_id": plan_id})
            if not doc:
                return f"❌ Không tìm thấy plan_id={plan_id}", [], [], None, None, gr.update(choices=[], value=None)
            # Normalize
            posts = doc.get("posts", []) if isinstance(doc.get("posts"), list) else []
            for i, p in enumerate(posts):
                if isinstance(p, dict):
                    p["post_id"] = p.get("post_id", f"post_{i}")
            # Update current plan state for edit sync
            self.current_plan_state = posts

            plan_data_for_df = []
            for i, post in enumerate(posts):
                if not isinstance(post, dict):
                    continue
                plan_data_for_df.append([
                    i,
                    post.get("date", ""),
                    post.get("title", ""),
                    post.get("summary", ""),
                ])

            selector_choices = [f"#{row[0]}: {row[2][:50]}..." if len(row[2]) > 50 else f"#{row[0]}: {row[2]}" for row in plan_data_for_df]
            selector_update = gr.update(choices=selector_choices, value=None)

            return f"✅ Đã load plan_id={plan_id}", plan_data_for_df, plan_data_for_df, doc, plan_id, selector_update
        except Exception as e:
            return f"❌ Lỗi load plan: {e}", [], [], None, None, gr.update(choices=[], value=None)

    def _save_plan_to_db(self, full_plan: Optional[Dict[str, Any]]):
        """Save the full plan (including embedded posts) to MongoDB.

        Generates a new plan_id (uuid) each time user clicks save.
        Returns: (message, plan_id)
        """

        if not full_plan or not isinstance(full_plan, dict):
            return "❌ Chưa có kế hoạch để lưu (hãy Run Session trước)", None
        if self.content_plans_col is None:
            return "❌ MongoDB chưa được cấu hình/kết nối (kiểm tra MONGODB_URL và mongo_db)", None

        plan_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Ensure posts include post_id
        posts = full_plan.get("posts", [])
        if isinstance(posts, list):
            for i, p in enumerate(posts):
                if isinstance(p, dict) and not p.get("post_id"):
                    p["post_id"] = f"post_{i}"
                if isinstance(p, dict):
                    p.setdefault("images", [])

        doc = {
            "plan_id": plan_id,
            "task": full_plan.get("task"),
            "timeframe": full_plan.get("timeframe"),
            "network": full_plan.get("network"),
            "summary": full_plan.get("summary"),
            "posts": posts,
            "created_at": now,
            "updated_at": now,
        }
        try:
            self.content_plans_col.insert_one(doc)
            return f"✅ Đã lưu kế hoạch vào MongoDB (db=agentsocial, collection=content_plans)\nplan_id={plan_id}", plan_id
        except Exception as e:
            return f"❌ Lỗi lưu MongoDB: {e}", None
    
    # ------------- UI -------------
    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="AI Agent with Long-term Memory") as demo:
            gr.Markdown("# 🤖 AI Agent with Long-term Memory + Multi-Agent Dashboard")
            gr.Markdown("OpenAI integration, RAG, Vector DB, and Multi-Agent orchestration")
            
            with gr.Tabs() as tabs:
                # Chat Tab
                with gr.Tab("💬 Chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(label="Conversation", height=400)
                            message_input = gr.Textbox(
                                label="Your message",
                                placeholder="Type your message here...",
                                lines=2
                            )
                            show_context_cb = gr.Checkbox(
                                label="Show RAG Context",
                                value=True
                            )
                            send_btn = gr.Button("Send", variant="primary")
                        
                        with gr.Column(scale=1):
                            context_output = gr.Textbox(
                                label="Context Info",
                                lines=20,
                                interactive=False
                            )
                    
                    def send_message(msg, history, show_ctx):
                        response, context = self.chat(msg, show_context=show_ctx)
                        messages = history or []
                        # Normalize to messages format expected by newer Gradio versions
                        if messages and isinstance(messages[0], (list, tuple)):
                            converted = []
                            for pair in messages:
                                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                                    converted.append({"role": "user", "content": pair[0]})
                                    converted.append({"role": "assistant", "content": pair[1]})
                            messages = converted
                        messages.append({"role": "user", "content": msg})
                        messages.append({"role": "assistant", "content": response})
                        return messages, context, ""
                    
                    send_btn.click(
                        send_message,
                        inputs=[message_input, chatbot, show_context_cb],
                        outputs=[chatbot, context_output, message_input]
                    )
                    
                    message_input.submit(
                        send_message,
                        inputs=[message_input, chatbot, show_context_cb],
                        outputs=[chatbot, context_output, message_input]
                    )
                
                # Configuration Tab
                with gr.Tab("⚙️ Configuration"):
                    gr.Markdown("### Agent Configuration (Active Agent)")
                    
                    with gr.Row():
                        agent_name = gr.Textbox(
                            label="Agent Name",
                            value=self.agent.config.agent.name
                        )
                        agent_age = gr.Number(
                            label="Age",
                            value=self.agent.config.agent.age
                        )
                        agent_gender = gr.Dropdown(
                            label="Gender",
                            choices=["male", "female", "other"],
                            value=self.agent.config.agent.gender
                        )
                    
                    with gr.Row():
                        agent_language = gr.Dropdown(
                            label="Language",
                            choices=["vi", "en", "ja", "zh"],
                            value=self.agent.config.agent.language
                        )
                        agent_personality = gr.Dropdown(
                            label="Personality",
                            choices=["friendly", "professional", "casual", "humorous"],
                            value=self.agent.config.agent.personality,
                            allow_custom_value=True
                        )
                        agent_style = gr.Dropdown(
                            label="Speaking Style",
                            choices=["natural", "formal", "casual", "poetic"],
                            value=self.agent.config.agent.speaking_style,
                            allow_custom_value=True
                        )
                    
                    agent_role = gr.Textbox(
                        label="Role",
                        value=self.agent.config.agent.role
                    )
                    
                    config_btn = gr.Button("Update Configuration", variant="primary")
                    config_output = gr.Textbox(label="Result", interactive=False)
                    
                    config_btn.click(
                        self.update_agent_config,
                        inputs=[
                            agent_name, agent_age, agent_gender,
                            agent_language, agent_personality, agent_style, agent_role
                        ],
                        outputs=config_output
                    )
                    
                    gr.Markdown("### View Current Configuration")
                    view_config_btn = gr.Button("View Configuration")
                    config_display = gr.Textbox(
                        label="Current Configuration",
                        interactive=False,
                        lines=15
                    )
                    
                    view_config_btn.click(
                        self.get_agent_config,
                        outputs=config_display
                    )
                
                # Knowledge Management Tab
                with gr.Tab("📚 Knowledge Management"):
                    gr.Markdown("### Knowledge Base Statistics")

                    agent_choices = [a.agent_id for a in self.agent_manager.list_agents()]
                    knowledge_target_agent = gr.Dropdown(
                        label="Target agent_id",
                        choices=agent_choices,
                        value=self.active_agent_id,
                        interactive=True
                    )

                    stats_btn = gr.Button("Refresh Statistics (target)")
                    stats_output = gr.Textbox(label="Statistics", interactive=False, lines=6)

                    def _stats_for_agent(agent_id: str):
                        try:
                            ag = self.agent_manager.instantiate(agent_id)
                            st = ag.get_knowledge_stats()
                            return f"📊 Knowledge Base Statistics (Agent: {agent_id}):\n- Total entries: {st['total_entries']}\n- Active entries: {st['active_entries']}\n- Index size: {st['index_size']}\n- Database path: {st['db_path']}"
                        except Exception as e:
                            return f"❌ Error: {e}"

                    stats_btn.click(_stats_for_agent, inputs=[knowledge_target_agent], outputs=[stats_output])

                    gr.Markdown("### Upload Document (.docx / .pdf / .txt)")
                    knowledge_file = gr.File(label="Upload file", file_types=[".docx", ".pdf", ".txt"], file_count="single")
                    knowledge_file_source = gr.Textbox(label="Source", value="upload")
                    add_file_btn = gr.Button("Add File to Knowledge", variant="primary")
                    add_file_result = gr.Textbox(label="Result", interactive=False)

                    add_file_btn.click(
                        self.add_file_knowledge_to_agent,
                        inputs=[knowledge_target_agent, knowledge_file, knowledge_file_source],
                        outputs=[add_file_result]
                    )

                    gr.Markdown("### Add Manual Knowledge (Target Agent)")
                    with gr.Row():
                        knowledge_input = gr.Textbox(
                            label="Knowledge Content",
                            placeholder="Enter knowledge to store...",
                            lines=3
                        )
                        knowledge_source = gr.Textbox(
                            label="Source",
                            value="manual",
                            lines=1
                        )
                    
                    add_knowledge_btn = gr.Button("Add Knowledge", variant="primary")
                    knowledge_output = gr.Textbox(label="Result", interactive=False)
                    
                    add_knowledge_btn.click(
                        self.add_manual_knowledge_to_agent,
                        inputs=[knowledge_target_agent, knowledge_input, knowledge_source],
                        outputs=[knowledge_output]
                    )
                    
                    gr.Markdown("### Danger Zone (Active Agent)")
                    clear_knowledge_btn = gr.Button("Clear All Knowledge (active)", variant="stop")
                    clear_knowledge_output = gr.Textbox(label="Result", interactive=False)
                    clear_knowledge_btn.click(self.clear_knowledge, outputs=clear_knowledge_output)
                
                # Tools Tab
                with gr.Tab("🛠️ Tools"):
                    gr.Markdown("### Available Tools (Active Agent)")
                    
                    tool_list = gr.Textbox(
                        label="Tools",
                        value="\n".join(self.tool_manager.list_tools()),
                        interactive=False,
                        lines=5
                    )
                    
                    gr.Markdown("### Execute Tool")
                    with gr.Row():
                        tool_name = gr.Dropdown(
                            label="Tool",
                            choices=self.tool_manager.list_tools()
                        )
                        tool_input = gr.Textbox(
                            label="Input (JSON)",
                            placeholder='{"query": "..."}',
                            lines=3
                        )
                    
                    execute_btn = gr.Button("Execute", variant="primary")
                    tool_output = gr.Textbox(
                        label="Output",
                        interactive=False,
                        lines=10
                    )
                    
                    def execute_tool(tool, inp):
                        try:
                            import json
                            kwargs = json.loads(inp) if inp else {}
                            result = self.tool_manager.execute_tool(tool, **kwargs)
                            return json.dumps(result, ensure_ascii=False, indent=2)
                        except Exception as e:
                            return f"Error: {str(e)}"
                    
                    execute_btn.click(
                        execute_tool,
                        inputs=[tool_name, tool_input],
                        outputs=tool_output
                    )
                
                # Dashboard Tab
                with gr.Tab("🧭 Dashboard"):
                    gr.Markdown("## Agents")
                    list_agents_btn = gr.Button("List Agents")
                    agents_json = gr.Textbox(label="Agents", lines=10)
                    list_agents_btn.click(self.list_agents_json, outputs=agents_json)
                    
                    gr.Markdown("### Create Agent")
                    with gr.Row():
                        new_agent_id = gr.Textbox(label="agent_id", value="agent_2")
                        new_agent_name = gr.Textbox(label="name", value="Nova")
                        new_agent_role = gr.Textbox(label="role", value="researcher")
                    with gr.Row():
                        new_agent_lang = gr.Dropdown(label="language", choices=["vi","en","ja","zh"], value="vi")
                        new_agent_personality = gr.Textbox(label="personality", value="friendly")
                        new_agent_style = gr.Textbox(label="speaking_style", value="natural")
                    create_agent_btn = gr.Button("Create Agent")
                    create_agent_out = gr.Textbox(label="Result")
                    create_agent_btn.click(
                        self.create_agent,
                        inputs=[new_agent_id, new_agent_name, new_agent_role, new_agent_lang, new_agent_personality, new_agent_style],
                        outputs=create_agent_out
                    )
                    
                    gr.Markdown("### Set Active Agent")
                    active_agent_inp = gr.Textbox(label="agent_id", value=self.active_agent_id)
                    set_active_btn = gr.Button("Set Active")
                    set_active_out = gr.Textbox(label="Result")
                    set_active_btn.click(self.set_active_agent, inputs=active_agent_inp, outputs=set_active_out)
                    
                    gr.Markdown("### Delete Agent")
                    del_agent_inp = gr.Textbox(label="agent_id")
                    del_agent_btn = gr.Button("Delete Agent")
                    del_agent_out = gr.Textbox(label="Result")
                    del_agent_btn.click(self.delete_agent, inputs=del_agent_inp, outputs=del_agent_out)
                    
                    gr.Markdown("## Groups")
                    list_groups_btn = gr.Button("List Groups")
                    groups_json = gr.Textbox(label="Groups", lines=10)
                    list_groups_btn.click(self.list_groups_json, outputs=groups_json)
                    
                    gr.Markdown("### Create Group")
                    with gr.Row():
                        new_group_id = gr.Textbox(label="group_id", value="grp_1")
                        new_group_name = gr.Textbox(label="name", value="Team Alpha")
                        new_group_members = gr.Textbox(label="members (comma-separated agent_ids)", value=self.active_agent_id)
                    with gr.Row():
                        roles_json = gr.Textbox(label="roles_json (agent_id -> role)", value="{}", placeholder='{"agent_default":"market_analyst","agent_2":"content_planner"}', lines=2)
                        executor_id = gr.Textbox(label="executor_id (agent who outputs final)", value="")
                    with gr.Row():
                        action_type = gr.Dropdown(label="action type", choices=["none","mongodb","postgresql"], value="none")
                        execute_action = gr.Checkbox(label="Execute action (not dry-run)", value=False)
                        target_json = gr.Textbox(label="target_json", value="{}", placeholder='{"collection":"content_plans"} or {"table":"content_plans"}', lines=2)
                    create_group_btn = gr.Button("Create Group")
                    create_group_out = gr.Textbox(label="Result")
                    create_group_btn.click(self.create_group, inputs=[new_group_id, new_group_name, new_group_members, roles_json, executor_id, action_type, execute_action, target_json], outputs=create_group_out)
                    
                    gr.Markdown("### Delete Group")
                    del_group_inp = gr.Textbox(label="group_id")
                    del_group_btn = gr.Button("Delete Group")
                    del_group_out = gr.Textbox(label="Result")
                    del_group_btn.click(self.delete_group, inputs=del_group_inp, outputs=del_group_out)
                    
                    gr.Markdown("## 📂 Kế hoạch đã lưu (MongoDB)")

                    with gr.Row():
                        refresh_saved_plans_btn = gr.Button("🔄 Tải danh sách kế hoạch")
                        saved_plans_dd = gr.Dropdown(label="Chọn kế hoạch (plan_id)", choices=[], interactive=True)
                        load_plan_btn = gr.Button("📥 Load kế hoạch", variant="primary")

                    load_plan_result = gr.Textbox(label="Trạng thái load", interactive=False, lines=2)
                    
                    gr.Markdown("## Multi-Agent Session")
                    with gr.Row():
                        run_group_id = gr.Textbox(label="group_id", value="grp_1")
                        run_rounds = gr.Number(label="rounds", value=3)
                        final_only_cb = gr.Checkbox(label="Final only (skip dialogue)", value=False)
                        auto_finalize_cb = gr.Checkbox(label="Auto finalize (dynamic rounds)", value=False)
                    run_prompt = gr.Textbox(label="task prompt", value="Hãy bàn luận và tóm tắt 3 lợi ích của RAG")
                    run_btn = gr.Button("Run Session")
                    transcript_out = gr.Textbox(label="Transcript", lines=18)
                    
                    # Plan display section - Table with inline edit buttons
                    gr.Markdown("### Kế hoạch Nội dung")

                    # Hidden state to store current plan
                    plan_state = gr.State(value=[])
                    full_plan_state = gr.State(value=None)
                    plan_id_state = gr.State(value=None)

                    with gr.Row():
                        save_plan_btn = gr.Button("💾 Lưu kế hoạch", variant="primary")
                        save_plan_result = gr.Textbox(label="Lưu kế hoạch", interactive=False, lines=2)

                    # Add click handler for save_plan_btn
                    save_plan_btn.click(
                        self._save_plan_to_db,
                        inputs=[full_plan_state],
                        outputs=[save_plan_result, plan_id_state]
                    )

                    
                    with gr.Row():
                        with gr.Column(scale=4):
                            plan_df = gr.Dataframe(
                                headers=["Idx", "Ngày", "Tiêu đề", "Tóm tắt"],
                                datatype=["number", "str", "str", "str"],
                                interactive=False,
                                wrap=True
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("**Hành động**")
                            edit_post_selector = gr.Radio(
                                label="Chọn bài để sửa",
                                choices=[],
                                interactive=True
                            )
                    
                    # Edit section (hidden initially)
                    with gr.Column(visible=False) as edit_section:
                        gr.Markdown("### ✏️ Chỉnh sửa bài post")

                        # -------- Article (detailed content) --------
                        gr.Markdown("### 📝 Bài viết chi tiết (theo kịch bản)")
                        article_state = gr.State(value=None)  # dict
                        article_post_id_state = gr.State(value=None)  # string post_id

                        with gr.Row():
                            generate_article_btn = gr.Button("📝 Tạo bài viết chi tiết", variant="primary")
                            load_article_btn = gr.Button("🔄 Tải bài viết đã lưu", variant="secondary")

                        article_title = gr.Textbox(label="📰 Tiêu đề bài viết", lines=2)
                        article_content = gr.Textbox(label="📄 Nội dung (Markdown)", lines=14)

                        with gr.Accordion("🖼️ Ảnh (URL)", open=False):
                            image_url_inp = gr.Textbox(label="Dán URL ảnh", placeholder="https://...")
                            add_image_btn = gr.Button("➕ Thêm ảnh URL")
                            images_json = gr.Textbox(label="Danh sách ảnh (JSON)", lines=3, interactive=False)

                        with gr.Accordion("🤖 Yêu cầu Agent chỉnh sửa bài viết", open=False):
                            article_revision_prompt = gr.Textbox(
                                label="Hướng dẫn",
                                lines=3,
                                value="Hãy cải thiện bài viết: rõ ràng hơn, hấp dẫn hơn, thêm cấu trúc heading, bullet points."
                            )
                            agent_revise_article_btn = gr.Button("🤖 Agent sửa bài viết", variant="secondary")

                        with gr.Row():
                            save_article_btn = gr.Button("💾 Lưu bài viết", variant="primary")
                            save_article_result = gr.Textbox(label="Kết quả bài viết", interactive=False, lines=2)

                    def _get_post_obj_by_idx(selected, posts_list):
                        """Return post object based on selected radio value.

                        Supported formats:
                        - "#0: title..." (from edit_post_selector)
                        - 0 / "0" (index)
                        """
                        if selected is None or selected == "" or not posts_list:
                            return None

                        idx_int = None
                        try:
                            if isinstance(selected, str) and selected.strip().startswith("#"):
                                idx_int = int(selected.split(":")[0].replace("#", "").strip())
                            else:
                                idx_int = int(selected)
                        except Exception:
                            return None

                        if idx_int < 0 or idx_int >= len(posts_list):
                            return None
                        p = posts_list[idx_int]
                        return p if isinstance(p, dict) else None

                    def _load_or_init_article(post_idx, full_plan, current_article):
                        """Build article object from selected post (no DB)."""
                        post = _get_post_obj_by_idx(post_idx, full_plan.get("posts", []) if isinstance(full_plan, dict) else [])
                        if not post:
                            return None, None, "", "", "[]"
                        post_id = post.get("post_id")
                        art = {
                            "post_id": post_id,
                            "title": post.get("title", ""),
                            "content": "",
                            "images": []
                        }
                        return art, post_id, art["title"], art["content"], json.dumps(art["images"], ensure_ascii=False)

                    def _generate_article(post_idx, full_plan, plan_id):
                        post = _get_post_obj_by_idx(post_idx, full_plan.get("posts", []) if isinstance(full_plan, dict) else [])
                        if not post:
                            return None, None, "", "", "[]", "❌ Chưa chọn kịch bản"
                        if not plan_id:
                            return None, None, "", "", "[]", "❌ Chưa có plan_id (hãy Lưu kế hoạch trước)"
                        post_id = post.get("post_id")

                        prompt = f"""Bạn là content writer. Hãy viết 1 bài viết chi tiết bằng tiếng Việt dựa trên kịch bản sau.

Kịch bản (JSON):
{json.dumps(post, ensure_ascii=False, indent=2)}

Yêu cầu:
- Viết dạng Markdown
- Có tiêu đề H1, ít nhất 3 mục H2
- Có phần mở bài (hook), nội dung chi tiết, kết luận + CTA
- Giữ đúng chủ đề/title của kịch bản

Chỉ trả về JSON hợp lệ theo schema:
{{"title": "...", "content": "..."}}
"""
                        res = self.agent.process_message(prompt)
                        if res.get("error"):
                            return None, None, "", "", "[]", f"❌ Lỗi model: {res.get('error')}"
                        txt = res.get("response", "")
                        try:
                            import re
                            m = re.search(r"\{[\s\S]*\}", txt)
                            obj = json.loads(m.group(0) if m else txt)
                            title = obj.get("title", post.get("title", ""))
                            content = obj.get("content", "")
                        except Exception:
                            title = post.get("title", "")
                            content = txt
                        art = {
                            "plan_id": plan_id,
                            "post_id": post_id,
                            "title": title,
                            "content": content,
                            "images": []
                        }
                        return art, post_id, title, content, "[]", "✅ Đã tạo bài viết (chưa lưu)"

                    def _save_article(plan_id, post_id, title, content, images_json_str):
                        if not plan_id:
                            return "❌ Chưa có plan_id (hãy Lưu kế hoạch trước)"
                        if not post_id:
                            return "❌ Chưa chọn kịch bản"
                        if self.articles_col is None:
                            return "❌ MongoDB articles chưa sẵn sàng"
                        try:
                            imgs = []
                            if images_json_str and images_json_str.strip():
                                imgs = json.loads(images_json_str)
                                if not isinstance(imgs, list):
                                    imgs = []
                        except Exception:
                            imgs = []
                        now = datetime.utcnow().isoformat()
                        doc = {
                            "plan_id": plan_id,
                            "post_id": post_id,
                            "title": title,
                            "content": content,
                            "images": imgs,
                            "updated_at": now
                        }
                        try:
                            # upsert by (plan_id, post_id)
                            self.articles_col.update_one(
                                {"plan_id": plan_id, "post_id": post_id},
                                {"$set": doc, "$setOnInsert": {"created_at": now}},
                                upsert=True
                            )
                            return "✅ Đã lưu bài viết vào MongoDB (collection=articles)"
                        except Exception as e:
                            return f"❌ Lỗi lưu bài viết: {e}"

                    def _load_article_from_db(plan_id, post_id):
                        if not plan_id or not post_id:
                            return None, post_id, "", "", "[]", "❌ Thiếu plan_id/post_id"
                        if self.articles_col is None:
                            return None, post_id, "", "", "[]", "❌ MongoDB articles chưa sẵn sàng"
                        doc = self.articles_col.find_one({"plan_id": plan_id, "post_id": post_id})
                        if not doc:
                            return None, post_id, "", "", "[]", "⚠️ Chưa có bài viết đã lưu cho kịch bản này"
                        title = doc.get("title", "")
                        content = doc.get("content", "")
                        imgs = doc.get("images", []) or []
                        return doc, post_id, title, content, json.dumps(imgs, ensure_ascii=False), "✅ Đã tải bài viết"

                    def _add_image_url(images_json_str, url):
                        url = (url or "").strip()
                        if not url:
                            return images_json_str
                        try:
                            imgs = json.loads(images_json_str) if images_json_str else []
                            if not isinstance(imgs, list):
                                imgs = []
                        except Exception:
                            imgs = []
                        imgs.append(url)
                        return json.dumps(imgs, ensure_ascii=False)

                    def _agent_revise_article(plan_id, post_id, title, content, images_json_str, user_prompt):
                        if not plan_id or not post_id:
                            return title, content, images_json_str, "❌ Thiếu plan_id/post_id"
                        prompt = f"""Bài viết hiện tại (JSON):
{{"title": {json.dumps(title, ensure_ascii=False)}, "content": {json.dumps(content, ensure_ascii=False)}}}

Yêu cầu sửa đổi:
{user_prompt}

Hãy trả về JSON hợp lệ theo schema:
{{"title": "...", "content": "..."}}
"""
                        res = self.agent.process_message(prompt)
                        if res.get("error"):
                            return title, content, images_json_str, f"❌ Lỗi model: {res.get('error')}"
                        txt = res.get("response", "")
                        try:
                            import re
                            m = re.search(r"\{[\s\S]*\}", txt)
                            obj = json.loads(m.group(0) if m else txt)
                            new_title = obj.get("title", title)
                            new_content = obj.get("content", content)
                            return new_title, new_content, images_json_str, "✅ Agent đã sửa bài viết (chưa lưu)"
                        except Exception:
                            return title, txt, images_json_str, "⚠️ Agent trả về không-JSON; đã thay content bằng raw"

                    # Wiring
                    generate_article_btn.click(
                        _generate_article,
                        inputs=[edit_post_selector, full_plan_state, plan_id_state],
                        outputs=[article_state, article_post_id_state, article_title, article_content, images_json, save_article_result]
                    )

                    load_article_btn.click(
                        _load_article_from_db,
                        inputs=[plan_id_state, article_post_id_state],
                        outputs=[article_state, article_post_id_state, article_title, article_content, images_json, save_article_result]
                    )

                    add_image_btn.click(
                        _add_image_url,
                        inputs=[images_json, image_url_inp],
                        outputs=[images_json]
                    )

                    agent_revise_article_btn.click(
                        _agent_revise_article,
                        inputs=[plan_id_state, article_post_id_state, article_title, article_content, images_json, article_revision_prompt],
                        outputs=[article_title, article_content, images_json, save_article_result]
                    )

                    save_article_btn.click(
                        _save_article,
                        inputs=[plan_id_state, article_post_id_state, article_title, article_content, images_json],
                        outputs=[save_article_result]
                    )

                    with gr.Row():
                        edit_date = gr.Textbox(label="📅 Ngày đăng", placeholder="YYYY-MM-DD", scale=1)
                        gr.HTML("<div style='width: 20px;'></div>")
                        
                    edit_title = gr.Textbox(label="📝 Tiêu đề", lines=2)
                    edit_summary = gr.Textbox(label="📄 Tóm tắt", lines=4)
                        
                    # Agent revision prompt section
                    with gr.Accordion("Yêu cầu Agent sửa đổi", open=False):
                        agent_revision_prompt = gr.Textbox(
                            label="Hướng dẫn cho Agent",
                            placeholder="Ví dụ: Làm cho tiêu đề hấp dẫn hơn, thêm emoji, tập trung vào lợi ích cho người dùng...",
                            lines=3,
                            value="Vui lòng cải thiện bài post này để:\n1. Tiêu đề hấp dẫn và thu hút hơn\n2. Tóm tắt rõ ràng, súc tích\n3. Phù hợp với đối tượng mục tiêu trên Farcaster"
                        )
                        request_agent_edit_btn = gr.Button("🤖 Gửi yêu cầu cho Agent", variant="secondary")

                    with gr.Row():
                        save_edit_btn = gr.Button("💾 Lưu thay đổi", variant="primary")
                        cancel_edit_btn = gr.Button("❌ Hủy")

                    edit_result = gr.Textbox(label="Kết quả", interactive=False)

                    def _update_post_selector(posts_data):
                        """Update radio buttons with post options"""
                        if not posts_data or len(posts_data) == 0:
                            return gr.update(choices=[], value=None)
                        choices = [f"#{row[0]}: {row[2][:50]}..." if len(row[2]) > 50 else f"#{row[0]}: {row[2]}" for row in posts_data]
                        return gr.update(choices=choices, value=None)
                    
                    def _load_post_for_edit(selected_post, posts_data):
                        """Load selected post data into edit form"""
                        if not selected_post or not posts_data:
                            return gr.update(visible=False), "", "", "", ""
                        
                        # Extract post ID from selection
                        post_id = int(selected_post.split(":")[0].replace("#", ""))
                        
                        # Find the post
                        for row in posts_data:
                            # NOTE: Dataframe/State can coerce numbers to float; normalize for comparison
                            try:
                                row_id = int(row[0])
                            except Exception:
                                row_id = row[0]
                            if row_id == post_id:
                                return (
                                    gr.update(visible=True),  # edit_section
                                    row[1],  # edit_date
                                    row[2],  # edit_title
                                    row[3],  # edit_summary
                                    ""  # edit_result
                                )
                        
                        return gr.update(visible=False), "", "", "", ""
                    
                    def _save_manual_edit(selected_post, date, title, summary, posts_data):
                        """Save manual edits to the post"""
                        print(f"[DEBUG] _save_manual_edit called with selected_post={selected_post}, type={type(selected_post)}")
                        if not selected_post or not posts_data:
                            return "❌ Không có bài post nào được chọn", posts_data, posts_data
                        
                        # Update the post in the list
                        for i, row in enumerate(posts_data):
                            try:
                                row_id = int(row[0])
                            except Exception:
                                row_id = row[0]
                            post_id = int(str(selected_post).split(":")[0].replace("#", ""))
                            if row_id == post_id:
                                posts_data[i] = [post_id, date, title, summary]
                                # Keep internal plan state in sync (if available)
                                if i < len(self.current_plan_state) and isinstance(self.current_plan_state[i], dict):
                                    self.current_plan_state[i]["date"] = date
                                    self.current_plan_state[i]["title"] = title
                                    self.current_plan_state[i]["summary"] = summary
                                break
                        
                        return "✅ Đã lưu thay đổi", posts_data, posts_data
                    
                    def _request_agent_revision(selected_post, date, title, summary, user_prompt, posts_data):
                        """Request agent to revise the post with user's custom instructions"""
                        print(f"[DEBUG] _request_agent_revision called with selected_post={selected_post}, type={type(selected_post)}")
                        if not selected_post:
                            return "❌ Không có bài post nào được chọn", posts_data, posts_data
                        
                        # Create revision prompt with user's instructions
                        post_json = json.dumps({
                            "date": date,
                            "title": title,
                            "summary": summary
                        }, ensure_ascii=False, indent=2)
                        
                        revision_prompt = f"""Bài post hiện tại:

{post_json}

Yêu cầu từ người dùng:
{user_prompt}

Hãy sửa đổi bài post theo yêu cầu trên. Chỉ trả về JSON với các trường: date, title, summary"""
                        
                        # Call agent
                        result = self.agent.process_message(revision_prompt)
                        if result.get("error"):
                            return f"❌ Lỗi: {result['error']}", posts_data, posts_data
                        
                        response = result.get("response", "")
                        
                        # Try to parse JSON from response
                        try:
                            import re
                            json_match = re.search(r'\{[\s\S]*\}', response)
                            if json_match:
                                revised = json.loads(json_match.group(0))
                                # Update the post
                                post_id = int(str(selected_post).split(":")[0].replace("#", ""))
                                for i, row in enumerate(posts_data):
                                    try:
                                        row_id = int(row[0])
                                    except Exception:
                                        row_id = row[0]
                                    if row_id == post_id:
                                        posts_data[i] = [
                                            int(post_id),
                                            revised.get("date", date),
                                            revised.get("title", title),
                                            revised.get("summary", summary)
                                        ]
                                        if i < len(self.current_plan_state) and isinstance(self.current_plan_state[i], dict):
                                            self.current_plan_state[i].update(revised)
                                        break
                                return "✅ Agent đã sửa đổi bài post", posts_data, posts_data
                        except Exception as e:
                            return f"⚠️ Agent phản hồi:\n{response}", posts_data, posts_data
                        
                        return f"⚠️ Không thể phân tích phản hồi:\n{response}", posts_data, posts_data
                    
                    def _cancel_edit():
                        """Cancel editing"""
                        return gr.update(visible=False), "", "", "", ""

                    # Wire up event handlers

                    # Saved plans handlers (must be wired AFTER plan_df/edit_post_selector are created)
                    refresh_saved_plans_btn.click(
                        self._list_saved_plans,
                        outputs=[saved_plans_dd, load_plan_result]
                    )

                    load_plan_btn.click(
                        self._load_plan_from_db,
                        inputs=[saved_plans_dd],
                        outputs=[load_plan_result, plan_df, plan_state, full_plan_state, plan_id_state, edit_post_selector]
                    )

                    edit_post_selector.change(
                        _load_post_for_edit,
                        inputs=[edit_post_selector, plan_state],
                        outputs=[edit_section, edit_date, edit_title, edit_summary, edit_result]
                    )
                    
                    save_edit_btn.click(
                        _save_manual_edit,
                        inputs=[edit_post_selector, edit_date, edit_title, edit_summary, plan_state],
                        outputs=[edit_result, plan_state, plan_df]
                    )
                    
                    request_agent_edit_btn.click(
                        _request_agent_revision,
                        inputs=[edit_post_selector, edit_date, edit_title, edit_summary, agent_revision_prompt, plan_state],
                        outputs=[edit_result, plan_state, plan_df]
                    )
                    
                    cancel_edit_btn.click(
                        _cancel_edit,
                        outputs=[edit_section, edit_date, edit_title, edit_summary, edit_result]
                    )
                    
                    def _update_plan_display(transcript, posts_data, full_plan):
                        """Update plan display after session completes"""
                        selector = _update_post_selector(posts_data)
                        # Update full_plan_state with the complete plan data
                        # Ensure full plan carries the latest edited posts (from self.current_plan_state)
                        if isinstance(full_plan, dict) and isinstance(getattr(self, "current_plan_state", None), list):
                            full_plan["posts"] = self.current_plan_state
                        return transcript, posts_data, posts_data, selector, full_plan
                    
                    run_btn.click(
                        self.run_multi_agent, 
                        inputs=[run_group_id, run_prompt, run_rounds, final_only_cb, auto_finalize_cb], 
                        outputs=[transcript_out, plan_state, full_plan_state]
                    ).then(
                        _update_plan_display,
                        inputs=[transcript_out, plan_state, full_plan_state],
                        outputs=[transcript_out, plan_df, plan_state, edit_post_selector, full_plan_state]
                    )
                    
                    gr.Markdown("## Shared Knowledge")
                    shared_stats_btn = gr.Button("Refresh Shared Stats")
                    shared_stats_out = gr.Textbox(label="Shared Stats", lines=6)
                    shared_stats_btn.click(self.get_shared_stats, outputs=shared_stats_out)
                    
                    shared_add_input = gr.Textbox(label="Add Shared Knowledge", lines=3)
                    shared_add_btn = gr.Button("Add to Shared KB")
                    shared_add_out = gr.Textbox(label="Result")
                    shared_add_btn.click(self.add_shared_knowledge, inputs=shared_add_input, outputs=shared_add_out)
                
                # About Tab
                with gr.Tab("ℹ️ About"):
                    gr.Markdown("""
                    ## AI Agent with Long-term Memory
                    
                    ### Features
                    - ✅ OpenAI Integration (GPT-4)
                    - ✅ Long-term Memory (FAISS Vector DB)
                    - ✅ RAG Pipeline (Extract → Judge → Embed → Store → Retrieve)
                    - ✅ Automatic Knowledge Extraction
                    - ✅ Duplicate Detection & Noise Filtering
                    - ✅ Configurable Personality & Roles
                    - ✅ Tool Integration
                    - ✅ Multi-Agent Dashboard (Agents, Groups, Shared KB)
                    """)
        
        return demo


def launch_ui(config_path: str = "config.yaml"):
    print(f"[UI] Loading AgentUI from: {__file__}")
    print("[UI] Mode: Multi-Agent Dashboard enabled")
    ui = AgentUI(config_path)
    demo = ui.create_interface()
    # Use local UI by default; set GRADIO_SHARE=1 to enable sharing
    import os
    share = os.getenv("GRADIO_SHARE", "0") == "1"
    demo.launch(share=share)


if __name__ == "__main__":
    launch_ui()
