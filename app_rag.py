from agentic_chatbot_rag_backend import (
    chatbot,
    get_all_threads,
    ingest_rag_document,
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

import streamlit as st
import uuid
import tempfile
import os


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# Helper functions
# ============================================================
def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["raw_messages"] = []
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id: str) -> list:
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        return state.values.get("messages", [])
    except Exception as e:
        st.error(f"Could not load conversation: {e}")
        return []


def get_thread_title(thread_id: str) -> str:
    messages = load_conversation(thread_id)
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            title = str(msg.content).strip().replace("\n", " ")
            return (title[:28] + "…") if len(title) > 28 else title
    return f"Chat {thread_id[:6]}"


# ============================================================
# Session state initialization
# ============================================================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "raw_messages" not in st.session_state:
    st.session_state["raw_messages"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    try:
        st.session_state["chat_threads"] = list(get_all_threads())
    except Exception:
        st.session_state["chat_threads"] = []

add_thread(st.session_state["thread_id"])


# ============================================================
# Sidebar — conversations only (PDF uploader removed)
# ============================================================
st.sidebar.title("💬 My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

for thread_id in reversed(st.session_state["chat_threads"]):
    is_active = thread_id == st.session_state["thread_id"]
    label = ("🟢 " if is_active else "") + get_thread_title(thread_id)

    col1, col2 = st.sidebar.columns([0.82, 0.18])

    with col1:
        if st.button(
            label,
            key=f"thread_{thread_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []
            for message in messages:
                if isinstance(message, HumanMessage):
                    temp_messages.append(
                        {"role": "user", "content": message.content}
                    )
                elif isinstance(message, AIMessage):
                    temp_messages.append(
                        {"role": "assistant", "content": message.content}
                    )
            st.session_state["message_history"] = temp_messages
            st.session_state["raw_messages"] = messages
            st.rerun()

    with col2:
        if st.button("🗑", key=f"del_{thread_id}", help="Delete conversation"):
            st.session_state["chat_threads"].remove(thread_id)
            if thread_id == st.session_state["thread_id"]:
                reset_chat()
            st.rerun()


# ============================================================
# Main chat area
# ============================================================
st.title("🤖 Agentic Chatbot with LangGraph")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# Chat input (with optional PDF attach)
# ============================================================
submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"],
)

user_input = None
pdf_uploaded = False

if submission:
    user_input = (submission.text or "").strip()
    uploaded_files = submission.files

    # ---------- Process attached PDF ----------
    if uploaded_files:
        uploaded_pdf = uploaded_files[0]
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(uploaded_pdf.getvalue())
                temp_path = tmp.name

            with st.spinner(f"Processing {uploaded_pdf.name}…"):
                ingest_rag_document(temp_path)

            pdf_uploaded = True
            st.toast(
                f"{uploaded_pdf.name} processed successfully.",
                icon="✅",
            )

            # If no text was provided, auto-generate a summary request
            if not user_input:
                user_input = (
                    f"I just uploaded a PDF named '{uploaded_pdf.name}'. "
                    "Please use the rag_tool to retrieve its contents and "
                    "provide a brief, well-structured summary of the document. "
                    "Include:\n"
                    "1. The main topic/purpose of the document.\n"
                    "2. Key sections or themes covered.\n"
                    "3. Any important findings, conclusions, or recommendations.\n"
                    "Keep it concise (around 150-250 words) and use bullet points where helpful."
                )

        except Exception as e:
            st.error(f"PDF processing failed: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================
# Send message + stream response
# ============================================================
if user_input:
    # --- Save + render user message ---
    display_text = user_input
    if pdf_uploaded and not (submission and submission.text):
        display_text = "📄 *(PDF uploaded — generating summary…)*"

    st.session_state["message_history"].append(
        {"role": "user", "content": display_text}
    )
    with st.chat_message("user"):
        st.markdown(display_text)

    # --- Config for LangGraph ---
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_trace",
    }

    # --- Stream assistant response ---
    with st.chat_message("assistant"):
        status_holder = {"box": None}
        tool_history = []

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    tool_history.append(
                        {
                            "name": tool_name,
                            "content": message_chunk.content,
                        }
                    )

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True,
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

        if tool_history:
            with st.expander("🔍 Tool call details", expanded=False):
                for i, tool in enumerate(tool_history, 1):
                    st.markdown(f"**{i}. `{tool['name']}`**")
                    st.code(str(tool["content"])[:2000], language="text")

    # ---- Persist assistant message ----
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )