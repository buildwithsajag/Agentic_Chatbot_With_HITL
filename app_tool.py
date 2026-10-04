from agentic_chatbot_tool_backend import chatbot, get_all_threads

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage
)
import streamlit as st
import uuid


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
    """Add a thread id to session list without duplicates."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    """Start a brand-new conversation."""
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["raw_messages"] = []   # full LangChain message trace
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id: str) -> list:
    """Fetch messages from the LangGraph checkpointer."""
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        return state.values.get("messages", [])
    except Exception as e:
        st.error(f"Could not load conversation: {e}")
        return []


def get_thread_title(thread_id: str) -> str:
    """Return a friendly title for a thread (first user message)."""
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
# Sidebar — conversation list
# ============================================================
st.sidebar.title("💬 My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# Show newest first
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
            # Switch conversation
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)

            # Convert LangChain → Streamlit format
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

            # If we deleted the current one, start a new chat
            if thread_id == st.session_state["thread_id"]:
                reset_chat()
            st.rerun()


# ============================================================
# Main chat area
# ============================================================
st.title("🤖 Agentic Chatbot with LangGraph")

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# Chat input + streaming response
# ============================================================
user_input = st.chat_input("Type your message…")

if user_input:
    # --- Show user message ---
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Config for LangGraph ---
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_trace",
    }

    # --- Stream assistant ---
    with st.chat_message("assistant"):
        status_holder = {"box": None}
        tool_history = []   # track all tool calls made in this turn

        def ai_only_stream():
            """Yield only AIMessage text chunks, and record tool events."""
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # -- Tool messages: capture & show status --
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
                            f"🔧 Using `{tool_name}`…",
                            expanded=True,
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                # -- Stream assistant text only --
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize the tool status box
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

        # Optional: show tool details in an expander
        if tool_history:
            with st.expander("🔍 Tool call details", expanded=False):
                for i, tool in enumerate(tool_history, 1):
                    st.markdown(f"**{i}. `{tool['name']}`**")
                    st.code(str(tool["content"])[:2000], language="text")

    # --- Persist to session state ---
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )