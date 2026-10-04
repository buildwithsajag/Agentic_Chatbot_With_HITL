from agentic_chatbot_hitl_backend import (
    chatbot,
    get_all_threads,
    ingest_rag_document
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os
import html


# ========================= Page configuration =========================
# Must be the first Streamlit command executed in the script.

st.set_page_config(
    page_title="AI Assistant",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ========================= Theme / CSS injection =========================
# One <style> block, injected once per run. It does four jobs:
#   1. Defines design tokens (colors, radii, shadows) as CSS variables so the
#      whole theme can be retuned from a single place.
#   2. Strips Streamlit's default chrome (menu, footer, deploy button).
#   3. Restyles the chat surface: messages, input bar, send button.
#   4. Restyles the sidebar and native widgets so nothing looks "default".
#
# Notes on robustness:
#   - Selectors use Streamlit's stable data-testid attributes, not the
#     auto-generated emotion class names (.st-emotion-cache-xxxx), which
#     change between releases.
#   - The font is applied to text elements only, never with `*`, because
#     Streamlit renders its icons with an icon font. Overriding it would
#     turn icons into raw words like "keyboard_arrow_down".

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;650&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---------- 1. Design tokens ---------- */
:root {
    --bg:            #0f172a;   /* app background (deep slate)        */
    --bg-sidebar:    #0b1222;   /* sidebar sits one step darker       */
    --surface:       #1e293b;   /* cards, inputs, user messages       */
    --surface-soft:  #151f33;   /* assistant messages, code blocks    */
    --border:        #2a3850;   /* hairline borders                   */
    --border-strong: #3b4a63;   /* hover / emphasised borders         */
    --text:          #e2e8f0;
    --text-strong:   #f8fafc;
    --muted:         #94a3b8;   /* secondary copy                     */
    --accent:        #6366f1;   /* indigo                             */
    --accent-hover:  #5558e6;
    --accent-soft:   rgba(99, 102, 241, 0.14);
    --accent-ring:   rgba(99, 102, 241, 0.28);
    --warn:          #f59e0b;
    --radius-sm:     10px;
    --radius-md:     14px;
    --radius-lg:     16px;
    --shadow-sm:     0 1px 2px rgba(2, 6, 23, 0.35);
    --shadow-md:     0 8px 24px -8px rgba(2, 6, 23, 0.55);
    --font-ui:       'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono:     'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace;
}

/* ---------- 2. Base typography + app shell ---------- */
.stApp,
[data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--text);
}

.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, label, input, textarea,
button, select, h1, h2, h3, h4, h5, h6,
[data-testid="stSidebar"] p, [data-testid="stCaptionContainer"] {
    font-family: var(--font-ui) !important;
    font-feature-settings: 'cv11', 'ss01';
    -webkit-font-smoothing: antialiased;
}

/* Put the icon font back wherever Streamlit uses ligature icons. */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-icons {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

.stMarkdown p, .stMarkdown li {
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text);
}
h1, h2, h3, h4 {
    color: var(--text-strong);
    letter-spacing: -0.02em;
    font-weight: 600;
}

/* Centered column: ~760px keeps lines readable (< 80 characters). */
.block-container {
    max-width: 760px;
    padding-top: 2.5rem;
    padding-bottom: 7rem;
}

/* ---------- 3. Hide default Streamlit chrome ---------- */
#MainMenu,
footer,
[data-testid="stDecoration"],
[data-testid="stMainMenu"],
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
}
/* Keep the header element (it holds the sidebar toggle) but make it invisible. */
[data-testid="stHeader"] {
    background: transparent;
    height: 2.75rem;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] button {
    color: var(--muted);
}

/* Thin, quiet scrollbars */
* { scrollbar-width: thin; scrollbar-color: var(--border) transparent; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }

/* ---------- 4. Chat messages ---------- */
[data-testid="stChatMessage"] {
    background: var(--surface-soft);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
    gap: 0.9rem;
    box-shadow: var(--shadow-sm);
}
/* User messages: lighter surface and a stronger shadow, so they read as "raised". */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="chatAvatarIcon-user"]
) {
    background: var(--surface);
    border-color: var(--border-strong);
    box-shadow: var(--shadow-md);
}
/* Avatars: small rounded squares instead of the default colored circles. */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    width: 1.9rem;
    height: 1.9rem;
    border-radius: 9px;
    color: var(--text-strong);
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="chatAvatarIcon-user"] {
    background: #475569;
}
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--accent);
}
[data-testid="stChatMessageContent"] { min-width: 0; }

/* Code */
.stMarkdown code {
    font-family: var(--font-mono) !important;
    font-size: 0.85em;
    background: var(--bg-sidebar);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.1em 0.4em;
    color: #c7d2fe;
}
.stMarkdown pre, [data-testid="stCode"] pre {
    background: var(--bg-sidebar) !important;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
}
.stMarkdown pre code { border: 0; padding: 0; background: transparent; color: var(--text); }

/* ---------- 5. Chat input bar ---------- */
/* Bottom dock: same background as the page, same width as the content column. */
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background: var(--bg);
}
[data-testid="stBottomBlockContainer"] {
    max-width: 760px;
    padding-bottom: 1.5rem;
}

[data-testid="stChatInput"] {
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
/* Streamlit nests extra wrappers inside; neutralise their own borders. */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] div[data-baseweb="base-input"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
/* Focus state: indigo border + soft ring. */
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-ring), var(--shadow-md);
}
[data-testid="stChatInput"] textarea {
    color: var(--text-strong);
    font-size: 0.95rem;
    caret-color: var(--accent);
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted); opacity: 1; }
/* Attachment button (quiet) */
[data-testid="stChatInput"] button {
    background: transparent;
    color: var(--muted);
    border: 0;
    border-radius: var(--radius-sm);
}
[data-testid="stChatInput"] button:hover { background: var(--accent-soft); color: var(--text-strong); }
/* Send button (accent) */
[data-testid="stChatInputSubmitButton"] {
    background: var(--accent) !important;
    color: #fff !important;
    border-radius: var(--radius-sm);
    transition: background 0.15s ease, transform 0.1s ease;
}
[data-testid="stChatInputSubmitButton"]:hover:not(:disabled) { background: var(--accent-hover) !important; }
[data-testid="stChatInputSubmitButton"]:active:not(:disabled) { transform: scale(0.95); }
[data-testid="stChatInputSubmitButton"]:disabled {
    background: var(--border) !important;
    color: var(--muted) !important;
}

/* ---------- 6. Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
[data-testid="stSidebar"] hr {
    border: 0;
    border-top: 1px solid var(--border);
    margin: 1rem 0;
}
.brand {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.25rem 0.1rem 0.5rem;
}
.brand-mark {
    width: 1.9rem;
    height: 1.9rem;
    display: grid;
    place-items: center;
    background: var(--accent);
    border-radius: 9px;
    font-size: 1rem;
}
.brand-name {
    font-weight: 600;
    font-size: 0.98rem;
    letter-spacing: -0.01em;
    color: var(--text-strong);
}
.side-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--muted);
    margin: 0 0 0.4rem 0.15rem;
}

/* "New chat" button: outlined, the only boxed button in the sidebar. */
.st-key-new_chat button {
    background: var(--surface);
    border: 1px solid var(--border-strong);
    color: var(--text-strong);
    justify-content: center;
}
.st-key-new_chat button:hover {
    border-color: var(--accent);
    background: var(--accent-soft);
}

/* Conversation list: borderless rows; the active one gets a filled background. */
[class*="st-key-thread_"] button {
    background: transparent;
    border: 0;
    box-shadow: none;
    color: var(--muted);
    justify-content: flex-start;
    text-align: left;
    padding: 0.45rem 0.65rem;
    min-height: 0;
    font-weight: 450;
}
[class*="st-key-thread_"] button p {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 0.875rem;
}
[class*="st-key-thread_"] button:hover {
    background: var(--surface-soft);
    color: var(--text-strong);
}
/* Active conversation is rendered with type="primary" in Python. */
[class*="st-key-thread_"] button[kind="primary"],
[class*="st-key-thread_"] [data-testid="stBaseButton-primary"] {
    background: var(--surface);
    color: var(--text-strong);
    box-shadow: inset 2px 0 0 var(--accent);
}

/* ---------- 7. Native widgets ---------- */
/* Buttons (secondary) */
.stButton > button,
[data-testid="stBaseButton-secondary"] {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.5rem 1rem;
    box-shadow: var(--shadow-sm);
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--border-strong);
    background: #243247;
    color: var(--text-strong);
}
/* Buttons (primary) */
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: var(--accent-hover);
    border-color: var(--accent-hover);
    color: #fff;
}
/* Visible keyboard focus everywhere */
button:focus-visible, a:focus-visible, [role="button"]:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px;
}

/* Expanders + st.status (tool activity) */
[data-testid="stExpander"],
[data-testid="stStatus"] {
    background: var(--surface-soft);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    box-shadow: none;
    overflow: hidden;
}
[data-testid="stExpander"] details,
[data-testid="stStatus"] details { border: 0; background: transparent; }
[data-testid="stExpander"] summary,
[data-testid="stStatus"] summary { color: var(--muted); font-size: 0.9rem; }
[data-testid="stExpander"] summary:hover,
[data-testid="stStatus"] summary:hover { color: var(--text-strong); }

/* Select boxes, text inputs and their dropdown menus */
[data-baseweb="select"] > div,
[data-baseweb="input"],
[data-baseweb="textarea"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text);
}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"]:focus-within,
[data-baseweb="textarea"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-ring) !important;
}
[data-baseweb="popover"] > div,
[data-baseweb="menu"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-md);
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [aria-selected="true"] { background: var(--accent-soft) !important; }

/* Alerts and toasts */
[data-testid="stAlert"] {
    background: var(--surface-soft);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    color: var(--text);
}
[data-testid="stToast"] {
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-md);
}

/* ---------- 8. Custom components (rendered from Python) ---------- */
/* Empty state shown before the first message */
.empty-state { padding: 18vh 0 2rem; }
.empty-state h1 {
    font-size: 2rem;
    line-height: 1.2;
    margin: 0 0 0.6rem;
    padding: 0;
}
.empty-state p {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.6;
    max-width: 34rem;
    margin: 0;
}

/* Human-approval card (shown while the graph is paused on interrupt()) */
.approval-card {
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg);
    padding: 1rem 1.2rem;
    margin: 0.5rem 0 0.9rem;
    box-shadow: var(--shadow-md);
}
.approval-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 600;
    color: var(--text-strong);
    margin-bottom: 0.35rem;
}
.approval-title::before {   /* amber dot = "waiting on you" */
    content: "";
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: var(--warn);
    box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.18);
}
.approval-body {
    color: var(--text);
    font-size: 0.93rem;
    line-height: 1.6;
    white-space: normal;
}

/* ---------- 9. Accessibility ---------- */
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .empty-state { padding-top: 10vh; }
    .empty-state h1 { font-size: 1.6rem; }
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ========================= Thread helpers =========================

# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())


# Add a new thread ID to the conversation list
def add_thread(thread_id):

    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # ========================= HITL ADDED =========================
    # Clear any pending human approval request
    st.session_state["pending_hitl"] = None
    # =============================================================

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])


# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])


# Turn the first user message into a short sidebar label
def make_title(text, max_length=40):

    # Collapse whitespace and newlines into single spaces
    clean = " ".join(str(text).split())

    if len(clean) > max_length:
        clean = clean[: max_length - 1].rstrip() + "…"

    return clean or "New chat"


# Return the sidebar label for a thread.
# Titles are cached in session state so each thread is only loaded once.
def get_thread_title(thread_id):

    titles = st.session_state["thread_titles"]

    if thread_id in titles:
        return titles[thread_id]

    first_user_text = None

    if thread_id == st.session_state["thread_id"]:

        # The open conversation is already in the UI history
        for item in st.session_state["message_history"]:
            if item["role"] == "user":
                first_user_text = item["content"]
                break

    else:

        # Other conversations are read from the checkpointer
        try:
            for item in load_conversation(thread_id):
                if isinstance(item, HumanMessage):
                    first_user_text = item.content
                    break
        except Exception:
            first_user_text = None

    if first_user_text:
        titles[thread_id] = make_title(first_user_text)
        return titles[thread_id]

    # Conversation has no messages yet
    return "New chat"


# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.

    Returns:
        The pending Interrupt object, or None.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # Read the current checkpoint state
        state_snapshot = chatbot.get_state(config)

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other LangGraph versions store interrupts inside tasks
        tasks = getattr(
            state_snapshot,
            "tasks",
            ()
        ) or ()

        for task in tasks:

            task_interrupts = getattr(
                task,
                "interrupts",
                ()
            ) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:

        # A newly created thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """

    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:

        save_pending_interrupt(
            thread_id,
            pending_interrupt
        )

    else:

        current_pending = st.session_state.get(
            "pending_hitl"
        )

        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.

    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """

    pending_hitl = st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or reject."
        )

        return

    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
    resume_config = {
        "configurable": {
            "thread_id": interrupted_thread_id
        },
        "metadata": {
            "thread_id": interrupted_thread_id
        },
        "run_name": "hitl_resume_trace",
    }

    try:

        # Display the resumed response
        with st.chat_message("assistant"):

            status_holder = {
                "box": st.status(
                    "Resuming the action…",
                    expanded=True
                )
            }

            def resumed_ai_only_stream():

                # Resume the graph with the human decision
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):

                    # Update tool execution status
                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        status_holder["box"].update(
                            label=f"Using `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                    # Stream only assistant-generated text
                    if isinstance(
                        message_chunk,
                        AIMessage
                    ):

                        if message_chunk.content:
                            yield message_chunk.content

            # Display the streamed final answer
            resumed_ai_message = st.write_stream(
                resumed_ai_only_stream()
            )

            # Check whether another interrupt occurred
            next_interrupt = get_pending_interrupt(
                interrupted_thread_id
            )

            if next_interrupt is not None:

                save_pending_interrupt(
                    interrupted_thread_id,
                    next_interrupt
                )

                status_holder["box"].update(
                    label="Another approval is required",
                    state="complete",
                    expanded=False
                )

            else:

                # No more pending approval
                st.session_state["pending_hitl"] = None

                status_holder["box"].update(
                    label="Action completed",
                    state="complete",
                    expanded=False
                )

        # Save the assistant response in Streamlit UI history
        if resumed_ai_message:

            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        # Rerun so the response appears in normal chat order
        st.rerun()

    except Exception as error:

        st.error(
            f"Could not resume the action: {error}"
        )


# ========================= Session state =========================

# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()


# Cache of sidebar labels, keyed by thread ID
if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}


# ========================= HITL ADDED =========================

# Store the currently pending human approval request
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

# =============================================================


# Add the current thread to the conversation list
add_thread(st.session_state["thread_id"])


# ========================= HITL ADDED =========================

# Recover pending approval after page refresh or rerun
sync_pending_interrupt(
    st.session_state["thread_id"]
)

# =============================================================


# ========================= Sidebar =========================

# Brand block (styled by .brand in the CSS above)
st.sidebar.markdown(
    '<div class="brand">'
    '<div class="brand-mark">✨</div>'
    '<div class="brand-name">AI Assistant</div>'
    '</div>',
    unsafe_allow_html=True
)


# Create a button for starting a new conversation.
# The key gives the element a stable CSS hook: .st-key-new_chat
if st.sidebar.button(
    "New chat",
    key="new_chat",
    icon=":material/add:",
    use_container_width=True
):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()


st.sidebar.divider()

st.sidebar.markdown(
    '<div class="side-label">Conversations</div>',
    unsafe_allow_html=True
)


# Display all conversation threads in reverse order
# This shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:

    is_active = thread_id == st.session_state["thread_id"]

    # One sidebar row per conversation.
    # Keys starting with "thread_" are styled by [class*="st-key-thread_"].
    # The active conversation uses type="primary" to get its highlight.
    if st.sidebar.button(
        get_thread_title(thread_id),
        key=f"thread_{thread_id}",
        type="primary" if is_active else "secondary",
        use_container_width=True,
        help=str(thread_id)
    ):

        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []

        # Loop through all saved messages
        for message in messages:

            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types, such as ToolMessage
            else:
                continue

            # Convert the LangChain message into a dictionary
            temp_messages.append({
                "role": role,
                "content": message.content
            })

        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # ========================= HITL ADDED =========================

        # Restore any pending approval for this conversation
        sync_pending_interrupt(thread_id)

        # =============================================================

        # Rerun the application to display the loaded messages
        st.rerun()


# ========================= Main chat interface =========================

# Pending approval for the currently selected conversation (if any)
pending_hitl = st.session_state.get(
    "pending_hitl"
)

current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id")
    == st.session_state["thread_id"]
)


# Empty state: shown only before the first message of a conversation
if not st.session_state["message_history"] and not current_thread_has_pending_hitl:

    st.markdown(
        '<div class="empty-state">'
        '<h1>What can I help you with?</h1>'
        '<p>Ask a question, attach a PDF to chat about it, or request a '
        'stock purchase. You approve any purchase before it happens.</p>'
        '</div>',
        unsafe_allow_html=True
    )


# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Render as markdown so lists, bold text and code blocks display properly
        st.markdown(message["content"])


# ========================= HITL approval interface =========================

# Display approval controls
if current_thread_has_pending_hitl:

    # Escape the prompt text before placing it in custom HTML
    safe_prompt = html.escape(
        pending_hitl["prompt"]
    ).replace("\n", "<br>")

    st.markdown(
        '<div class="approval-card">'
        '<div class="approval-title">Your approval is needed</div>'
        f'<div class="approval-body">{safe_prompt}</div>'
        '</div>',
        unsafe_allow_html=True
    )

    approve_column, reject_column = st.columns(2)

    # Approve button
    with approve_column:

        if st.button(
            "Approve purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            icon=":material/check:",
            use_container_width=True
        ):

            # Send "yes" back to interrupt()
            resume_hitl_execution("yes")

    # Reject button
    with reject_column:

        if st.button(
            "Reject",
            key=f"reject_{st.session_state['thread_id']}",
            icon=":material/close:",
            use_container_width=True
        ):

            # Send "no" back to interrupt()
            resume_hitl_execution("no")


# ========================= Fixed chat input with PDF upload =========================

# Keep st.chat_input directly in the main body.
# This keeps it fixed at the bottom of the screen.
#
# accept_file=True adds the attachment button inside the chat input.
# file_type=["pdf"] allows PDF files only.
submission = st.chat_input(
    (
        "Waiting for your decision above"
        if current_thread_has_pending_hitl
        else "Message AI Assistant"
    ),
    accept_file=True,
    file_type=["pdf"],

    # Disable input while waiting for human approval
    disabled=current_thread_has_pending_hitl
)


# Default user input value
user_input = None


# Process the submitted text and PDF
if submission:

    # Get the text entered by the user
    user_input = submission.text

    # Get the uploaded files
    # This is always a list when accept_file is enabled
    uploaded_files = submission.files

    # Process the uploaded PDF if one was attached
    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        # Store the temporary file path
        temporary_file_path = None

        try:

            # Save the uploaded PDF as a temporary local file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temporary_file:

                temporary_file.write(
                    uploaded_pdf.getvalue()
                )

                temporary_file_path = temporary_file.name

            # Call the existing backend RAG ingestion function
            with st.spinner(
                f"Processing {uploaded_pdf.name}…"
            ):

                ingest_rag_document(
                    temporary_file_path
                )

            # Display PDF processing confirmation
            st.toast(
                f"{uploaded_pdf.name} is ready to use.",
                icon=":material/check_circle:"
            )

        except Exception as error:

            # Display PDF processing error
            st.error(
                f"Could not process the PDF: {error}"
            )

        finally:

            # Delete the temporary PDF after indexing
            if (
                temporary_file_path
                and os.path.exists(temporary_file_path)
            ):
                os.remove(temporary_file_path)


# Run this block after the user submits a text message
if user_input:

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.markdown(user_input)

    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):

        # Use a mutable holder so the generator can set/modify it
        status_holder = {
            "box": None
        }

        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                # Lazily create & update the SAME status container
                # when any tool runs
                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"Using `{tool_name}`…",
                            expanded=True
                        )

                    else:

                        status_holder["box"].update(
                            label=f"Using `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(
                    message_chunk,
                    AIMessage
                ):
                    yield message_chunk.content

            # ========================= HITL ADDED =========================

            # interrupt() pauses the graph without returning
            # a completed ToolMessage.
            #
            # Inspect the saved checkpoint after streaming ends.
            pending_interrupt = get_pending_interrupt(
                st.session_state["thread_id"]
            )

            if pending_interrupt is not None:

                # Save the interrupt for displaying approval buttons
                save_pending_interrupt(
                    st.session_state["thread_id"],
                    pending_interrupt
                )

                yield (
                    "\n\nThis stock purchase needs your approval. "
                    "Use the **Approve purchase** or **Reject** "
                    "button below."
                )

            # =============================================================

        ai_message = st.write_stream(
            ai_only_stream()
        )

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:

            # Check whether execution is waiting for approval
            if get_pending_interrupt(
                st.session_state["thread_id"]
            ) is not None:

                status_holder["box"].update(
                    label="Waiting for your approval",
                    state="complete",
                    expanded=False
                )

            else:

                status_holder["box"].update(
                    label="Tool finished",
                    state="complete",
                    expanded=False
                )

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })

    # Decide whether the page needs one more rerun
    needs_rerun = False

    # ========================= HITL ADDED =========================

    # Approval controls are rendered earlier in the script.
    # Rerun so they appear immediately after interrupt().
    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id")
        == st.session_state["thread_id"]
    ):
        needs_rerun = True

    # =============================================================

    # The sidebar is drawn before the chat runs, so a brand-new conversation
    # would still read "New chat". Rerun once to show its real title.
    if st.session_state["thread_id"] not in st.session_state["thread_titles"]:
        needs_rerun = True

    if needs_rerun:
        st.rerun()