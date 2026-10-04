from agentic_chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage
import streamlit as st
import uuid

st.title("Agentic Chatbot with LangGraph and Groq")

# Use a unique thread_id per user session to prevent shared memory
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # ⬇️⬇️⬇️ THIS ENTIRE BLOCK MUST BE INDENTED (4 spaces)
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )

    # ⬇️⬇️⬇️ This line must also be indented to be inside `if user_input:`
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})