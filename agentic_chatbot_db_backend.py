from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# 1. REMOVED: from app_thread import CONFIG (This will crash standalone scripts)
# We will create a local CONFIG for testing below.

load_dotenv()  # Load environment variables from .env file

# 2. FIXED: Changed the model to a valid Groq model
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']
    
    # send to llm
    response = llm.invoke(messages)
    
    # response store state
    return {'messages': [response]}

# 3. FIXED: Removed duplicate graph and checkpoint initializations
# Create the database connection
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False) 

# Initialize the SqliteSaver WITH the connection
checkpoint = SqliteSaver(conn) 

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# Compile the graph with the checkpointer
chatbot = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)