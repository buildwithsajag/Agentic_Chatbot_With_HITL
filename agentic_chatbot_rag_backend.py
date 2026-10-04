from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import math
import requests
import os

# RAG imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()


# ==========================================================
# LLM — Groq (NO token limits, larger TPM model, auto-retry)
# ==========================================================
llm = ChatGroq(
    model="openai/gpt-oss-120b",       # <-- Use the current production model
    temperature=0.7,
    max_retries=3,
)

# ==========================================================
# Embeddings — FREE, runs locally
# ==========================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


DB_PATH = "faiss_db"


def ingest_rag_document(file_path: str) -> None:
    """Load a PDF, split it, embed it, and persist a FAISS index."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Smaller chunks = fewer tokens per RAG call (no LLM limit needed)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,       # was 1000
        chunk_overlap=100,    # was 200
    )
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)
    print(f"[RAG] Ingested {len(chunks)} chunks into '{DB_PATH}'")


def get_retriever():
    """Load the FAISS index from disk and return a retriever."""
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},   # was 4 — fewer chunks = fewer tokens
    )


# ==========================================================
# Tools
# ==========================================================
@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    try:
        retriever = get_retriever()
        documents = retriever.invoke(query)

        if not documents:
            return "No relevant information was found in the PDF."

        formatted = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "Unknown page")
            formatted.append(
                f"Document {i}\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content: {doc.page_content}"
            )
        return "\n\n".join(formatted)

    except FileNotFoundError:
        return "No PDF has been uploaded yet. Please upload a document first."
    except Exception as e:
        return f"RAG retrieval error: {e}"


search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY is not set."}

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"error": f"Failed to fetch stock price: {e}"}


@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Dhaka", "London, UK", or "New York, US".

    Returns:
        A formatted current weather report.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"
        geo_response = requests.get(
            geocoding_url,
            params={"q": location, "limit": 1, "appid": api_key},
            timeout=10,
        )
        geo_response.raise_for_status()
        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        weather_response = requests.get(
            weather_url,
            params={
                "lat": latitude,
                "lon": longitude,
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        visibility_meters = weather_data.get("visibility")
        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        parts = [resolved_name]
        if state:
            parts.append(state)
        if country:
            parts.append(country)
        display_location = ", ".join(parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"
        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."
        return f"Weather API returned an HTTP error: {status_code}"
    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"
    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"


# ==========================================================
# Tool list + tool-aware LLM
# ==========================================================
tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
]
llm_with_tools = llm.bind_tools(tools)


# ==========================================================
# State
# ==========================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================================
# Node 1 — chat node (with history trimming)
# ==========================================================
def chat_node(state: ChatState):
    """LLM node that can answer directly or call an appropriate tool."""

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer. "
            "Keep answers concise unless the user asks for detail."
        )
    )

    # 🧹 Trim old history so long conversations don't blow the TPM budget.
    # The LLM itself is NOT limited — only the *input* history is bounded.
    trimmed_history = trim_messages(
        state["messages"],
        max_tokens=4000,       # keep last ~4K tokens of history
        strategy="last",
        token_counter=llm,
        include_system=False,
        start_on="human",
    )

    messages = [system_message, *trimmed_history]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ==========================================================
# Node 2 — tool node
# ==========================================================
tool_node = ToolNode(tools)


# ==========================================================
# Checkpointer (SQLite)
# ==========================================================
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)


# ==========================================================
# Graph
# ==========================================================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


# ==========================================================
# Helper — get all thread IDs from the checkpointer
# ==========================================================
def get_all_threads() -> list[str]:
    """
    Return a list of all unique thread IDs stored in the checkpointer.
    SqliteSaver.list(None) yields tuples: (config_dict, checkpoint, metadata)
    """
    all_threads = set()
    try:
        for ckpt_tuple in checkpoint.list(None):
            config = ckpt_tuple[0] if isinstance(ckpt_tuple, tuple) else ckpt_tuple
            thread_id = (
                config.get("configurable", {}).get("thread_id")
                if isinstance(config, dict)
                else None
            )
            if thread_id:
                all_threads.add(thread_id)
    except Exception as e:
        print(f"[get_all_threads] Warning: {e}")
    return list(all_threads)