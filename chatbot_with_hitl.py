import math
import os
import sqlite3
import traceback
from typing import Annotated, Any, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

load_dotenv()


FAISS_DIR = "faiss_db"
SQLITE_DB = "chatbot.db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.7,
    max_tokens=4096,
    max_retries=3,
    api_key=os.getenv("GROQ_API_KEY"),
)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def ingest_rag_document(file_path: str) -> None:
    """Load a PDF, split it, and store embeddings locally."""
    docs = PyPDFLoader(file_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    FAISS.from_documents(chunks, embeddings).save_local(FAISS_DIR)


def get_retriever():
    store = FAISS.load_local(
        folder_path=FAISS_DIR,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


@tool
def rag_tool(query: str) -> str:
    """Look up content in the ingested PDF.

    Use for factual or conceptual questions that should be answered
    from the stored document rather than from general knowledge.
    """
    try:
        documents = get_retriever().invoke(query)
    except Exception as exc:
        return f"PDF index unavailable: {exc}"

    if not documents:
        return "No matching content found in the PDF."

    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        parts.append(
            f"[{i}] source={source} page={page}\n{doc.page_content}"
        )
    return "\n\n".join(parts)


search_tool = TavilySearch(max_results=5)


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression, e.g. '2 + 2' or 'math.sqrt(16)'."""
    allowed = {
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def get_stock_price(symbol: str) -> dict:
    """Return the latest quote for a stock symbol such as 'AAPL' or 'TSLA'."""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}"
        "&apikey=9MZO2JUBR7IFNTOI"
    )
    return requests.get(url).json()


@tool
def purchase_stock(order: str) -> dict:
    """Place a simulated buy order, pending human approval.

    The order argument is a single string in the form
    'SYMBOL,QUANTITY' -- for example 'AAPL,5'. Execution pauses
    and asks the caller to approve before the order is confirmed.
    """
    try:
        symbol, qty_str = order.split(",")
        symbol = symbol.strip().upper()
        quantity = int(qty_str.strip())
    except Exception:
        return {
            "status": "error",
            "message": "Expected format 'SYMBOL,QUANTITY', e.g. 'AAPL,5'.",
        }

    decision = interrupt(
        f"Approve buying {quantity} shares of {symbol}? (yes/no)"
    )

    if isinstance(decision, str) and decision.strip().lower() == "yes":
        return {
            "status": "success",
            "symbol": symbol,
            "quantity": quantity,
            "message": f"Order placed for {quantity} shares of {symbol}.",
        }

    return {
        "status": "cancelled",
        "symbol": symbol,
        "quantity": quantity,
        "message": f"Order for {quantity} shares of {symbol} was declined.",
    }


@tool
def get_current_weather(location: str) -> str:
    """Return the current weather for a city, e.g. 'Dhaka' or 'London, UK'."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "OPENWEATHER_API_KEY is not set."

    try:
        geo = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": location, "limit": 1, "appid": api_key},
            timeout=10,
        )
        geo.raise_for_status()
        matches: list[dict[str, Any]] = geo.json()
        if not matches:
            return f"Location not found: {location}"

        place = matches[0]
        weather = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": place["lat"],
                "lon": place["lon"],
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )
        weather.raise_for_status()
        data = weather.json()

        main = data["main"]
        wind = data.get("wind", {}).get("speed", "n/a")
        visibility = data.get("visibility")
        visibility_km = (
            round(visibility / 1000, 1) if visibility is not None else "n/a"
        )

        name_parts = [place.get("name", location)]
        if place.get("state"):
            name_parts.append(place["state"])
        if place.get("country"):
            name_parts.append(place["country"])
        display = ", ".join(name_parts)

        return (
            f"Current weather in {display}:\n"
            f"- Condition: {data['weather'][0]['description'].title()}\n"
            f"- Temperature: {main['temp']} C\n"
            f"- Feels like: {main['feels_like']} C\n"
            f"- Humidity: {main['humidity']}%\n"
            f"- Pressure: {main['pressure']} hPa\n"
            f"- Wind speed: {wind} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "Weather request timed out."
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response else "unknown"
        if code == 401:
            return "OpenWeather API key is invalid or inactive."
        return f"Weather API returned HTTP {code}."
    except requests.RequestException as exc:
        return f"Weather service unreachable: {exc}"
    except (KeyError, TypeError, ValueError) as exc:
        return f"Unexpected weather response: {exc}"


tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
    purchase_stock,
]

llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = (
    "You are a helpful agentic chatbot with access to several tools.\n\n"

    "Rules:\n"
    "1. Greetings and casual conversation are answered directly; no tools.\n"
    "2. Only call a tool when the request clearly needs one.\n"
    "3. Never guess tool arguments. Use the exact formats below.\n\n"

    "Tools:\n"
    "- rag_tool(query): questions about the ingested PDF.\n"
    "- search_tool(query): current events and web lookups.\n"
    "- calculator(expression): arithmetic, e.g. '2+2' or 'math.sqrt(16)'.\n"
    "- get_stock_price(symbol): e.g. 'AAPL'.\n"
    "- purchase_stock(order): 'SYMBOL,QUANTITY' e.g. 'AAPL,5'. Requires human approval.\n"
    "- get_current_weather(location): e.g. 'Dhaka'.\n\n"

    "Answer general questions directly when no tool is needed. "
    "After a tool returns, give a clear final response."
)


def chat_node(state: ChatState):
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as exc:
        # Surfacing the underlying error makes debugging much easier than
        # letting the graph blow up with an opaque traceback.
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return {"messages": [AIMessage(content=f"Model error: {exc}")]}

    return {"messages": [response]}


tool_node = ToolNode(tools)


conn = sqlite3.connect(database=SQLITE_DB, check_same_thread=False)
checkpoint = SqliteSaver(conn)


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads() -> list[str]:
    threads = set()
    for ckpt in checkpoint.list(None):
        threads.add(ckpt.config["configurable"]["thread_id"])
    return list(threads)


def _print_assistant(message: BaseMessage) -> None:
    content = message.content
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    print(f"Bot: {content}\n")


def main() -> None:
    print("Agentic chatbot. Type 'exit' to quit.\n")

    thread_id = "demo-thread"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("Bye.")
            return

        state = {"messages": [HumanMessage(content=user_input)]}

        try:
            result = chatbot.invoke(state, config=config)
        except Exception as exc:
            print(f"Graph run failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        # purchase_stock pauses here until the caller approves or declines.
        interrupts = result.get("__interrupt__", [])
        if interrupts:
            print(f"Approval needed: {interrupts[0].value}")
            decision = input("Your decision: ").strip().lower()
            result = chatbot.invoke(Command(resume=decision), config=config)

        _print_assistant(result["messages"][-1])


if __name__ == "__main__":
    main()