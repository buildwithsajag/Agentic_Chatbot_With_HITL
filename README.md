
# 🤖 Agentic Chatbot With HITL

A modern, stateful agentic chatbot built with **LangGraph** featuring **Human-in-the-Loop (HITL)** workflows, dynamic tool calling, Retrieval-Augmented Generation (RAG), and a polished **Streamlit** interface.

🔗 **Live Demo:** [https://agentic-chatbot-with-hitl-10.onrender.com/](https://agentic-chatbot-with-hitl-10.onrender.com/)

> ⚠️ **Note:** This app is deployed on **Render's free tier**. If the page appears blank or slow on first load, it's a **cold start** — please wait 30–60 seconds and refresh.

---

## 📖 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#️-installation--setup)
- [Environment Variables](#-environment-variables)
- [Running the App](#️-running-the-app)
- [Deployment](#️-deployment)
- [Example Prompts](#-example-prompts)
- [Human-in-the-Loop Workflow](#-human-in-the-loop-workflow)
- [RAG Pipeline](#-rag-pipeline)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🚀 Features

- **Agentic Workflow** — Uses LangGraph to create cyclic, stateful agent graphs with conditional routing.
- **Human-in-the-Loop (HITL)** — Interrupt points that let users review, approve, or reject the agent's planned actions before execution.
- **Dynamic Tool Calling** — Extensible architecture where the agent picks from custom Python tools (search, calculator, stock, weather, RAG, purchase).
- **RAG Capabilities** — Retrieval-Augmented Generation using a local **FAISS** vector store with **FastEmbed** embeddings (no heavy torch dependency).
- **Persistent Memory** — SQLite-backed thread-level persistence via `langgraph-checkpoint-sqlite`.
- **Streamlit UI** — Clean, interactive frontend for chatting, managing threads, and handling approvals.
- **Free-Tier Friendly** — Runs on Render's free plan by using lightweight local embeddings instead of torch/sentence-transformers.

---

## 📂 Project Structure

```
AgenticChatbot/
│
├── app_hitl.py                          # 🌟 Main Streamlit app (HITL workflow)
├── app_tool.py                          # Streamlit demo: basic tool calling
├── app_simple.py                        # Streamlit demo: minimal chatbot
├── app_thread.py                        # Streamlit demo: thread management
├── app_db.py                            # Streamlit demo: SQLite interaction
├── app_rag.py                           # Streamlit demo: RAG features
│
├── agentic_chatbot_hitl_backend.py      # 🧠 Core LangGraph backend (HITL)
├── agentic_chatbot_tool_backend.py      # Core backend: standard tool calling
├── agentic_chatbot_db_backend.py        # Core backend: SQLite checkpointing
├── agentic_chatbot_rag_backend.py       # Core backend: RAG / vector search
├── chatbot_with_hitl.py                 # Experimental HITL script
│
├── faiss_db/                            # Local FAISS vector store
│   ├── index.faiss
│   └── index.pkl
│
├── my_paper.pdf                         # Sample PDF for RAG testing
├── product_developer_roadmap.pdf        # Additional sample PDF
├── Chatbot_workflow.ipynb               # Prototyping / workflow visualization
├── Persistence.excalidraw               # Architecture diagram (Excalidraw)
│
├── chatbot.db                           # Local SQLite DB (gitignored)
├── requirements.txt                     # Python dependencies
├── Dockerfile                           # Container build for Render
├── .dockerignore                        # Docker build exclusions
├── .gitignore                           # Git exclusions
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Orchestration** | LangGraph |
| **LLM** | Groq (`openai/gpt-oss-120b`) |
| **Embeddings** | FastEmbed (`BAAI/bge-small-en-v1.5`) |
| **Vector Store** | FAISS (CPU) |
| **Persistence** | SQLite via `langgraph-checkpoint-sqlite` |
| **Tools** | Tavily Search, yfinance, OpenWeather, PyPDF, custom calculator |
| **Frontend** | Streamlit |
| **PDF Loading** | `pypdf` + `PyPDFLoader` |
| **Deployment** | Render (Docker) |
| **Python** | 3.11 |

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/buildwithsajag/Agentic_Chatbot_With_HITL.git
cd Agentic_Chatbot_With_HITL
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

### 5. (Optional) Rebuild the FAISS index

If you change the embedding model or add new PDFs:

```bash
python -c "from agentic_chatbot_hitl_backend import ingest_rag_document; ingest_rag_document('my_paper.pdf')"
```

---

## 🔑 Environment Variables

| Variable | Required | Purpose | Where to Get It |
| :--- | :---: | :--- | :--- |
| `GROQ_API_KEY` | ✅ | LLM provider (Groq) | [console.groq.com/keys](https://console.groq.com/keys) |
| `TAVILY_API_KEY` | ✅ | Web search tool | [tavily.com](https://tavily.com) |
| `OPENWEATHER_API_KEY` | ✅ | Current weather tool | [openweathermap.org/api](https://openweathermap.org/api) |
| `PORT` | ✅ (Render only) | Port Streamlit binds to | Set to `10000` on Render |

---

## 🖥️ Running the App

### Locally

```bash
streamlit run app_hitl.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### With Docker

```bash
docker build -t agentic-chatbot .
docker run -p 8501:8501 \
  -e GROQ_API_KEY=your_key \
  -e TAVILY_API_KEY=your_key \
  -e OPENWEATHER_API_KEY=your_key \
  agentic-chatbot
```

---

## ☁️ Deployment

This project is deployed on **Render** using Docker.

🔗 **Live URL:** [https://agentic-chatbot-with-hitl-10.onrender.com/](https://agentic-chatbot-with-hitl-10.onrender.com/)

### Deploy Your Own

1. **Push the repo to GitHub.**
2. On [Render](https://render.com), click **New + → Web Service**.
3. Connect your GitHub repo.
4. Set:
   - **Language:** `Docker`
   - **Dockerfile Path:** *(leave blank)*
   - **Instance Type:** `Free`
5. Add environment variables:
   - `PORT` = `10000`
   - `GROQ_API_KEY` = your key
   - `TAVILY_API_KEY` = your key
   - `OPENWEATHER_API_KEY` = your key
6. Click **Create Web Service**.
7. Wait for the build to finish — the app goes live at `https://<your-service>.onrender.com`.

> 💡 **Tip:** To avoid cold starts on the free tier, set up a free uptime pinger like [UptimeRobot](https://uptimerobot.com) to hit your URL every 10 minutes.

---

## 💬 Example Prompts

Try these once the app is running:

| Prompt | What It Triggers |
| :--- | :--- |
| `What is the latest stock price of AAPL?` | `get_stock_price` (yfinance) |
| `What is the latest stock price of GOOGL?` | `get_stock_price` (yfinance) |
| `Tell me the current weather in Tokyo.` | `get_current_weather` |
| `Search the web for the latest AI news.` | `TavilySearch` |
| `What does my uploaded PDF say about transformers?` | `rag_tool` (FAISS retrieval) |
| `Calculate math.sqrt(144) + 10` | `calculator` |
| `Buy 5 shares of TSLA` | `purchase_stock` → **HITL approval** |

---

## 🧑‍⚖️ Human-in-the-Loop Workflow

The HITL feature lets a human approve or reject sensitive actions before they execute.

**Flow:**

1. User asks: *"Buy 5 shares of TSLA."*
2. Agent decides to call `purchase_stock(order="TSLA,5")`.
3. The graph **interrupts** and Streamlit shows a prompt: *"Approve buying 5 shares of TSLA? (yes/no)"*.
4. User types **yes** → purchase executes.
5. User types anything else → purchase is cancelled.

This is powered by LangGraph's `interrupt()` and `Command` primitives, plus the SQLite checkpointer that persists state between turns.

---

## 📚 RAG Pipeline

1. **Load PDF** — `PyPDFLoader` reads the document.
2. **Split** — `RecursiveCharacterTextSplitter` chunks it (1000 chars, 200 overlap).
3. **Embed** — `FastEmbedEmbeddings` (`BAAI/bge-small-en-v1.5`) converts chunks to vectors.
4. **Store** — FAISS saves the index to `faiss_db/`.
5. **Retrieve** — On a query, `rag_tool` fetches the top-4 most similar chunks.

> ⚠️ **Important:** If you change the embedding model, you **must** rebuild `faiss_db/` — vectors from different models are incompatible.

---

## 🐛 Troubleshooting

| Problem | Cause | Fix |
| :--- | :--- | :--- |
| **Blank page on Render for 1–3 min** | Free-tier cold start | Wait & refresh, or use an uptime pinger |
| **`ModuleNotFoundError` at runtime** | Missing dependency | Add it to `requirements.txt`, push, redeploy |
| **`ResolutionImpossible` during build** | Conflicting package versions | Loosen version pins or use `pipreqs` to regenerate |
| **Stock price returns "couldn't retrieve"** | Alpha Vantage rate limit (25/day) | Switched to `yfinance` — no limits |
| **RAG returns irrelevant results** | `faiss_db/` built with a different embedding model | Rebuild with `ingest_rag_document()` |
| **`chatbot.db` grows or resets** | Render free tier has ephemeral disk | Add a persistent disk (paid) or accept reset |
| **`TavilySearch` fails** | Missing `TAVILY_API_KEY` | Add it in Render's Environment tab |

---

## 📜 License

This project is released for **educational and demonstration purposes**. Feel free to fork, modify, and learn from it.

---

## 🙌 Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) — Stateful agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — Building blocks for LLM apps
- [Streamlit](https://streamlit.io) — Fast, beautiful UI
- [Groq](https://groq.com) — Ultra-fast LLM inference
- [FastEmbed](https://github.com/qdrant/fastembed) — Lightweight local embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Tavily](https://tavily.com) — Web search for AI agents
- [Render](https://render.com) — Deployment platform

---

<div align="center">

**⭐ If you found this project useful, consider giving it a star! ⭐**

Made with ❤️ by [Sajag](https://github.com/buildwithsajag)

</div>
