# Agentic Chatbot With HITL

A modern, stateful agentic chatbot built with **LangGraph** featuring **Human-in-the-Loop (HITL)** workflows, dynamic tool calling, and a polished **Streamlit** interface.

## 🚀 Features

*   **Agentic Workflow:** Utilizes LangGraph to create cyclic, stateful agent graphs.
*   **Human-in-the-Loop (HITL):** Built-in interruption points that allow users to review, approve, or edit the agent's planned actions before execution.
*   **Dynamic Tool Calling:** Extensible architecture allowing the agent to use custom Python functions as tools.
*   **RAG Capabilities:** Includes experimental Retrieval-Augmented Generation modules using local vector stores (FAISS).
*   **Persistent Memory:** Uses SQLite for thread-level persistence and checkpointing.
*   **Streamlit UI:** A clean, interactive frontend for chatting with the agent, managing threads, and handling approvals.

## 📂 Project Structure

Here is a brief overview of the key files in this repository:

### 🖥️ User Interfaces (Streamlit Apps)
*   `app_hitl.py`: **(Main App)** The primary Streamlit interface featuring the Human-in-the-Loop approval workflow.
*   `app_tool.py`: Streamlit app demonstrating basic tool-calling capabilities.
*   `app_simple.py`: A minimal, basic implementation of the chatbot.
*   `app_thread.py`: Demonstrates thread management and conversation persistence.
*   `app_db.py`: UI implementation interacting with the SQLite database.
*   `app_rag.py`: UI implementation demonstrating the Retrieval-Augmented Generation features.

### ⚙️ Backend Logic (LangGraph & Agents)
*   `agentic_chatbot_hitl_backend.py`: The core LangGraph logic defining the HITL state machine, nodes, and interrupt conditions.
*   `agentic_chatbot_tool_backend.py`: The core logic for standard agentic tool-calling workflows.
*   `agentic_chatbot_db_backend.py`: Backend logic handling SQLite database interactions and state checkpointing.
*   `agentic_chatbot_rag_backend.py`: Backend logic handling document retrieval and vector search integration.
*   `chatbot_with_hitl.py`: An alternative/experimental script for HITL execution.

### 📁 Data & Assets
*   `faiss_db/`: Directory containing the local FAISS vector store for RAG.
*   `Chatbot_workflow.ipynb`: Jupyter Notebook used for prototyping and visualizing the LangGraph workflows.
*   `Persistence.excalidraw`: Excalidraw diagram file detailing the architecture/flow.
*   `chatbot.db`: *(Ignored in Git)* Local SQLite database for storing chat histories.
*   `my_paper.pdf`: Sample document used for testing RAG ingestion.
*   `product_developer_roadmap.pdf`: Additional sample document for RAG.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/buildwithsajag/Agentic_Chatbot_With_HITL.git
   cd Agentic_Chatbot_With_HITL
