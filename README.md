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
- [Running the App](#-running-the-app)
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
