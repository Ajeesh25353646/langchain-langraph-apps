# LangChain & LangGraph Apps 🤖

> **🚀 Started in June 2025** — Building AI agents before the agents boom. This repository showcases production-grade LLM applications built with LangChain, LangGraph, and modern AI orchestration frameworks, demonstrating real-world implementations of multi-agent systems, RAG pipelines, and autonomous workflows.

---

## 📋 Repository Overview

This repository traces my **journey from AI beginner to production-grade developer**. Started in **June 2025** with simple chatbots and evolved into complex multi-agent systems with advanced RAG pipelines. The projects showcase how I progressed from basic implementations to building scalable, production-ready AI applications.

**What you'll find:**
- 🌱 **Learning Projects**: Simple chatbots and basic RAG (great for understanding fundamentals)
- 🚀 **Production-Grade Systems**: Multi-agent orchestration, multimodal RAG, comprehensive testing
- 📚 **Complete Journey**: Step-by-step evolution of AI engineering skills

### 🎯 What’s Inside

### 🌱 Learning Fundamentals (simple-apps/)
- **💬 Basic Chatbots**: Stateless and stateful conversation patterns
- **💰 Tool-Using Agents**: Currency converter with ReAct pattern
- **📄 Text RAG**: PDF question-answering basics

### 🚀 Production-Grade Systems (Advanced_apps/)
- **🤖 Multi-Agent Orchestration**: Coordinated workflows with LangGraph
- **🖼️ Multimodal RAG**: PDF Q&A with image/table understanding (CV project)
- **💼 Strategic Intelligence**: CFO-level forex analysis agent
- **📺 Video Analysis**: YouTube transcript analyzer with timestamps
- **🔒 Dual-Mode Architecture**: Both cloud-enhanced and 100% local operation

---

## 📁 Project Structure

```
langchain-langraph-apps/
│
├── 📂 simple-apps/                          # 🌱 Learning fundamentals
│   ├── 💬 chatbot_without_memory/           # Stateless chatbot (basic patterns)
│   ├── 🧠 chatbot_with_memory/              # Multi-turn conversation (state management)
│   ├── 💰 currency_converter_bot/           # Real-time FX with ReAct agent
│   └── 📄 Text_pdf_RAG/                     # Text-based PDF Q&A (basic RAG)
│
├── 📂 Advanced_apps/                        # 🚀 Production-grade systems
│   ├── 🏦 Advanced_Strategic_Finance_Agent/ # CFO-level forex intelligence
│   │   ├── Strategic_Finance_Agent.py       # Multi-agent orchestration (LangGraph)
│   │   ├── test_agent.py                    # 10 comprehensive unit tests
│   │   └── requirements.txt
│   │
│   ├── 🖼️ Multimodal_PDF_RAG/              # ⭐ CV Project - Image-aware PDF assistant
│   │   ├── Advanced_multimodal_rag.py       # VLM-powered image/table understanding
│   │   ├── requirements.txt                 # Dependencies
│   │   └── .env.example                     # Configuration template
│   │
│   ├── 📑 Text_Pdf_QA/                      # Advanced PDF RAG with multiple retrievers
│   │   ├── Advanced_pdf_rag.py              # Contextual compression, multi-query
│   │   └── requirements.txt
│   │
│   └── 📺 youtube_QA_bot/                   # YouTube transcript analyzer
│       ├── app.py                           # Timestamp-referenced answers
│       └── requirements.txt
│
├── 📄 requirements.txt                      # Root dependencies
└── 📄 README.md                             # This file
```

---

## 🚀 Quick Start

### Prerequisites

* **Python 3.11+** (recommended) or Python 3.8+
* **pip** package manager
* **API Keys** (optional for cloud models):
  - [Google AI Studio](https://aistudio.google.com/) - Gemini models
  - [Tavily AI](https://tavily.com/) - AI-optimized search

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Ajeesh25353646/langchain-langraph-apps.git
   cd langchain-langgraph-apps
   ```

2. **Navigate to your project of choice**

   ```bash
   # For simple apps
   cd simple-apps/chatbot_with_memory

   # For advanced apps
   cd Advanced_apps/Advanced_Strategic_Finance_Agent
   ```

3. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

   > *Note: Instructions optimized for WSL2/Ubuntu. Windows users may need slight adjustments.*

4. **Install dependencies**

   Each project has its own `requirements.txt` with pinned versions:

   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**

   Create a `.env` file in the project directory:

   ```env
   GOOGLE_API_KEY=your_api_key_here
   TAVILY_API_KEY=your_api_key_here  # Optional, for enhanced search
   ```

   > **Confidential Mode**: Several projects support 100% local operation using Ollama — no API keys required!

---

## 📊 Projects at a Glance

| Project | Type | Tech Stack | Highlights |
|---------|------|------------|------------|
| **Advanced_Strategic_Finance_Agent** | Multi-Agent | LangGraph, Gemini, Tavily | CFO-level forex analysis, 10 unit tests |
| **Multimodal_PDF_RAG** | RAG + VLM | Gemini 2.5 Pro, FAISS, Unstructured | Image/table understanding, CV project |
| **Text_Pdf_QA** | Advanced RAG | LangChain, Qwen3 Embeddings | Multiple retrievers, local mode |
| **youtube_QA_bot** | Analysis | Gemini, YouTube API | Timestamp citations, transcript analysis |
| **currency_converter_bot** | Agent | ReAct, Tavily, DDG | Real-time FX, dual search fallback |
| **chatbot_with_memory** | Chat | LangGraph, FAISS | Persistent conversations |
| **chatbot_without_memory** | Chat | Gemini/Ollama | Stateless, local/cloud modes |

---

## 🌟 Featured Project: Multimodal PDF RAG

**Why it stands out:** This is my **flagship CV project** demonstrating production-grade RAG with true multimodal understanding.

**Key differentiators:**
- ✅ **Image & Table Understanding**: Uses VLMs to caption visual content
- ✅ **100% Local Mode**: Confidential processing with Gemma3 via Ollama
- ✅ **Page-Referenced Answers**: Citations for verification
- ✅ **Image Extraction UI**: Download charts/tables with one click
- ✅ **Smart Caching**: Sub-2-second responses after initial processing

**Tech Stack:** LangChain, Gemini 2.5 Pro, Qwen3 Embeddings, FAISS, Streamlit, Unstructured, Ollama

[View Full Documentation →](Advanced_apps/Multimodal_PDF_RAG/README.md)

---

## 🏗️ Architecture Patterns

This repository demonstrates several advanced AI engineering patterns:

### 1. **Multi-Agent Orchestration (LangGraph)**
Specialized agents coordinate via stateful graphs:
```
Market Research → News Intel → CFO Strategist → Executive Report
```

### 2. **Multimodal RAG Pipeline**
```
PDF → Unstructured Loader → Image Captioning (VLM) → 
Content Injection → Qwen3 Embeddings → FAISS → Retrieval → Generation
```

### 3. **ReAct Agent Pattern**
```
Reason → Act (Search) → Observe → Reason → Answer
```

### 4. **Privacy-First Design**
- Local embeddings (Qwen3 via Hugging Face)
- Ollama for offline inference (Gemma3, Llama3.1)
- Zero data leakage in confidential mode

---

## 🛠️ Tech Stack Overview

| Category | Technologies |
|----------|-------------|
| **Frameworks** | LangChain, LangGraph, LangSmith |
| **LLMs** | Gemini 2.5 Pro, Gemini 3.1 Flash Lite, Gemma3, Llama3.1 |
| **Embeddings** | Qwen/Qwen3-Embedding-0.6B, Hugging Face |
| **Vector Stores** | FAISS |
| **RAG Retrievers** | MultiQuery, ContextualCompression, VectorStore |
| **Search** | Tavily AI, DuckDuckGo |
| **UI** | Streamlit |
| **Local Inference** | Ollama |
| **Testing** | pytest, unittest.mock |

---

## 🎯 Why This Repository?

1. **Ahead of the Curve**: Started in **June 2025**, before the agents boom
2. **Learning Journey**: Clear progression from basics to production systems
3. **Production-Ready**: Advanced projects feature clean code, tests, documentation
4. **Privacy-First**: All projects support local/offline modes
5. **Multimodal**: Goes beyond text — understands images, tables, figures
6. **Well-Documented**: Professional READMEs with architecture diagrams
7. **Modular Design**: Each project is self-contained and reproducible

---

## 📖 Learning Path

### **🌱 Beginner** (Fundamentals)
Start with simple-apps to understand core concepts:
1. `chatbot_without_memory` → Basic LangGraph patterns, stateless interactions
2. `chatbot_with_memory` → Stateful conversations, memory management
3. `currency_converter_bot` → ReAct agent pattern, tool usage

### **📚 Intermediate** (Building Complexity)
Move to basic RAG and retrieval:
4. `Text_pdf_RAG` → Basic RAG pipeline, vector search
5. `youtube_QA_bot` → Transcript analysis, timestamp citations

### **🚀 Advanced** (Production-Grade)
Master complex systems in Advanced_apps:
6. `Text_Pdf_QA` → Multiple retrievers, contextual compression, local models
7. `Advanced_Strategic_Finance_Agent` → Multi-agent orchestration with LangGraph, unit testing
8. `Multimodal_PDF_RAG` → ⭐ **CV Project** - Full-stack RAG with VLM, dual-mode architecture

This progression shows how I evolved from basic chatbots to sophisticated multi-agent systems.

---

## 🧪 Testing

Projects with unit tests include comprehensive test suites:

```bash
cd Advanced_apps/Advanced_Strategic_Finance_Agent
pytest test_agent.py -v
```

**Test Coverage:**
- ✅ Agent state schemas
- ✅ Tool functions (mocked APIs)
- ✅ Graph structure and flow
- ✅ Full integration tests

---

## 🔐 Confidentiality & Ethics

Several projects feature **Confidential Mode** for sensitive documents:
- 100% local processing with Ollama
- No data leaves your machine
- Ideal for legal, financial, or proprietary documents

All API keys should be stored in `.env` files (gitignored) and never committed.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share your own agent implementations

---

## 📧 Contact & Connect

- **GitHub**: [@Ajeesh25353646](https://github.com/Ajeesh25353646)
- **Portfolio**: [Add your portfolio link]
- **LinkedIn**: [Add your LinkedIn]

For questions about specific projects, check their individual README files or open an issue.

---

## 🌟 Show Your Support

If you find these projects helpful, consider:
- ⭐ Starring the repository
- 📢 Sharing with your network
- 🤝 Contributing improvements

---

**Built with ❤️ using LangChain, LangGraph, and cutting-edge AI orchestration**



