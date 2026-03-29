# 🚀 LangSmith + LangGraph Playground

This repository contains a collection of experiments and implementations using \*\*LangSmith\*\* and \*\*LangGraph\*\* to understand LLM workflows, tracing, and agent-based systems.

---

## 📌 Overview

This repo demonstrates:

- 🔹 Basic LLM calls with tracing

- 🔹 Sequential chains

- 🔹 Retrieval-Augmented Generation (RAG)

- 🔹 Optimization of embeddings and latency

- 🔹 Agent creation and execution tracing

- 🔹 Integration of LangGraph with LangSmith

---

## 📂 Project Structure


├── 1\_simple\_llm\_call.py      # Basic LangSmith LLM call example

├── 2\_sequential\_chain.py     # Sequential chain execution

├── 3\_rag\_v1.py               # Initial RAG implementation (basic)

├── 3\_rag\_v2.py               # Improved RAG (fixed tracing issues)

├── 3\_rag\_v3.py               # Combined tracing (Python + non-Python)

├── 3\_rag\_v4.py               # Optimized RAG (embedding storage)

├── 4\_agent.py                # Agent creation + tracing

├── 5\_langgraph.py            # LangGraph workflow integration

├── requirements.txt          # Dependencies

└── .gitignore

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

\`\`\`bash

git clone https://github.com/PranshuSama/LangSmith.git

cd LangSmith

### **2️⃣ Create virtual environment**

python -m venv myenv

source myenv/bin/activate # Mac/Linux

myenv\\Scripts\\activate # Windows

### **3️⃣ Install dependencies**

pip install -r requirements.txt

**🔑 Environment Variables**
----------------------------

Create a .env file and add:

OPENAI\_API\_KEY=your\_openai\_api\_key

LANGCHAIN\_API\_KEY=your\_langsmith\_api\_key

LANGCHAIN\_TRACING\_V2=true

LANGCHAIN\_PROJECT=your\_project\_name

**🧠 What You’ll Learn**
------------------------

### **✅ LLM Basics**

*   How to make simple API calls
    
*   Track requests using LangSmith
    

### **✅ Chains**

*   Build sequential workflows
    
*   Debug intermediate outputs
    

### **✅ RAG (Retrieval-Augmented Generation)**

*   Document loading
    
*   Chunking strategies
    
*   Embeddings & vector storage
    
*   Performance optimization
    

### **✅ Agents**

*   Tool-based reasoning
    
*   Execution tracing in LangSmith
    

### **✅ LangGraph**

*   Graph-based workflows
    
*   Stateful LLM pipelines
    

**📊 Key Improvements Across Versions**
---------------------------------------

**Version**                                                   **Improvement**
                                                                     
RAG v1                                                         Initial implementation

RAG v2                                                         Fixed tracing issues

RAG v3                                                         Unified tracing

RAG v4                                                         Reduced latency using stored embeddings

**🔍 Observability with LangSmith**
-----------------------------------

All scripts are instrumented to:

*   📈 Track execution
    
*   🐞 Debug failures
    
*   🔗 Visualize chains & agents
    
*   🧪 Compare runs
    

**🚧 Future Improvements**
--------------------------

*   Add UI for RAG queries
    
*   Integrate vector databases (FAISS, Pinecone)
    
*   Multi-agent workflows
    
*   Deployment on cloud
