# Corrective_RAG 
Basic Understanding by implementing the corrective RAG using langgraph

# Corrective RAG with LangGraph 

**A step-by-step implementation of Corrective RAG (CRAG) using LangGraph, built up from a simple RAG pipeline to a self-correcting one.**

Standard RAG retrieves documents and passes them straight to the LLM, even when the retrieved context is irrelevant. Corrective RAG adds a feedback loop: it **evaluates** what was retrieved, **refines** the good parts, and **falls back to web search** when the knowledge base can't answer the question.

This repo walks through that idea in six small notebooks, each adding one capability on top of the last.

>  **Status:** In progress. Notebooks are being added and refined.

---

## Table of Contents

- [Why Corrective RAG?](#why-corrective-rag)
- [How It Works](#how-it-works)
- [Learning Path](#learning-path)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Notebooks](#running-the-notebooks)
- [Key Concepts](#key-concepts)
- [Roadmap](#roadmap)
- [References](#references)
- [Author](#author)

---

## Why Corrective RAG?

Basic RAG has a blind spot: **it trusts whatever the retriever returns.** That causes problems when:

- the knowledge base has no relevant document, so the LLM answers from noise or hallucinates
- retrieved chunks are only partly relevant and the useful part is buried
- the user's query is vague, so retrieval misses the right content
- the question needs fresh information the documents don't contain

Corrective RAG addresses these by grading retrieval quality and choosing a different path depending on the result.

## How It Works

```
                    ┌──────────────┐
   User question ─▶ │   Retrieve   │  (vector store over /documents)
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │   Evaluate   │  grade relevance of each document
                    └──────┬───────┘
         ┌─────────────────┼──────────────────┐
         ▼                 ▼                  ▼
     Correct           Ambiguous           Incorrect
  (relevant docs)    (partly relevant)   (nothing relevant)
         │                 │                  │
         ▼                 ▼                  ▼
   Refine docs       Refine docs +      Rewrite query
   (keep useful      web search         ─▶ web search
    parts only)           │                  │
         └─────────────────┴──────────────────┘
                           ▼
                    ┌──────────────┐
                    │   Generate   │  answer from the corrected context
                    └──────────────┘
```

<!-- VERIFY: adjust the branches above to match your final graph in step 6 -->

Each box is a **node** in a LangGraph state graph, and the three-way split is a **conditional edge** that routes on the evaluator's verdict.

## Learning Path

| Step | Notebook | What it adds |
|---|---|---|
| 1 | `step1_simple_RAG.ipynb` | Baseline RAG: load documents, embed, retrieve, generate. The reference point for everything after. |
| 2 | `step2_retrieval_refinement.ipynb` | Refine retrieved documents so only the relevant parts reach the LLM. |
| 3 | `step3__evaluator_rag3.ipynb` | An evaluator that grades retrieved documents for relevance and drives routing. |
| 4 | `step4_websearch_refinement_rag.ipynb` | Web search fallback when the knowledge base is insufficient. |
| 5 | `step5_rewrite_query.ipynb` | Query rewriting so retrieval and search work better on poorly phrased questions. |
| 6 | `step6_ambiguous_RAG.ipynb` | Handles the ambiguous case by combining refined internal documents with web results. |

Run them in order: each notebook builds on the idea from the previous one.

## Project Structure

```
Corrective_RAG/
├── documents/                          # Source documents for the knowledge base
├── step1_simple_RAG.ipynb              # Baseline RAG
├── step2_retrieval_refinement.ipynb    # Document refinement
├── step3__evaluator_rag3.ipynb         # Relevance evaluator
├── step4_websearch_refinement_rag.ipynb# Web search fallback
├── step5_rewrite_query.ipynb           # Query rewriting
├── step6_ambiguous_RAG.ipynb           # Ambiguous-case handling
└── README.md
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Orchestration | LangGraph, LangChain |
| LLM | <!-- VERIFY: e.g. Groq / OpenAI --> |
| Embeddings | <!-- VERIFY: e.g. HuggingFace / OpenAI --> |
| Vector store | <!-- VERIFY: e.g. FAISS / Chroma --> |
| Web search | <!-- VERIFY: e.g. Tavily --> |
| Environment | Jupyter Notebook |

## Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab (or VS Code with the Jupyter extension)
- An API key for your LLM provider
- A web search API key (needed from step 4 onward) <!-- VERIFY: e.g. Tavily: https://www.tavily.com/ -->

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SyedaNaziaGit/Corrective_RAG.git
cd Corrective_RAG
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langgraph langchain langchain-community python-dotenv jupyter
```

<!-- VERIFY: add your LLM, embedding, vector store, and search packages
     (e.g. langchain-groq, langchain-tavily, faiss-cpu), or add a requirements.txt
     and replace this block with: pip install -r requirements.txt -->

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
# Replace with the keys your notebooks actually use
GROQ_API_KEY=your_llm_api_key
TAVILY_API_KEY=your_search_api_key
```

>  Never commit your `.env` file. Add it to `.gitignore` before pushing.

### 5. Add your documents

Put the files you want to query in the `documents/` folder.

## Running the Notebooks

```bash
jupyter notebook
```

Open the notebooks in order, from `step1_simple_RAG.ipynb` to `step6_ambiguous_RAG.ipynb`, and run each cell top to bottom.

## Key Concepts

- **Retrieval-Augmented Generation (RAG):** grounding an LLM's answer in retrieved documents.
- **Retrieval evaluator:** a grader that scores whether retrieved documents actually answer the question.
- **Knowledge refinement:** filtering retrieved documents down to the parts that matter.
- **Web search fallback:** going outside the knowledge base when internal retrieval fails.
- **Query rewriting:** reformulating the question so retrieval and search return better results.
- **Conditional routing:** LangGraph edges that choose the next node based on state, here the evaluator's verdict.

## Roadmap

- [ ] Finish and clean up all six notebooks
- [ ] Combine the final graph into a single runnable Python script
- [ ] Add a `requirements.txt`
- [ ] Add evaluation of CRAG vs. baseline RAG on sample questions
- [ ] Simple Streamlit UI for asking questions

## References

- Yan et al., [*Corrective Retrieval Augmented Generation*](https://arxiv.org/abs/2401.15884) (CRAG paper)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)



---

⭐ If you find this useful, consider giving it a star.
