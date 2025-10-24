# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG-based insight agent with a Streamlit chatbot interface backed by a LangGraph workflow. The agent processes queries by routing them through a RAG layer loaded with sales data (CSV) and PDF documents. It analyzes query types and routes to appropriate processing nodes (PDF analysis vs. sales data analysis with charts and recommendations).

## Essential Commands

### Initial Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Core Workflow
```bash
# Build/rebuild the FAISS vectorstore from PDFs and CSV
python rag_setup.py

# Optionally specify a model (defaults to MODEL_DEFAULT from .env)
python rag_setup.py gpt-3.5-turbo
python rag_setup.py gemini-2.5-flash
python rag_setup.py claude-3-5-sonnet-20240620
python rag_setup.py gemma3:12b

# Run evaluation test suite
python rag_eval.py

# Launch Streamlit UI
streamlit run rag_app.py
```

## Architecture

### Three-Layer System

1. **RAG Setup Layer** (`rag_setup.py`):
   - Loads PDFs from `data/` directory and extracts text using pdfplumber
   - Reads `data/sales_data.csv` and creates aggregated summary documents
   - Chunks documents using `RecursiveCharacterTextSplitter`
   - Generates stable document IDs via SHA256 hash of content + metadata
   - Maintains `.faiss_manifest.txt` for deduplication across runs
   - Creates model-specific FAISS indices: `insight_index_{model_name}/`
   - Maps models to embedding providers: Ollama (gemma3), Google (gemini), OpenAI (gpt-3.5, claude)

2. **Agent Layer** (`rag_agent.py`):
   - LangGraph `StateGraph` with typed `State` dictionary
   - **Routing logic**: `type_analysis_node` calls `test_retrieval_only()` to count document types in top-k retrieval results, then routes based on whether more PDFs or sales summaries are returned
   - **PDF path**: `pdf_type_node` → END
   - **Sales path**: `sales_analysis_node` → `chart_node` → `recommend_node` → END
   - `chart_node` expects LLM to return JSON with:
     - `chart_type`: One of "line chart", "bar chart", "stacked bar chart", "scatter plot", "area chart"
     - `chart_data`: List of dictionaries with data points
     - `chart_title`: Descriptive title for the chart (optional)
     - `x_axis_label`: Label for X-axis (optional)
     - `y_axis_label`: Label for Y-axis (optional)
     - `summary`: Brief insight description (max 12 words)
   - Comprehensive data validation ensures chart_data is valid before rendering
   - Uses `MemorySaver` checkpointer for conversation persistence by thread_id

3. **UI Layer** (`rag_app.py`):
   - Streamlit interface with chat history
   - Model selector dropdown dynamically populated based on:
     - API key availability (OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY)
     - Vectorstore existence (must run `python rag_setup.py <model>` first)
   - Available models:
     - gpt-3.5-turbo (requires OPENAI_API_KEY + vectorstore)
     - gemini-2.5-flash (requires GOOGLE_API_KEY + vectorstore)
     - claude-3-5-sonnet-20240620 (requires ANTHROPIC_API_KEY + vectorstore)
     - gemma3:12b (Ollama local, requires vectorstore)
   - Precanned questions dropdown in sidebar
   - Stores `rag_chain` in `st.session_state` and recreates when model changes
   - **Enhanced Chart Rendering**:
     - Supports 5 chart types: line, bar, stacked bar, scatter, area
     - Dynamic field detection based on data structure
     - Custom axis labels and titles from LLM metadata
     - Intelligent tooltip generation based on available fields
     - Adaptive color encoding (product → region → single color)
     - Robust error handling with user-friendly warnings
     - Responsive sizing (600x400 with container width)

### Key Modules

- `utils/prompt_loader.py`: LRU-cached prompt loading from files or env vars (supports `PROMPT_NAME` or `PROMPT_NAME_PATH` patterns). Call `load_prompt.cache_clear()` if changing prompts during development.
- `utils/line_loader.py`: Loads multi-line data (used for comprehensive sales summary)
- `utils/clean_json.py`: Strips markdown code fences from LLM JSON responses
- `utils/chart_utils.py`: Chart validation, field inference, and data formatting utilities
  - `validate_chart_data()`: Validates structure and content of chart data
  - `infer_chart_fields()`: Automatically determines x, y, and color fields
  - `suggest_chart_type()`: Recommends chart type based on data characteristics
  - `format_chart_data()`: Cleans and normalizes chart data
- `logging_config.py`: Centralized logging configuration
- `prompts/`: Prompt templates (PDF_TYPE_PROMPT, SALES_DATA_ANALYSIS_PROMPT, CHART_PROMPT, RECOMMENDATION_PROMPT, EVAL_PROMPT)

### Document Metadata System

Documents have `metadata['type']` set to:
- `"pdf"`: Content from PDF files
- `"sales_summary"`: Aggregated sales data with fields like `month`, `product`, `region`, `total_sales`, `avg_satisfaction`, `gender`, `age`

Sales summaries are created in three aggregation groups:
1. Monthly product summaries (month + product → total sales + avg satisfaction)
2. Monthly region summaries (month + region → total sales + avg satisfaction)
3. Gender demographics (gender → avg sales + avg satisfaction + avg age)

## Environment Configuration

Required `.env` variables:
- `OPENAI_API_KEY`: Required for using gpt-3.5-turbo (default model)
- `GOOGLE_API_KEY`: Required for using gemini-2.5-flash
- `ANTHROPIC_API_KEY`: Required for using claude-3-5-sonnet-20240620
- `MODEL_DEFAULT`: Default LLM (default: "gpt-3.5-turbo", also supports "gemini-2.5-flash", "claude-3-5-sonnet-20240620", "gemma3:12b")
  - Note: The UI dropdown will only show models for which API keys are available
  - Ollama models (gemma3:12b) don't require API keys and are always available
- `MODEL_TEMPERATURE`: Temperature for LLM inference (default: 0)
- `FAISS_INDEX_PATH`: Base path for vectorstore (default: "insight_index")
- `CHUNK_SIZE`: Text chunk size (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 150)
- `NUM_RETURNED_DOCS`: Number of docs retrieved from RAG (default: 50)
- `SALES_CSV_PATH`: Path to sales CSV (default: "data/sales_data.csv")
- `PDFS`: JSON array of PDF paths (e.g., `["data/AI_business_model_innovation.pdf", ...]`)
- `EVAL_TEST_DATA_FILE_PATH`: Path to evaluation test cases
- `RAG_CHAIN_TYPE`: LangChain chain type (default: "stuff")
- `OLLAMA_EMBED_MODEL`: Ollama embedding model (default: "mxbai-embed-large")

Prompt configuration (use either direct value or file path):
- `PDF_TYPE_PROMPT` or `PDF_TYPE_PROMPT_PATH`
- `SALES_DATA_ANALYSIS_PROMPT` or `SALES_DATA_ANALYSIS_PROMPT_PATH`
- `CHART_PROMPT` or `CHART_PROMPT_PATH`
- `RECOMMENDATION_PROMPT` or `RECOMMENDATION_PROMPT_PATH`
- `EVAL_PROMPT` or `EVAL_PROMPT_PATH`

## Critical Implementation Details

### Deduplication Strategy
- Each chunk gets `metadata['stable_id']` = SHA256 hash of content + source + row_id + summary_id + type
- Manifest file tracks processed IDs across runs
- `RECREATE_VECTORSTORE=True` (in `rag_setup.py`) deletes existing index before rebuild

### Model-Specific Indices
- FAISS index path constructed as: `{FAISS_INDEX_PATH_BASE}_{model}`
- Each model requires its own index because embeddings differ
- Changing models requires rebuilding the index or having pre-built indices

### LLM Response Formatting
- Agent prefixes queries with `[{model}, {temperature}] - {query}`
- `rag_eval.py` strips this prefix when evaluating test cases
- Chart node requires LLM to return valid JSON (uses `clean_json_string()` to remove markdown fences)

### Prompt Loading Pattern
- Prompts loaded via `load_prompt(key_name)` which checks env vars in order: `{key}_PATH`, then `{key}`
- Files can contain a `CUTOFF_MARKER` line to ignore trailing content
- LRU cache means prompt changes won't reload until cache cleared

## Testing and Evaluation

- Test cases in `eval_data/rag_eval_test_data.txt`
- Format: `query|expected_answer` (pipe-delimited)
- `rag_eval.py` uses LangChain's `QAEvalChain` for answer quality scoring
- Can test specific subsets: `rag_eval_test_data - sales_summary.txt`, `rag_eval_test_data - complete.txt`

## Common Patterns

### Adding a New LLM
1. Add model constant in `rag_setup.py` (e.g., `MODEL_NEW = "model-name"`)
2. Update `create_embeds_genrtr()` to map model to embedding provider
3. Update `create_llm()` to instantiate appropriate LLM client
4. Update command-line validation in `__main__` section
5. Add to `get_available_models()` in `rag_app.py` with appropriate API key check
6. Add to evaluation support in `rag_eval.py` `evaluate_predictions()`
7. Run `python rag_setup.py model-name` to build index

### Debugging RAG Retrieval
Use `test_retrieval_only(model, query, k)` to see raw retrieval results without LLM:
```python
from rag_setup import test_retrieval_only
docs = test_retrieval_only("gpt-3.5-turbo", "What are total sales?", k=10)
```

### Modifying Agent Flow
- Agent nodes in `rag_agent.py`: Update methods and `build_workflow()`
- Each node returns dict with state updates (partial state pattern)
- Conditional routing via `map_decision_to_node()` based on state values
- State type hints defined in `State(TypedDict)`

## Gotchas

- **Concurrent FAISS writes**: Don't run multiple `rag_setup.py` processes simultaneously pointing at same index path
- **Model availability requirements** (Both conditions must be met):
  1. **API key must be set**:
     - `OPENAI_API_KEY` for gpt-3.5-turbo
     - `GOOGLE_API_KEY` for gemini-2.5-flash
     - `ANTHROPIC_API_KEY` for claude-3-5-sonnet-20240620
  2. **Vectorstore must be built**: Run `python rag_setup.py <model>` to create the FAISS index
  - The UI dropdown only shows models that meet BOTH requirements
  - If no models are available, the UI shows a helpful error message with setup instructions
- **Claude embeddings**: Claude uses OpenAI embeddings (text-embedding-3-small) for RAG, so OPENAI_API_KEY is also needed when using Claude models
- **Prompt cache**: After editing prompt files, either restart the process or call `load_prompt.cache_clear()`
- **RECREATE_VECTORSTORE**: Set to `False` in `rag_setup.py` if you want incremental updates instead of full rebuild
- **Model in message prefix**: The UI and eval code expect messages formatted as `[model, temp] - content`
