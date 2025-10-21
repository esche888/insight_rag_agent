# AI Agent leveraging a RAG Layer

## Overview
The app has a chatbot interface (based on Streamlit) which is backed by a LangGraph based agent.
Based on a query entered by the user the agent involves a RAG layer to prompt an LLM to get the answer to the query.
The RAG layer is loaded with data from the data/ directory which comprises a CSV with mock sales data 
as well as 4 PDFs about various subjects.

For testing the performance of the agent, a evaluation module runs a set of test cases which cover queries about the sales data
as well as the data in the PDFs.

The UI allows you to choose between three different LLMs (gemini-2.5-flash, gpt-3.5-turbo, gemma3:12b - to run locally on your laptop using Ollama).
In the side pannel there is also a dropdown for precanned questions.

## Primary modules
* rag_setup.py: implementation of the RAG layer loading
* rag_agent.py: Implementation of agent functionality
* rag_app.py: Streamlit UI implementation
* rag_eval.py: Execution of test cases

## Setup instructions
1. Clone repository
2. Set up ven environment:  # python -m venv venv
3. Ensure that API Keys are defined in environment (OPENAI_API_KEY, GOOGLE_API_KEY)
4. Define default LLM to be used for RAG loading and inference: Set MODEL_DEFAULT in .env
5. Load RAG layer: Run rag_setup.py
6. Test agent: Run rag_eval.py
7. Run agent: Run rag_app.py as a streamlit application (# streamlit run rag_app.py)
