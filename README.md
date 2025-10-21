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
2. Set up ven environment:  
    > python -m venv venv
    > . venv/bin/activate
3. Ensure that API Keys are defined in environment (OPENAI_API_KEY, GOOGLE_API_KEY)
    > echo $GOOGLE_API_KEY    
    > echo $OPENAI_API_KEY 
4. Define default model to be used for RAG loading and inference: 
    - Set MODEL_DEFAULT in .env to the corresponding model
    - Like MODEL_DEFAULT="gpt-3.5-turbo"
5. Ensure you have the latest version of pip installed
    > python -m pip install --upgrade pip
5. Install python packages in virtual environment
    > pip install -r requirements.txt   # This can take a while
6. Load RAG layer: 
    > python rag_setup.py 
7. Test agent: 
    > python rag_eval.py
8. Run agent as a streamlit application 
    > streamlit run rag_app.py 


Still TBD:
* Make chats persistent
* Some issues installing all packages in requirements.txt in one go
* Some testcases still fail