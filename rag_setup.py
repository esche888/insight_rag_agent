"""
rag_set.py
Setting up Streamlit UI to access LangGraph chain for querying RAG layer
"""

import os
import sys
import logging 
import shutil

import pdfplumber
import pandas as pd
from dotenv import load_dotenv
import logging
import hashlib
import json

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document  
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from logging_config import configure_logging

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Supported LLMs
MODEL_GEMMA  = "gemma3:12b"
MODEL_GEMINI = "gemini-2.5-flash"  
MODEL_GPT35  = "gpt-3.5-turbo"

# Load environment variables from .env
load_dotenv(verbose=True, override=True)
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
MODEL_DEFAULT = os.getenv("MODEL_DEFAULT", MODEL_GPT35)
FAISS_INDEX_PATH_BASE = os.getenv("FAISS_INDEX_PATH", "insight_index")
CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 150))
NUM_RETURNED_DOCS = int(os.getenv("NUM_RETURNED_DOCS", 50))
SALES_CSV_PATH = os.getenv("SALES_CSV_PATH", "data/sales_data.csv")
PDFS_STRING = os.getenv("PDFS", ["UNDEFINED!!"])
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0))
EVAL_TEST_DATA_FILE_PATH = os.getenv("EVAL_TEST_DATA_FILE_PATH", "UNDEFINED!!")
RAG_CHAIN_TYPE = os.getenv("RAG_CHAIN_TYPE", "stuff")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Document types
DOC_TYPE_PDF="pdf"
DOC_TYPE_SALES_SUMMARY="sales_summary"
LOAD_PDFS = True
LOAD_SALES_DATA=True
RECREATE_VECTORSTORE=True   # If true, then deletes existing vectorstore; otherwise update existing one

# Check if gen model provided as first parameter is valid
if len(sys.argv) > 1:
    model = sys.argv[1]
    if model != MODEL_GEMMA and model != MODEL_GEMINI and model != MODEL_GPT35:
        logger.fatal(f"🛑 FATAL: Invalid model specified as parameter: {model}")
        sys.exit(2)
else:
    model = MODEL_DEFAULT
logger.debug(f"Model for loading RAG: {model}")

# Check whether API key is valid
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     sys.exit(
#         "🛑 FATAL: The GEMINI_API_KEY environment variable is not set. "
#         "Please define it in your shell environment or a .env file."
#     )
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Vectorstore
FAISS_INDEX_PATH = f"{FAISS_INDEX_PATH_BASE}_{model}"
logger.debug(f"Vectorstore directory for {model}: {FAISS_INDEX_PATH}")
MANIFEST_PATH = os.path.join(FAISS_INDEX_PATH, ".faiss_manifest.txt")
logger.debug(f"Manifest: {MANIFEST_PATH}")
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_PATH, "index.faiss")
logger.debug(f"Manifest file for {model}: {FAISS_INDEX_PATH}")

# PDFs from .env
try:
    PDFS = json.loads(PDFS_STRING)
except json.JSONDecodeError as e:
    logger.error(f"🛑 ERROR: Failed parsing JSON for PDFs: {e}")
    PDFS = [] 
logger.debug(f"PDFs: {PDFS}")


#------------------------FOR-RAG-CREATION--------------------------------

def generate_doc_id(content: str, metadata: dict) -> str:
    """ Generates a stable, unique ID for a document chunk using SHA256 hash """

    # Combine content and key metadata fields for a stable identifier
    key_data = f"{content}::{metadata.get('source')}::{metadata.get('row_id')}::{metadata.get('summary_id')}::{metadata.get('type')}"
    return hashlib.sha256(key_data.encode('utf-8')).hexdigest()


def text_from_pdf(path: str) -> str:
    """ Extract raw text from all pages of a PDF """

    logger.debug(f"Extracting text from PDF: {path}")
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
    except Exception as e:
        logger.error(f"🛑 ERROR: Failed extracting PDF text from {path}: {e}")
        return ""
    return "\n".join(texts)


def create_sales_sumry(df: pd.DataFrame, csv_path: str) -> list[dict]:
    """ Performs aggregations on the DataFrame and returns a list of formatted summary strings """

    # Aggregates sales data to summary sentences
    summaries = []
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.to_period("M")

    # Create summaries about sum of sales and mean customer satisfaction for month and product
    if all(col in df.columns for col in ["Month", "Product", "Sales", "Customer_Satisfaction"]):
        logger.info(f"- Summaries about sum of sales and mean customer satisfaction for month and product")
        monthly_product = df.groupby(["Month", "Product"]).agg({"Sales": "sum", "Customer_Satisfaction": "mean"}).reset_index()
        def row_to_doc_monthly_product(row):
            year = row['Month'].year
            month = row['Month'].month
            summary_text = f"Monthly Product Summary: In month {month} of year {year}, total sales for {row['Product']} were ${row['Sales']:.2f}, with an average customer satisfaction score of {row['Customer_Satisfaction']:.1f}."
            metadata = {
                "source": csv_path,
                "type": "sales_summary",
                "year": year,
                "month": month,
                "product": row['Product'],
                "total_sales": row['Sales'],
                "avg_satisfaction": row['Customer_Satisfaction']
            }
            document = Document(page_content=summary_text, metadata=metadata)
            logger.debug(document)
            return document
        summaries.extend([row_to_doc_monthly_product(r) for _, r in monthly_product.iterrows()])

    # Create summaries about mean sales and mean customer satisfaction for month and region
    if all(col in df.columns for col in ["Month", "Region", "Sales", "Customer_Satisfaction"]):
        logger.info(f"- Summaries about mean sales and mean customer satisfaction for month and region")
        monthly_region = df.groupby(["Month", "Region"]).agg({"Sales": "sum", "Customer_Satisfaction": "mean"}).reset_index()
        def row_to_doc_monthly_region(row):
            year = row['Month'].year
            month = row['Month'].month
            summary_text = f"Monthly Region Summary: In month {month} of year {year}, total sales for region {row['Region']} were ${row['Sales']:.2f}, with an average customer satisfaction score of {row['Customer_Satisfaction']:.1f}."
            metadata = {
                "source": csv_path,
                "type": "sales_summary",
                "year": year,
                "month": month,
                "region": row['Region'],
                "total_sales": row['Sales'],
                "avg_satisfaction": row['Customer_Satisfaction']
            }
            document = Document(page_content=summary_text, metadata=metadata)
            logger.debug(document)
            return document
        summaries.extend([row_to_doc_monthly_region(r) for _, r in monthly_region.iterrows()])

    # Create summaries about mean sales, mean customer satisfaction and mean customer age for customer gender
    if all(col in df.columns for col in ["Customer_Gender", "Sales", "Customer_Satisfaction", "Customer_Age"]):
        logger.info(f"- Summaries about mean sales, mean customer satisfaction and mean customer age for customer gender")
        demographics = df.groupby(["Customer_Gender"]).agg({"Sales": "mean", "Customer_Satisfaction": "mean", "Customer_Age": "mean"}).reset_index()
        def row_to_doc_customer_gender(row):
            summary_text = f"Demographic Summary: Average sales for customer gender {row['Customer_Gender']} were ${row['Sales']:.2f}, with an average customer satisfaction score of {row['Customer_Satisfaction']:.1f} and an average age of {row['Customer_Age']:.1f}."
            metadata = {
                "source": csv_path,
                "type": "sales_summary",
                "total_sales": row['Sales'],
                "gender": row['Customer_Gender'],
                "age": row['Customer_Age'],
                "avg_satisfaction": row['Customer_Satisfaction']
            }
            document = Document(page_content=summary_text, metadata=metadata)
            logger.debug(document)
            return document
        summaries.extend([row_to_doc_customer_gender(r) for _, r in demographics.iterrows()])
        
    logger.info(f"Created {len(summaries)} sales summary documents")
    return summaries


def load_chunk_data(pdf_paths: list[str], csv_path: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """ Loads new data from PDFs and CSV rows, chunks it, generates stable IDs, 
        and returns a list of LangChain Document objects ready for embedding. """
    
    # Process PDFs by extracting the text and adding each document to the raw documents list
    raw_documents = [] 
    if LOAD_PDFS: 
        logger.info("Loading and chunking data from PDF documents")
        if pdf_paths:
            for pdf_path in pdf_paths:
                logger.info(f"- {pdf_path}")
                text = text_from_pdf(pdf_path)
                if text:
                    raw_documents.append(Document(page_content=text, metadata={"source": pdf_path, "type": "pdf"}))

     # Read in the sales data from the CSV
    if LOAD_SALES_DATA: 
        logger.info(f"Loading and chunking summaries of sales data from CSV file {csv_path}")
        try:
            new_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            logger.error(f"🛑 ERROR: Sales data file not found: {csv_path}")
            raise

        # Process Aggregated Summaries by appending them to the list of all raw deocuments
        summary_docs = create_sales_sumry(new_df, csv_path)
        for i, summary_doc in enumerate(summary_docs):
            raw_documents.append(summary_doc)
    
    # Chunking & ID Assignment 
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    temp_documents = splitter.split_documents(raw_documents)
    
    # Assign a stable ID to each FINAL chunk based on its content and source metadata
    final_documents = []
    for doc in temp_documents:
        # Generate ID based on the chunk's content and its metadata
        doc.metadata['stable_id'] = generate_doc_id(doc.page_content, doc.metadata)
        final_documents.append(doc)

    logger.info(f"Total unique content chunks created: {len(final_documents)}")
    return final_documents


def remove_directory(path: str):
    """ Removes directory """

    # Check whether the directory exists 
    if os.path.exists(path):
        try:
            # Attempt to remove the directory
            shutil.rmtree(path)
            logging.info(f"Deleted directory [{path}]")
        except FileNotFoundError:
            logging.warning(f"Directory to be deleted not found: {path}")
        except OSError as e:
            # This specifically catches the 'Directory not empty' error
            logging.error(f"🛑 ERROR: Failed to delete directory '{path}': {e}")
            raise
    else:
        logging.info(f"Directory {path} doesn't exist and can therefore not be deleted")


def update_vectorstore(model: str, new_data_documents: list[Document]):
    """ Updates the existing FAISS index with new documents or creates a new index 
        if one does not exist, while ensuring deduplication via manifest. """
    
    # Delete vectorstore directory if it exists and is requested to be deleted
    if RECREATE_VECTORSTORE:
        remove_directory(FAISS_INDEX_PATH)

    # Initialize Embedding Model
    logger.info("Processing material to load into vectorstore")
    embeddings = create_embeds_genrtr(model)
      
    # Load manifest of already processed IDs
    processed_ids = set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            processed_ids = {line.strip() for line in f}
        logger.info(f"[DEDUPLICATION] Loaded {len(processed_ids)} previously processed IDs")
    
    # Filter new documents
    documents_to_add = []
    new_ids_to_record = []
    for doc in new_data_documents:
        stable_id = doc.metadata.get('stable_id')
        if stable_id and stable_id not in processed_ids:
            documents_to_add.append(doc)
            new_ids_to_record.append(stable_id)
            processed_ids.add(stable_id) 
        else:
            logger.debug(f"Skipping duplicate document with ID: {stable_id}")
    logger.info(f"Found {len(new_data_documents) - len(documents_to_add)} duplicates/processed chunks")
    
    # Check whether any documents to be updated
    if not documents_to_add:
        logger.warning("No unique new documents found to add.")
        return None # No update needed

    # Load index
    if os.path.exists(FAISS_INDEX_FILE):
        # Update existing index
        logger.info(f"Loading existing FAISS index for {model} from: {FAISS_INDEX_PATH}")
        vectorstore = load_vectorstore(model)
        logger.info(f"Adding {len(documents_to_add)} unique document chunks for {model} ...")
        vectorstore.add_documents(documents_to_add)
        logger.info(f"New unique documents added for {model}.")

    else:
        # Create and load new index
        logger.info(f"Index not found for {model}. Creating a new FAISS index.")
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True) 
        vectorstore = FAISS.from_documents(documents_to_add, embedding=embeddings)
        logger.info(f"New FAISS index created for [{model}]")

    # Save the Index and Update Manifest
    vectorstore.save_local(FAISS_INDEX_PATH)
    
    # Append the new IDs to the manifest file
    with open(MANIFEST_PATH, 'a') as f:
        for stable_id in new_ids_to_record:
            f.write(f"{stable_id}\n")
    
    logger.info(f"Saved updated FAISS index and manifest to [{FAISS_INDEX_PATH}] for {model}")
    return vectorstore


def get_chunk_count_faiss(model: str):
    """ Returns the total number of indexed vectors in the FAISS store """

    if not os.path.exists(FAISS_INDEX_PATH):
        # Index directory doesn't exist yet
        return 0
    try:
        # Get reference to vectorstore to get number of chunks in it
        vectorstore = load_vectorstore(model)
        return vectorstore.index.ntotal
    except Exception as e:
        logger.error(f"🛑 ERROR: Failed loading FAISS for {model} to get count: {e}")
        return 0


def test_retrieval_only(model: str, test_query: str, k: int = 3):
    """ Loads the existing FAISS index and runs a raw retrieval query 
        to see the context returned, without involving the LLM """
    
    logger.info(f"--- STARTING RAG RETRIEVAL TEST (NO LLM) for {model} ---")
    if not os.path.exists(FAISS_INDEX_PATH):
        logger.error(f"🛑 ERROR: FAISS index for {model} not found. Please run indexing first.")
        return

    try:
        # Create the retriever 
        vectorstore = load_vectorstore(model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        # Run the retrieval query
        retrieved_docs = retriever.invoke(test_query)
        logger.debug(f"Query: {test_query}")
        logger.debug(f"Retrieved {len(retrieved_docs)} relevant documents (k={k}) from {model}.")
        
        logger.debug("--- RETRIEVAL RESULTS (Context Chunks) from {model} ---")
        for i, doc in enumerate(retrieved_docs):
            logger.debug(f"--- Document Chunk {i+1} ---")
            logger.debug(f"Content (First 200 chars): {doc.page_content[:200]}...")
            logger.debug(f"Metadata: {doc.metadata}")
            logger.debug("-" * 20)

    except Exception as e:
        logger.error(f"🛑 ERROR: Error during retrieval test from {model}: {e}")
        raise

    return retrieved_docs


def create_embeds_genrtr(model: str):
    """ Creates embeddings generator corresponding to the LLM to be used for creating embeddings """

    embedding_model = None  
    if model == MODEL_GEMMA:
        embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
    elif model == MODEL_GEMINI:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    elif model == MODEL_GPT35:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small") 
    else:
        logger.error(f"🛑 ERROR: Unknown model {model}")
    return embedding_model


#------------------------FOR-RETRIEVAL--------------------------------

def load_vectorstore(model: str):
    """ Initialize vectorstore """

    FAISS_INDEX_PATH = f"{FAISS_INDEX_PATH_BASE}_{model}"
    logger.debug(f"Load vectorstore for {model} from directory [{FAISS_INDEX_PATH}]")
    embeddings = create_embeds_genrtr(model)
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def create_llm(model: str, temperature):
    """ Create LLM client for loading RAG or retrieval from RAG """

    llm = None  
    if model == MODEL_GEMMA:
        llm = OllamaLLM(model=model, temperature=temperature, keep_alive='10m')
    elif model == MODEL_GEMINI:
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
    elif model == MODEL_GPT35:
        llm = ChatOpenAI(model=model, temperature=temperature)
    else:
        logger.error(f"🛑 ERROR: Model undefined or unsupported: {model}")
        raise ValueError(f"Not able to create LLM client for {model}: {e}")
    return llm


def create_rag_chain(model, temperature):
    """ Create the rag chain for enabling retrieval from vectorstore """

    logger.info("-" * 80)
    logger.info(f"Create RAG chain for [{model}] and temperature {temperature}")
    vectorstore = load_vectorstore(model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": NUM_RETURNED_DOCS})
    llm = create_llm(model, temperature)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type=RAG_CHAIN_TYPE,
        return_source_documents=True
    )
    return rag_chain


# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
    # Load and chunk PDFs and data in sales CSV
    logger.info(f"Starting RAG indexing process for [{model}]")
    new_documents = load_chunk_data(PDFS, SALES_CSV_PATH, CHUNK_SIZE, CHUNK_OVERLAP)

    # Check current chunk count before running the update
    # Delete vectorstore directory if it exists and is requested to be deleted
    if RECREATE_VECTORSTORE:
        remove_directory(FAISS_INDEX_PATH)
    initial_chunks = get_chunk_count_faiss(model)
    logger.info(f"Chunks before update: {initial_chunks}")

    if not new_documents:
        logger.info("*** NO NEW DOCUMENTS WERE LOADED")
    else:
        # Create/update index
        final_vectorstore = update_vectorstore(model, new_documents)

        # Inspection of index
        total_chunks = get_chunk_count_faiss(model)
        chunks_added = total_chunks - initial_chunks
        logger.info(f"Total chunks in FAISS for {model}: {total_chunks} ({chunks_added} added)")

        if False:  # Excluding this test for now
            # Run RAG query to test things
            rag_chain = create_rag_chain(model, 0)
            query = "Based on the summary data, what was the average sales amount by gender, and what was the average customer satisfaction score?"
            logger.info(f"\nQUERY (Full RAG) using {model}: {query}")
            
            try:
                result = rag_chain.invoke({"query": query})
                logger.info(f"\n--- FULL RAG RESULT (with {model}) ---")
                logger.info(result['result'])
            except Exception as e:
                logger.error(f"🛑 ERROR: Failure running RAG chain with {model}: {e}")

            # Run retrieval test only (without LLM) 
            test_query_string = "What were the average sales and satisfaction scores by customer gender?"
            test_retrieval_only(model, test_query_string, k=20)

        logger.info(f"Indexing process for vectorstore for [{model}] finished")



