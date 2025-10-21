"""
rag_agent.py
LangGraph workflow for generating insights from sales data and answer questions about PDFs loaded into RAG layer
"""

from typing import TypedDict, Annotated
import logging
import time
from operator import add
import uuid
import traceback
import json
import os
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
# from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.prompts import PromptTemplate

from utils.prompt_loader import load_prompt
from utils.line_loader import load_lines
from utils.clean_json import clean_json_string

from rag_setup import test_retrieval_only
from rag_setup import create_llm, DOC_TYPE_PDF, DOC_TYPE_SALES_SUMMARY

USE_CHAT_PERSISTANCE = False
QUESTION_TYPE_PDF = "PDF Type Question"
QUESTION_TYPE_SALES_SUM = "Sales Summary Type Question"
RESPONSE_CUT_OFF = 200
CONVERSATIONS_DIR_PATH="conversations"
CONVERSATION_FILENAME="conversation.txt"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Set up logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear prompt cache to have it reloaded for each execution
load_prompt.cache_clear()

# Get summary data (extracted from sales_data.csv and then stored in text file to avoid having to recreate it each time)
COMPREHENSIVE_SUMMARY_DATA = load_lines("COMPREHENSIVE_SUMMARY_DATA")

# Graph state definition
class State(TypedDict):
    model: str
    temperature: float
    question: str
    request_type_analysis_prompt: str
    request_type_analysis_response: str
    pdf_type_prompt: str
    pdf_type_response: str
    sales_data_analysis_prompt: str
    sales_data_analysis_response: str
    recommendation_prompt: str
    recommendation_response: str
    final_response: str
    response_time: float
    messages: Annotated[list[BaseMessage], add]  
    counter: int
    session_data: dict
    chart_data: str
    chart_type: str
    summary: str


class InsightAgent:
    """Agent for querying and generating insights from sales data and PDFs"""
    
    def __init__(self, rag_chain):
        """ Initialize the insights agent """

        self.rag_chain = rag_chain
        self.lg_chain = None
        self.thread_id = str(uuid.uuid4())
        

    def run_insight_chain(self, model: str, temperature: float, rag_chain, question: str) -> dict:
        """ Execute a query through the insight agent by initiating LG chain """

        logger.info("_" * 80)
        logger.info(f"Run Insight chain using [{model}] - \"{question}\"")
        try:
            # Prepare input for LG chain with HumanMessage
            human_message = f"[{model}, {temperature}] - {question}"
            input_data = {
                "model": model,
                "temperature": temperature,
                "question": question,
                "sales_data_analysis_prompt": "",
                "sales_data_analysis_response": "",
                "messages": [HumanMessage(content=human_message)],  
                "counter": 0,
                "start_time": 0.0,
                "end_time": 0.0,
                "chart_data": "",
                "chart_type": "",
                "summary": ""
            }
            config = {"configurable": {"thread_id": self.thread_id}}

            # Invoke the chain
            result = self.lg_chain.invoke(input_data, config)
            response = result.get("final_response", "No response generated")
            returnValue = {
                "chart_data": result["chart_data"],
                "chart_type": result["chart_type"],
                "summary": result["summary"],
                "text": response,
                "model": model,
                "temperature": temperature
            }
            logger.debug(f"[run_insight_chain] returnValue: {returnValue}")
            return returnValue
        
        except Exception as e:
            logger.error(f"🛑 ERROR: Running query using {model}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            response = f"Encountered error: {str(e)}"
            returnValue = {
                "chart_data": None,
                "chart_type": None,
                "summary": None,
                "text": response,
                "model": model,
                "temperature": temperature
            }
            logger.debug(f"[run_insight_chain] returnValue: {returnValue}")
            return returnValue


#------------------------LANGGRAPH-NODES--------------------------------


    def type_analysis_node(self, state: State) -> dict:
        """ Determine type of query to route it to either PDF type processing or sales data processing """

        # Execute node to determine whether it is a PDF type query or a sales data type query
        logger.info("")
        logger.info("______________________TYPE_ANALYSIS_NODE___________________________")
        try:
            # Run RAG-only query (without LLM involvement) to determine to determine the type of documents retrieved
            logger.info(f"Calling RAG to retrieve documents related to query")
            retrieved_docs = test_retrieval_only(state['model'], state['question'], 10)
            count_pdf_documents = 0
            count_sales_summary_documents = 0
            count_undefined = 0
            for i, doc in enumerate(retrieved_docs):
                if doc.metadata.get('type', 'N/A') == DOC_TYPE_PDF:
                    count_pdf_documents += 1
                elif doc.metadata.get('type', 'N/A') == DOC_TYPE_SALES_SUMMARY:
                    count_sales_summary_documents += 1
                else:
                    count_undefined += 1

            logger.debug(f"Total number of documents retrieved: {len(retrieved_docs)}")
            logger.debug(f"Total number of PDF type documents retrieved: {count_pdf_documents}")
            logger.debug(f"Total number of Sales Summary type documents retrieved: {count_sales_summary_documents}")
            if count_undefined > 0:
                logger.warning(f"Total number of Undefined type documents retrieved: {count_undefined}")

            # Analyze the type of documents received from RAG layer
            if count_pdf_documents > count_sales_summary_documents:
                request_type_analysis_response = QUESTION_TYPE_PDF
            else:
                request_type_analysis_response = QUESTION_TYPE_SALES_SUM

            # Return updates to state
            returnValue = {
                "request_type_analysis_response": request_type_analysis_response,
            }
            logger.debug(f"returnValue: {returnValue}")
            return returnValue

        except Exception as e:
            logger.error(f"🛑 ERROR: Not able to get request type analysis response from {state['model']} for prompt. Exception: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise


    def map_decision_to_node(self, state):
        """ Maps the decision to the correct node """

        returnValue = {
            QUESTION_TYPE_PDF: QUESTION_TYPE_PDF,
            QUESTION_TYPE_SALES_SUM: QUESTION_TYPE_SALES_SUM,
        }.get(state["request_type_analysis_response"], QUESTION_TYPE_SALES_SUM)
        logger.debug(f"returnValue = {returnValue}")
        return returnValue


    def pdf_type_node(self, state: State) -> dict:
        """ Processes PDF type queries """

        # Processing of PDF related query
        logger.info("")
        logger.info("______________________PDF_TYPE_NODE________________________________")
        try:
            # Build the prompt
            pdf_type_template_text = load_prompt("PDF_TYPE_PROMPT")
            logger.debug(f"pdf_type_template_text: \n{pdf_type_template_text}\n")
            pdf_type_prompt = PromptTemplate(
                input_variables=["question"],
                template=pdf_type_template_text)
            state["pdf_type_prompt"] = pdf_type_prompt.format(question=state['question'])

            # Invoke the RAG chain
            logger.info(f"Calling RAG chain to request a PDF TYPE QUESTION using [{state['model']}:{state['temperature']}]")
            state["start_time"] = time.time()
            result = self.rag_chain.invoke({"query": state["pdf_type_prompt"]})
            end_time = time.time()
            duration = end_time - state["start_time"]

            # Extract response
            pdf_type_response = result.get("result", "No result")
            response = pdf_type_response[:RESPONSE_CUT_OFF] + "..." if len(pdf_type_response) > RESPONSE_CUT_OFF else pdf_type_response
            logger.debug(f"Response to PDF type query: {response}") 

            # Log sources
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("---- SOURCES ----")
                for doc in result.get("source_documents", []):
                    logger.debug(doc.metadata)

            # Return updates to state
            content = (
                f"[{state['model']}, {state['temperature']}] - PDF RELATED \n\n"
                f"{pdf_type_response}\n\n"
                f"PDF type response time: {duration:.2f}s"
            )
            returnValue = {
                "messages": [AIMessage(content=content)], 
                "final_response": pdf_type_response,
                "counter": state.get("counter", 0) + 1,
                "response_time": f"{duration:.2f}",
                "chart_data": None,
                "chart_type": None
            }
            logger.debug(f"[pdf_type_node] returnValue: {returnValue}")
            return returnValue

        except Exception as e:
            logger.error(f"🛑 ERROR: Not able to get PDF type response from {state['model']} for prompt. Exception: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise


    def sales_analysis_node(self, state: State) -> dict:
        """ Query the RAG system for analysis of sales data """

        # Process sales data related query
        logger.info("")
        logger.info("______________________SALES_ANALYSIS_NODE__________________________")
        try:
            # Build the prompt
            sales_data_analysis_template_text = load_prompt("SALES_DATA_ANALYSIS_PROMPT")
            logger.debug(f"sales_data_analysis_template_text: \n{sales_data_analysis_template_text}\n")
            sales_data_analysis_prompt = PromptTemplate(
                input_variables=["question"],
                template=sales_data_analysis_template_text)
            state["sales_data_analysis_prompt"] = sales_data_analysis_prompt.format(question=state['question'])
            logger.debug(f"sales_data_analysis_prompt: \n{json.dumps(vars(sales_data_analysis_prompt), indent=2, default=str)}\n")

            # Invoke the RAG chain
            logger.debug(f"Calling RAG chain to request an SALES DATA ANALYSIS using [{state['model']}:{state['temperature']}]")
            state["start_time"] = time.time()
            logger.info(f"Calling RAG chain to request a SALES ANALYSIS using {state['model']} with temperature {state['temperature']}")
            result = self.rag_chain.invoke({"query": state["sales_data_analysis_prompt"]})
            end_time = time.time()
            duration = end_time - state["start_time"]

            # Extract response
            sales_data_analysis_response = result.get("result", "No result")
            response = sales_data_analysis_response[:RESPONSE_CUT_OFF] + "..." if len(sales_data_analysis_response) > RESPONSE_CUT_OFF else sales_data_analysis_response
            logger.debug(f"Business analysis response: {response}") 

            # Optional: log sources
            with_metadata = False
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("---- SOURCES FOR SALES DATA ANALYSIS QUERY ----")
                for i, doc in enumerate(result.get("source_documents", [])):
                    if with_metadata:  
                        log_msg = f"[{i}] {doc.page_content} | {doc.metadata}"
                    else:
                        log_msg = f"[{i}] {doc.page_content}"
                    logger.debug(log_msg)
            logger.debug("---- show_retrieved_records ----")
            self.print_retrieved_docs(state)

            # Return updates to state
            returnValue = {
                "sales_data_analysis_response": sales_data_analysis_response,
                "counter": state.get("counter", 0) + 1,
                "response_time": f"{duration:.2f}"
            }
            logger.debug(f"[sales_data_analysis_node] returnValue: {returnValue}")
            return returnValue

        except Exception as e:
            logger.error(f"🛑 ERROR: Not able to get sales data analysis response from {state['model']} for prompt. Exception: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise


    def chart_node(self, state: State) -> dict:
        """ Query the RAG system for chart data """

        logger.info("")
        logger.info("______________________CHART_NODE___________________________________")
        try:
            # Load and format prompt
            chart_template_text = load_prompt("CHART_PROMPT")
            logger.debug(f"Chart prompt template: \n{chart_template_text}\n")

            # Ensure it's a string and strip any leading/trailing whitespace
            if not isinstance(chart_template_text, str):
                raise TypeError(f"load_prompt returned unexpected type: {type(chart_template_text)}")
            clean_template = chart_template_text.strip()
            chart_prompt_template = PromptTemplate(
                input_variables=["retrieved_data"],
                template=clean_template
            )
            logger.debug(f"chart_prompt_template: \n{json.dumps(vars(chart_prompt_template), indent=2, default=str)}\n")

            # Create chain for creating chart data and invoke it
            logger.info(f"Calling LLM to request a CHART DATA using {state['model']} with temperature {state['temperature']}")
            llm = create_llm(state["model"], state["temperature"])
            summary_data_from_rag = state['sales_data_analysis_response'] 
            chart_chain = chart_prompt_template | llm
            response_object = chart_chain.invoke({"retrieved_data": summary_data_from_rag})

            # Extract the content string (the JSON output)
            chart_info_str = response_object.content 
            chart_info_json_str = clean_json_string(chart_info_str)
            logger.debug(f"Chart info: \n{chart_info_json_str}")
            try:
                parsed_chart_info = json.loads(chart_info_json_str)
            except json.JSONDecodeError as e:
                logger.error(f"🛑 ERROR: Failed to parse JSON response for chart node: {e}. Raw response: {chart_info_str}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise ValueError(f"LLM did not return valid JSON for chart information: {e}")
            summary =  f"[{state['model']}, {state['temperature']}] - {parsed_chart_info.get('summary')}"
            returnValue = {
                "chart_data": parsed_chart_info.get("chart_data"), 
                "chart_type": parsed_chart_info.get("chart_type"),
                "summary": summary,
            }
            logger.debug(f"returnValue: {returnValue}")
            return returnValue
        
        except Exception as e:
            logger.error(f"🛑 ERROR: Failed getting chart data response from {state['model']} ")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise


    def recommend_node(self, state: State) -> dict:
        """ Query the RAG system for insights """

        logger.info("")
        logger.info("______________________RECOMMENDATION_NODE__________________________")
        try:
            # Build the prompt
            recommendation_template_text = load_prompt("RECOMMENDATION_PROMPT")
            logger.debug(f"recommendation_template_text: \n{recommendation_template_text}\n")
            recommendation_prompt = PromptTemplate(
                input_variables=["question", "analysis"],
                template=recommendation_template_text)
            state["recommendation_prompt"] = recommendation_prompt.format(question=state['question'], analysis=state['sales_data_analysis_response'])
            logger.debug(f"recommendation_prompt: \n{json.dumps(vars(recommendation_prompt), indent=2, default=str)}\n")

            # Invoke the RAG chain
            logger.info(f"Calling RAG chain to request a RECOMMENDATION using {state['model']} with temperature {state['temperature']}")
            start_time = time.time()
            result = self.rag_chain.invoke({"query": state["recommendation_prompt"]})
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract response
            recommendation_response = result.get("result", "No result")
            logger.debug(f"Response received in {duration:.2f}s")
            response = recommendation_response[:RESPONSE_CUT_OFF] + "..." if len(recommendation_response) > RESPONSE_CUT_OFF else recommendation_response
            logger.debug(f"Recommendation response: {response}")
        
            # Return updates to state
            content = (
                f"[{state['model']}, {state['temperature']}] - SALES ANALYSIS & RECOMMENDATION\n\n"
                f"{state['sales_data_analysis_response']}\n\n"
                f"{recommendation_response}\n\n"
                f"Recommendation response time: {duration:.2f}s"
            )        
            returnValue = {
                "messages": [AIMessage(content=content)], 
                "final_response": content,
                "counter": state.get("counter", 0) + 1,
                "response_time": f"{duration:.2f}",
            }
            logger.debug(f"returnValue: {returnValue}")
            return returnValue

        except Exception as e:
            logger.error(f"🛑 ERROR: Not able to get RECOMMENDATION response from {state['model']} for prompt. Exception: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise  


#------------------------BUILDING-WORKFLOW--------------------------------

    def build_workflow(self):
        """Build and compile the LangGraph workflow"""

        workflow = StateGraph(State)

        TYPE_ANALYSIS_NODE = "type_analysis_node"
        PDF_TYPE_NODE = "pdf_type_node"
        SALES_DATA_ANALYSIS_NODE = "sales_data_analysis_node"
        CHART_NODE = "chart_node"
        RECOMMENDATION = "recommend_node"

        # Add nodes to the workflow
        workflow.add_node(TYPE_ANALYSIS_NODE, self.type_analysis_node)
        workflow.add_node(PDF_TYPE_NODE, self.pdf_type_node)
        workflow.add_node(SALES_DATA_ANALYSIS_NODE, self.sales_analysis_node)
        workflow.add_node(CHART_NODE, self.chart_node)
        workflow.add_node(RECOMMENDATION, self.recommend_node)

        # Add edges to connect the nodes
        workflow.add_edge(START, TYPE_ANALYSIS_NODE)
        workflow.add_conditional_edges(
            TYPE_ANALYSIS_NODE, 
            self.map_decision_to_node,
            {
                QUESTION_TYPE_SALES_SUM: SALES_DATA_ANALYSIS_NODE,
                QUESTION_TYPE_PDF: PDF_TYPE_NODE
            })
        workflow.add_edge(SALES_DATA_ANALYSIS_NODE, CHART_NODE)
        workflow.add_edge(CHART_NODE, RECOMMENDATION)
        workflow.add_edge(RECOMMENDATION, END)
        workflow.add_edge(PDF_TYPE_NODE, END)

        # Compile the workflow with memory
        memory = MemorySaver()
        self.lg_chain = workflow.compile(checkpointer=memory)         
        logger.info(f"LangGraph workflow with memory compiled successfully")
        return self.lg_chain


#------------------------UTILITY-METHODS--------------------------------

    def print_retrieved_docs(self, state):
        """ Show documents retrieved for particular question """

        retrieved_docs = test_retrieval_only(state['model'], state['question'], 10)
        for i, doc in enumerate(retrieved_docs):
            logger.debug(f"[{i}]  {doc.page_content}")


    def chart_data_to_chat(self, response_dict: dict):
        """ Add an AIMessage with chart metadata to the conversation """

        content_text = response_dict.get("summary", "Analysis complete.")
        metadata = {
            "chart_type": response_dict.get("chart_type"),
            "chart_data": response_dict.get("chart_data"),
            "summary": response_dict.get("summary"),
        }

        # Just add one new message
        ai_message = AIMessage(content=content_text, additional_kwargs=metadata)
        config = {"configurable": {"thread_id": self.thread_id}}
        self.lg_chain.update_state(values={"messages": [ai_message]}, config=config)
        logger.debug(f"Added AIMessage for chart to chat with thread id {self.thread_id}: \n{json.dumps(vars(ai_message), indent=2, default=str)}")
        return ai_message

  
    def get_conversation_history(self) -> list[BaseMessage]:
        """ Get the full conversation history for the current thread """

        if self.lg_chain is None:
            return []
        
        # Get from the chain's state all messages that have the thread id of this agent
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            state = self.lg_chain.get_state(config)
            return state.values.get("messages", [])
        
        except Exception as e:
            logger.error(f"🛑 ERROR: Problems getting conversation history: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        

    def print_conversation_history(self, detailed: bool = False):
        """ Print the conversation history to the log """

        # Get the conversion history from agent
        messages = self.get_conversation_history()
        if not messages:
            logger.info("No conversation history yet.")
            return
        
        logger.info("=" * 60)
        logger.info(f"CONVERSATION HISTORY ({len(messages)} messages)".center(60))
        logger.info("=" * 60)
        
        # Loop through messages in conversation history and print each message to the log
        for i, msg in enumerate(messages, 1):
            if isinstance(msg, HumanMessage):
                logger.info(f"\n[{i}] 👤 USER:")
                logger.info(f"    {msg.content}")
            elif isinstance(msg, AIMessage):
                logger.info(f"{i}] 🤖 ASSISTANT:")
                # Truncate long responses for readability
                content = msg.content[:RESPONSE_CUT_OFF] + "..." if len(msg.content) > RESPONSE_CUT_OFF else msg.content
                logger.info(f"    {content}")
            
            if detailed and msg.additional_kwargs:
                logger.info(f"    Metadata: {msg.additional_kwargs}")
        
        logger.info("=" * 60 + "\n")
    

    def get_conversation_summary(self) -> dict:
        """ Get a summary of the conversation """

        messages = self.get_conversation_history()
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
        
        # Pack it all up and return the summary information
        returnValue = {
            "total_messages": len(messages),
            "user_messages": len(user_msgs),
            "ai_messages": len(ai_msgs),
            "thread_id": self.thread_id
        }
        logger.debug(f"Conversion summary: \n {returnValue}")
        return returnValue
    

    def export_conversation(self, filename: str = CONVERSATION_FILENAME):
        """ Export conversation history to a text file """

        try:
            # Create directory if it doesn't exist yet
            converations_dir_path = Path(CONVERSATIONS_DIR_PATH)
            converations_dir_path.mkdir(parents=True, exist_ok=True)
                
            # Create and write to the file
            file_path = f"{converations_dir_path}/{CONVERSATION_FILENAME}"
            messages = self.get_conversation_history()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("INSIGHTFORGE CONVERSATION HISTORY\n")
                f.write("=" * 60 + "\n")
                f.write(f"Thread ID: {self.thread_id}\n")
                f.write(f"Total Messages: {len(messages)}\n")
                f.write("=" * 60 + "\n\n")
                
                for i, msg in enumerate(messages, 1):
                    role = "USER" if isinstance(msg, HumanMessage) else "ASSISTANT"
                    f.write(f"[{i}] {role}:\n")
                    f.write(f"{msg.content}\n\n")
                    f.write("-" * 60 + "\n\n")  

            logging.info(f"File created successfully at: {file_path}")

        except OSError as e:
            # Catches permission errors or other system failures
            logging.critical(f"FATAL ERROR: Failed to create directory or write to file. Reason: {e}")


    def reset_conversation(self):
        """Reset the conversation by creating a new thread_id"""

        # Creating new thread id and replacing the old one. So, all messages related to the old one 
        # will not be visible anymore.
        self.thread_id = str(uuid.uuid4())
        logger.info(f"Conversation reset with new thread_id: {self.thread_id}")
    

#------------------------INSTANTIATING-AGENT--------------------------------

def create_insight_agent(rag_chain):
    """ Factory function to create an insight agent """

    # Instantiate agent with a RAG chain
    agent = InsightAgent(rag_chain)

    # Build the LangGraph workflow
    agent.build_workflow()

    return agent