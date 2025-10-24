"""
app.py
Streamlit UI for InsightForge
"""


if __name__ == "__main__":
    # This prevents issues when Streamlit reloads modules
    pass

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



import logging
import json
from dotenv import load_dotenv
import streamlit as st
import altair as alt
import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage

from logging_config import configure_logging
from utils.line_loader import load_lines
from utils.prompt_loader import load_prompt
from rag_setup import create_rag_chain, CHUNK_SIZE, CHUNK_OVERLAP, NUM_RETURNED_DOCS
from rag_setup import MODEL_GEMMA, MODEL_GEMINI, MODEL_GPT35, MODEL_CLAUDE
from rag_agent import create_insight_agent
from rag_eval import load_test_dataset, EVAL_TEST_DATA_FILE_PATH, eval_single_test, load_eval_prompt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Logging setup. For now without writing log messages to log file
configure_logging(level=logging.WARNING)
# configure_logging(level=logging.INFO, logfile="insightforge.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Clear prompt cache to pick up any changes in prompt files right away
load_prompt.cache_clear()

# Load environment variables from .env
load_dotenv(verbose=True)

# Determine available models based on API keys and vectorstore existence
def get_available_models():
    """Returns list of models that can be used based on available API keys AND existing vectorstores"""
    available_models = []

    # Import here to avoid circular dependency
    from rag_setup import FAISS_INDEX_PATH_BASE

    def vectorstore_exists(model_name):
        """Check if vectorstore exists for a given model"""
        vectorstore_path = f"{FAISS_INDEX_PATH_BASE}_{model_name}"
        index_file = os.path.join(vectorstore_path, "index.faiss")
        return os.path.exists(index_file)

    # Check for OpenAI API key AND vectorstore
    if os.getenv("OPENAI_API_KEY") and vectorstore_exists(MODEL_GPT35):
        available_models.append(MODEL_GPT35)

    # Check for Google API key (for Gemini) AND vectorstore
    if os.getenv("GOOGLE_API_KEY") and vectorstore_exists(MODEL_GEMINI):
        available_models.append(MODEL_GEMINI)

    # Check for Anthropic API key (for Claude) AND vectorstore
    if os.getenv("ANTHROPIC_API_KEY") and vectorstore_exists(MODEL_CLAUDE):
        available_models.append(MODEL_CLAUDE)

    # Ollama doesn't require an API key, but still needs vectorstore
    if vectorstore_exists(MODEL_GEMMA):
        available_models.append(MODEL_GEMMA)

    return available_models

# Get available models (will be validated in init)
MODEL_CHOICES = get_available_models() 


def init_session_state():
    """ Initialize all session state variables """

    # Validate that at least one model is available
    if not MODEL_CHOICES:
        error_msg = """
        🛑 **ERROR: No models available**

        Models require BOTH an API key AND a built vectorstore.

        **To fix this:**
        1. Set at least one API key in your environment or .env file:
           - `OPENAI_API_KEY` for gpt-3.5-turbo
           - `ANTHROPIC_API_KEY` for Claude
           - `GOOGLE_API_KEY` for Gemini

        2. Build the vectorstore for that model:
           ```bash
           python rag_setup.py gpt-3.5-turbo
           # or
           python rag_setup.py claude-3-5-sonnet-20240620
           # or
           python rag_setup.py gemini-2.5-flash
           ```
        """
        logger.critical(error_msg)
        st.error(error_msg)
        st.stop()

    # Initialize session state variables
    if "temperature" not in st.session_state:
        st.session_state.temperature = float(os.getenv("MODEL_TEMPERATURE", 0))

    if "model" not in st.session_state:
        # Check if gen model provided as first parameter
        if len(sys.argv) > 1:
            requested_model = sys.argv[1]
            if requested_model in MODEL_CHOICES:
                st.session_state.model = requested_model
            else:
                logger.error(f"Invalid or unavailable model specified as parameter: {requested_model}")
                logger.error(f"Available models based on API keys: {MODEL_CHOICES}")
                st.error(f"Invalid model: {requested_model}. Available: {', '.join(MODEL_CHOICES)}")
                st.stop()
        else:
            # Use MODEL_DEFAULT from env if it's available, otherwise use first available model
            default_model = os.getenv("MODEL_DEFAULT", MODEL_GPT35)
            st.session_state.model = default_model if default_model in MODEL_CHOICES else MODEL_CHOICES[0]

    if "num_returned_docs" not in st.session_state:
        st.session_state.num_returned_docs = os.getenv("NUM_RETURNED_DOCS", NUM_RETURNED_DOCS)

    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = os.getenv("CHUNK_SIZE", CHUNK_SIZE)

    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = os.getenv("CHUNK_OVERLAP", CHUNK_OVERLAP)

    if "eval_model" not in st.session_state:
        # Eval model should also be from available models
        default_eval = os.getenv("EVAL_MODEL", MODEL_GPT35)
        st.session_state.eval_model = default_eval if default_eval in MODEL_CHOICES else MODEL_CHOICES[0]

    # Create RAG chain if it doesn't exist yet
    if "rag_chain" not in st.session_state:
        try:
            st.session_state.rag_chain = create_rag_chain(st.session_state.model, st.session_state.temperature)
        except Exception as e:
            error_msg = f"🛑 ERROR: RAG layer not set up yet for model [{st.session_state.model}]. Might need to run:  python rag_setup.py"
            logger.critical(error_msg)
            st.write(error_msg)
            st.stop()
    
    # Create insights agent if it doesn't exist yet
    if "agent" not in st.session_state:
        logger.info("Creating insight agent")
        st.session_state.agent = create_insight_agent(st.session_state.rag_chain)
    
    # Chat history if it doesn't exist yet
    if "history" not in st.session_state:
        st.session_state.history = []


def ask_question(question: str):
    """ Asks a question to RAG-based agent """

    # Launch chain that involves RAG layer
    response_dict = st.session_state.agent.run_insight_chain(
        st.session_state.model, 
        st.session_state.temperature,
        st.session_state.rag_chain,
        question
    )   

    # Add AIMessage for chart if chart is required
    if response_dict.get("chart_type"):
        st.session_state.agent.chart_data_to_chat(response_dict)


def save_feedback(index):
    """ Saves user feedback to the LangChain BaseMessage object's 
        additional_kwargs dictionary within the agent's retrieved history """
    
    # Get the specific BaseMessage object
    history_messages = st.session_state.agent.get_conversation_history()
    if index < len(history_messages):
        message_to_update = history_messages[index]
        feedback_value = st.session_state[f"feedback_{index}"]
        
        # Store the feedback value in the message's additional_kwargs
        message_to_update.additional_kwargs["feedback"] = feedback_value
        
        logger.info(
            f"Feedback saved for {st.session_state.model}'s message {index}: {feedback_value}. "
            f"Key saved to message.additional_kwargs."
        )
    else:
        logger.error(f"Cannot save feedback: Index {index} is out of bounds for conversation history.")


def write_out_history():
    """ Display chat history from LangGraph """

    # Loop through messages to print them in the chat UI
    agent = st.session_state.agent
    messages = agent.get_conversation_history()
    for i, msg in enumerate(messages):
        # Check for type of Message
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(f"[{i}] {msg.content}")

        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                # Check for chart data
                chart_type = msg.additional_kwargs.get("chart_type")
                chart_data = msg.additional_kwargs.get("chart_data")
                chart_title = msg.additional_kwargs.get("chart_title", "")
                x_axis_label = msg.additional_kwargs.get("x_axis_label", "")
                y_axis_label = msg.additional_kwargs.get("y_axis_label", "")
                summary = msg.additional_kwargs.get("summary")

                # Check whether it is a chart type of message
                if chart_data is not None and chart_type:
                    # For a chart type of message, convert data back to DataFrame if necessary
                    df = pd.DataFrame(chart_data)

                    # Build dynamic tooltip based on available columns
                    tooltip_fields = []
                    for field in ['month', 'product', 'region', 'total_sales', 'avg_satisfaction', 'gender', 'age', 'year']:
                        if field in df.columns:
                            tooltip_fields.append(field)

                    # Define base chart encoding shared by all charts
                    # Use product for color if available, otherwise use first categorical column
                    color_field = 'product:N' if 'product' in df.columns else None
                    if not color_field and 'region' in df.columns:
                        color_field = 'region:N'

                    logger.debug(f"Chart data: \n {df}")
                    logger.debug(f"Chart type: {chart_type}")
                    logger.debug(f"Tooltip fields: {tooltip_fields}")

                    # Different handling of different chart types
                    chart_type = chart_type.lower()
                    if not summary:
                        summary = "UNDEFINED"
                    summary = f"[{i}] {summary}"

                    # Use chart_title if available, otherwise use summary
                    title = chart_title if chart_title else summary

                    try:
                        if chart_type == "line chart":
                            # Determine x and y fields
                            x_field = 'month:O' if 'month' in df.columns else None
                            y_field = 'total_sales:Q' if 'total_sales' in df.columns else None

                            if x_field and y_field:
                                x_label = x_axis_label if x_axis_label else "Month"
                                y_label = y_axis_label if y_axis_label else "Total Sales ($)"

                                chart = alt.Chart(df).mark_line(point=True).encode(
                                    x=alt.X(x_field, title=x_label),
                                    y=alt.Y(y_field, title=y_label),
                                    color=alt.Color(color_field, title=color_field.split(':')[0].title()) if color_field else alt.value('steelblue'),
                                    detail=color_field.split(':')[0] if color_field else alt.value(None),
                                    tooltip=tooltip_fields
                                ).properties(
                                    title=title,
                                    width=600,
                                    height=400
                                )
                            else:
                                st.warning("Insufficient data for line chart")
                                chart = None

                        elif chart_type == "bar chart":
                            # Determine x and y fields
                            x_field = 'month:O' if 'month' in df.columns else ('product:N' if 'product' in df.columns else None)
                            y_field = 'total_sales:Q' if 'total_sales' in df.columns else None

                            if x_field and y_field:
                                x_label = x_axis_label if x_axis_label else x_field.split(':')[0].title()
                                y_label = y_axis_label if y_axis_label else "Total Sales"

                                chart = alt.Chart(df).mark_bar().encode(
                                    x=alt.X(x_field, title=x_label),
                                    y=alt.Y(y_field, title=y_label),
                                    color=alt.Color(color_field, title=color_field.split(':')[0].title()) if color_field else alt.value('steelblue'),
                                    tooltip=tooltip_fields
                                ).properties(
                                    title=title,
                                    width=600,
                                    height=400
                                )
                            else:
                                st.warning("Insufficient data for bar chart")
                                chart = None

                        elif chart_type == "stacked bar chart":
                            # Stacked bar chart for part-to-whole relationships
                            x_field = 'month:O' if 'month' in df.columns else ('region:N' if 'region' in df.columns else None)
                            y_field = 'total_sales:Q' if 'total_sales' in df.columns else None

                            if x_field and y_field and color_field:
                                x_label = x_axis_label if x_axis_label else x_field.split(':')[0].title()
                                y_label = y_axis_label if y_axis_label else "Total Sales"

                                chart = alt.Chart(df).mark_bar().encode(
                                    x=alt.X(x_field, title=x_label),
                                    y=alt.Y(y_field, title=y_label, stack='zero'),
                                    color=alt.Color(color_field, title=color_field.split(':')[0].title()),
                                    tooltip=tooltip_fields
                                ).properties(
                                    title=title,
                                    width=600,
                                    height=400
                                )
                            else:
                                st.warning("Insufficient data for stacked bar chart (needs grouping variable)")
                                chart = None

                        elif chart_type == "scatter plot":
                            # Scatter plot for correlation analysis
                            x_field = 'total_sales:Q' if 'total_sales' in df.columns else None
                            y_field = 'avg_satisfaction:Q' if 'avg_satisfaction' in df.columns else None

                            if x_field and y_field:
                                x_label = x_axis_label if x_axis_label else "Total Sales ($)"
                                y_label = y_axis_label if y_axis_label else "Avg. Satisfaction"

                                chart = alt.Chart(df).mark_circle(size=100).encode(
                                    x=alt.X(x_field, title=x_label),
                                    y=alt.Y(y_field, title=y_label),
                                    color=alt.Color(color_field, title=color_field.split(':')[0].title()) if color_field else alt.value('steelblue'),
                                    tooltip=tooltip_fields
                                ).properties(
                                    title=title,
                                    width=600,
                                    height=400
                                )
                            else:
                                st.warning("Insufficient data for scatter plot")
                                chart = None

                        elif chart_type == "area chart":
                            # Area chart for cumulative trends
                            x_field = 'month:O' if 'month' in df.columns else None
                            y_field = 'total_sales:Q' if 'total_sales' in df.columns else None

                            if x_field and y_field:
                                x_label = x_axis_label if x_axis_label else "Month"
                                y_label = y_axis_label if y_axis_label else "Total Sales ($)"

                                chart = alt.Chart(df).mark_area(opacity=0.7).encode(
                                    x=alt.X(x_field, title=x_label),
                                    y=alt.Y(y_field, title=y_label),
                                    color=alt.Color(color_field, title=color_field.split(':')[0].title()) if color_field else alt.value('steelblue'),
                                    tooltip=tooltip_fields
                                ).properties(
                                    title=title,
                                    width=600,
                                    height=400
                                )
                            else:
                                st.warning("Insufficient data for area chart")
                                chart = None

                        else:
                            st.warning(f"Chart type '{chart_type}' is not supported. Supported types: line chart, bar chart, stacked bar chart, scatter plot, area chart")
                            chart = None

                    except Exception as chart_error:
                        st.error(f"Error creating chart: {str(chart_error)}")
                        logger.error(f"Chart rendering error: {chart_error}")
                        chart = None

                    # Display the chart in Streamlit
                    if chart is not None:
                        st.altair_chart(chart, use_container_width=True)
                        # Show summary text below the chart
                        if summary:
                            st.caption(summary)

                else:
                    # It is a text type of message
                    if not msg.content == "":
                        st.write(f"[{i}] {msg.content}")

                        # Retrieve feedback from the message's additional_kwargs
                        saved_feedback = msg.additional_kwargs.get("feedback", None)
                        widget_key = f"feedback_{i}"
                        if widget_key not in st.session_state or st.session_state[widget_key] is None:
                            # Initialize or restore the value from history metadata
                            st.session_state[widget_key] = saved_feedback


def render_sidebar():
    """ Render the sidebar with precanned questions """

    with st.sidebar:
        # Add inference controlls
        render_infer_panel()
        st.divider()

        # Add conversation tools
        render_convers_panel()

        # Add evaluation tools
        render_eval_panel()

        # Add debug tools
        render_debug_panel()

        # RAG panel
        render_rag_panel()


def render_infer_panel():
    """ Render the Inference panel"""

    # Print the header of this panel
    st.subheader("Inference Config")

    # Dropdown for precanned questions
    predefined_questions = load_lines("PREDEFINED_QUESTIONS")
    precanned_question_label = "-- Select a question --"
    def on_select_precanned():
        selected_question = st.session_state.quick_select.removeprefix("BUS | ").removeprefix("PDF | ")
        if selected_question != precanned_question_label:
            # Run the query for the predefined question that was selected
            ask_question(selected_question)   
            st.session_state.quick_select = precanned_question_label
    
    # Build question list with placeholder
    question_options = [precanned_question_label] + predefined_questions
    st.selectbox(
        "Precanned questions:",
        question_options,
        key="quick_select",
        on_change=on_select_precanned
    )

    # Create the slider for model temperature
    new_temperature = st.slider(
        'From 0 (higher precision) to 1 (higher variation)',
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.01,
        format='%.2f' 
    )
    if new_temperature != st.session_state.temperature:
        st.session_state.temperature = new_temperature
        logger.info(f"Changed model temperature to {st.session_state.temperature}")
        st.session_state.rag_chain = create_rag_chain(st.session_state.model, st.session_state.temperature)
    
    # Dropdown for models 
    currently_used_model = st.session_state.model
    def on_select_model():
        selected_model = st.session_state.model_selected
        if selected_model != currently_used_model:
            # New model selected; need to also create a new RAG chain for it
            st.session_state.model = selected_model
            try:
                st.session_state.rag_chain = create_rag_chain(st.session_state.model, st.session_state.temperature)
            except Exception as e:
                st.error(f"**CRITICAL ERROR:** Failed to load required vectorstore for **{st.session_state.model}**.")
                st.warning(f"Please ensure the vector store exists for model {st.session_state.model}.")
                st.warning(f"With MODEL_DEFAULT=\"{st.session_state.model}\" defined in .env run:     python rag_setup.py")
                st.stop()
            st.session_state.agent.rag_chain = st.session_state.rag_chain
            logger.info(f"New model selected: {st.session_state.model}")

    # Selection of model
    st.selectbox(
        "Select model:",
        MODEL_CHOICES,
        key="model_selected",
        index=MODEL_CHOICES.index(st.session_state.model),
        on_change=on_select_model
    )


def render_convers_panel():
    """ Render the Converation panel in sidebar """

    # Create expander for conversation related controls
    with st.expander("💬 Conversation"):
        summary = st.session_state.agent.get_conversation_summary()
        st.write(f"Messages: {summary['total_messages']}")
        
        if st.button("🖨️ Print History"):
            st.session_state.agent.print_conversation_history()
        
        if st.button("🔄 Reset"):
            st.session_state.agent.reset_conversation()
            st.rerun()

        if st.button("Export Conversation"):
            logger.info("")
            st.session_state.agent.export_conversation("insight_conversation.txt")
            st.success("Exported!")


def render_eval_panel(): 
    """ Render the Evaluation panel in sidebar """  

    # Create expander for evaluation panel
    with st.expander("Evaluation"):
        # Get list of testcases
        test_cases = load_test_dataset(EVAL_TEST_DATA_FILE_PATH)
        questions_of_testcases = [item['query'] for item in test_cases]

        # Button for default testcase, picking the first one in the list
        if st.button("Default eval testcase"):
            # Making the firs testcase in rag_eval_test_data.txt the default testcase
            graded_outputs = eval_single_test(test_cases[0], st.session_state.eval_model)   
            st.session_state.graded_outputs = graded_outputs

        # Dropdown for evaluation question/answer items
        default_item = "-- Select a test case --"
        def on_select_testcase():
            selected_question = st.session_state.select_testcase
            if selected_question != default_item:
                # Run the query to run the default testcase
                test_case = "UNDEFINED"
                for item in test_cases:
                     if item.get("query") == selected_question:
                        test_case = item
                graded_outputs = eval_single_test(test_case, st.session_state.eval_model)   
                st.session_state.select_testcase = default_item
                st.session_state.graded_outputs = graded_outputs
        
        # Build question list with placeholder
        testcase_options = [default_item] + questions_of_testcases
        st.selectbox(
            "Questions of testcases:",
            testcase_options,
            key="select_testcase",
            on_change=on_select_testcase
        )


def render_rag_panel():
    """ Render the debug panel in sidebar """ 

    # Create the RAG setup expander
    with st.expander("RAG Setup"):
        #----------------------------------------------------------------
        # Set the header for RAG creation
        st.subheader("RAG Layer Config")

        # Chunk size adjustment control
        if False:
            new_chunksize = st.number_input(
                label='Enter a new chunk size:',
                min_value=100,
                max_value=2000,
                value=int(st.session_state.chunk_size),
                step=1,
                key='chunk_size_input'
            )
            if new_chunksize != st.session_state.chunk_size:
                st.session_state.chunk_size = new_chunksize
                logger.info(f"Changed chunk size to {st.session_state.chunk_size}")

            # Chunk overlap adjustment control
            new_chunkoverlap = st.number_input(
                label='Enter a new chunk size:',
                min_value=10,
                max_value=200,
                value=int(st.session_state.chunk_overlap),
                step=1,
                key='chunk_overlap_input'
            )        
            if new_chunkoverlap != st.session_state.chunk_overlap:
                st.session_state.chunk_overlap = new_chunkoverlap
                logger.info(f"Changed chunk overlap to {st.session_state.chunk_overlap}")

        # Modifying number of documents retrieved from RAG layer
        new_num_docs_retrieved = st.number_input(
            label='Enter number of RAG docs to be retrieved:',
            min_value=10,
            max_value=200,
            value=int(st.session_state.num_returned_docs),
            step=1,
            key='num_docs_input'
        )
        if new_num_docs_retrieved != st.session_state.num_returned_docs:
            st.session_state.num_returned_docs = new_num_docs_retrieved
            st.session_state.rag_chain = create_rag_chain(st.session_state.model, st.session_state.temperature)
            logger.info(f"Number of documents to be retrieved from RAG layer set to [{st.session_state.num_returned_docs}]")

        debug_mode = st.checkbox("Enable debug mode", key="debug_mode")
        if debug_mode:
            st.subheader("🔍 Runtime Info")
            
            # Show session state info
            st.write("**History length:**", len(st.session_state.get("history", [])))
            st.write("**Vectorstore loaded:**", "vectorstore" in st.session_state)
            st.write("**Agent initialized:**", "agent" in st.session_state)
            
            # Show environment variables (safe subset)
            if st.toggle("Show environment variables (safe)"):
                env_vars = {
                    k: v for k, v in os.environ.items() 
                    if not any(secret in k.upper() for secret in ["KEY", "TOKEN", "SECRET", "PASSWORD"])
                }
                st.json(env_vars)
            
            # Show last query info
            if st.session_state.history:
                if st.toggle("Show last query details"):
                    last_messages = st.session_state.history[-2:] if len(st.session_state.history) >= 2 else st.session_state.history
                    st.json(last_messages)


def render_debug_panel():
    """Render the debug panel in sidebar"""
    with st.expander("Debug"):
        st.subheader("🔍 Runtime Info")
        
        # Show session state info
        st.write("**History length:**", len(st.session_state.get("history", [])))
        st.write("**Vectorstore loaded:**", "vectorstore" in st.session_state)
        st.write("**Agent initialized:**", "agent" in st.session_state)
        
        # Show environment variables (safe subset)
        if st.toggle("Show environment variables (safe)"):
            env_vars = {
                k: v for k, v in os.environ.items() 
                if not any(secret in k.upper() for secret in ["KEY", "TOKEN", "SECRET", "PASSWORD"])
            }
            st.json(env_vars)
        
        # Show last query info
        if st.session_state.history:
            if st.toggle("Show last query details"):
                last_messages = st.session_state.history[-2:] if len(st.session_state.history) >= 2 else st.session_state.history
                st.json(last_messages)


def run_streamlit_app():
    """ InsightForge Streamlit application """
    
    # Initialize session state
    logger.debug("---------------------------------------------------------")
    init_session_state()
    
    # Title and side bar
    st.title("InsightForge")
    st.caption("AI-powered sales data insights")
    render_sidebar()

    # Graded output
    if "graded_outputs" in st.session_state:
        eval_prompt = load_eval_prompt()
        st.write({"eval_prompt": eval_prompt})
        st.write(st.session_state.graded_outputs)
    
    # Chat input
    if question := st.chat_input("Enter your question here"):
        ask_question(question)   

    # Display chat history
    write_out_history()

    logger.info("")
    logger.info("Waiting for next user action ...")
    logger.info("")

if __name__ == "__main__":
    run_streamlit_app()