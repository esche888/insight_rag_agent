"""
rag_eval.py
Testing RAG layer using QAEvalChain
"""

import os
import sys
import json
import logging
import traceback
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

from langchain.evaluation.qa import QAEvalChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from logging_config import configure_logging
from utils.prompt_loader import load_prompt
from rag_setup import create_rag_chain, MODEL_DEFAULT, MODEL_TEMPERATURE, EVAL_TEST_DATA_FILE_PATH
from rag_setup import MODEL_GEMMA, MODEL_GEMINI, MODEL_GPT35
from rag_agent import create_insight_agent


# Set up logging 
configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(verbose=True)
EVAL_LLM_TEMPERATURE = float(os.getenv("EVAL_LLM_TEMPERATURE", 0.2))
EVAL_MODEL = os.getenv("EVAL_MODEL", MODEL_GPT35)


def load_test_dataset(eval_test_file) -> List[Dict]:
    """ Load test questions with expected answers """

    # Read in testcase data
    eval_test_data = None
    try:
        with open(eval_test_file, 'r') as json_file:
            # Parse the JSON content.
            eval_test_data = json.load(json_file)
        
        # Verify the results
        logger.debug(f"Successfully loaded data from {EVAL_TEST_DATA_FILE_PATH}.")
        
    except FileNotFoundError:
        logger.error(f"🛑 ERROR: The file {eval_test_file} was not found.", file=sys.stderr)
        
    except json.JSONDecodeError as e:
        logger.error(f"🛑 ERROR: Failed decoding JSON from file: {e}", file=sys.stderr)
    
    return eval_test_data


def make_predictions(agent, test_cases: List[Dict], model: str, temperature: float) -> List[Dict]:
    """ Run test cases through the agent to get predictions """

    # Looping through testcases
    logger.info(f"Running {len(test_cases)} test cases")
    predictions = []
    for i, test in enumerate(test_cases, 1):
        # Use test case data to run test
        logger.info("")
        logger.info(f"[Test {i}/{len(test_cases)}] {test['query']}")
        try:
            # Run the agent to get prediction for query of testcase
            result = agent.run_insight_chain(
                model=MODEL_DEFAULT,
                temperature=temperature,
                rag_chain=agent.rag_chain,
                question=test['query']
            )
            
            # Extract the response (which is the prediction); remove the model/temperature prefix if present
            response_text = result.get("text", "")
            if "] - " in response_text:
                actual_response = response_text.split("] - ", 1)[1].split("\n\n Duration:")[0]
            else:
                actual_response = response_text
            
            # Add cleaned prediction data to list of predictions for later processing
            predictions.append({
                "query": test['query'],
                "answer": test['answer'],
                "result": actual_response,
                "category": test.get('category', 'uncategorized'),
                "duration": result.get("duration", "N/A")
            })
            logger.info(f"✓ Completed making predictions")
            
        except Exception as e:
            logger.error(f"🛑 ERROR: Failed ot make a prediction. Exception: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            predictions.append({
                "query": test['query'],
                "answer": test['answer'],
                "result": f"ERROR: {str(e)}",
                "category": test.get('category', 'uncategorized'),
                "duration": "N/A"
            })
    
    return predictions



def evaluate_predictions(test_cases: List[Dict], predictions: List[Dict], eval_model: str = EVAL_MODEL, eval_temp: str = EVAL_LLM_TEMPERATURE) -> List[Dict]:
    """ Evaluate predictions using QAEvalChain """

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluating Predictions with QAEvalChain")
    logger.info("=" * 60)
    
    # Create evaluation chain with custom prompt to assess predictions (result) against ground truth (answer)
    eval_template_text = load_prompt("EVAL_PROMPT")
    eval_template = PromptTemplate(
        input_variables=["question", "answer", "result"],
        template=eval_template_text)
    logger.debug(f"Eval prompt template: \n{eval_template}\n")
    if eval_model == MODEL_GPT35:
        eval_llm = ChatOpenAI(temperature=eval_temp, model=eval_model)
    if eval_model == MODEL_GEMINI:
        eval_llm = ChatGoogleGenerativeAI(temperature=eval_temp, model=eval_temp)
    eval_chain = QAEvalChain.from_llm(eval_llm, prompt=eval_template)
    
    # Evaluate predictions (results) against ground truths (answers)
    graded_outputs = eval_chain.evaluate(
        test_cases,
        predictions,
        question_key="query",
        answer_key="answer",
        prediction_key="result"
    )
    return graded_outputs


def load_eval_prompt():
    """ Return eval template text for display in UI """

    # Get eval template text from template file
    eval_template_text = load_prompt("EVAL_PROMPT")
    logger.debug(f"Chart prompt template: \n{eval_template_text}\n")
    return eval_template_text


def print_eval_results(test_cases: List[Dict], predictions: List[Dict], graded_outputs: List[Dict]):
    """ Print detailed evaluation results """
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS".center(80))
    logger.info("=" * 80)
    
    # Loop through test results
    correct_count = 0
    incorrect_count = 0
    # partial_count = 0
    error_count = 0
    results_by_category = {}
    for i, (test, pred, grade) in enumerate(zip(test_cases, predictions, graded_outputs), 1):
        # Determine test outcome
        category = test.get('category', 'uncategorized')
        grade_text = grade['results'].upper()
        is_correct = "CORRECT" in grade_text and "INCORRECT" not in grade_text
        if "CORRECT" in grade_text and "INCORRECT" not in grade_text:
            status = "✅ CORRECT"
            correct_count += 1
        elif "INCORRECT" in grade_text:
            status = "🟡 INCORRECT"
            incorrect_count += 1
        # elif "PARTIAL" in grade_text:
        #     status = "⚠️ PARTIAL"
        #     partial_count += 1
        elif "ERROR" in pred['result']:
            status = "🛑 ERROR"
            error_count += 1
        else:
            status = "❓ UNCLEAR"
        
        # Calculate test statistics by category
        if category not in results_by_category:
            # Initialize with both correct and total
            results_by_category[category] = {'correct': 0, 'total': 0}

        # Increment 'total' always, but 'correct' only if result was correct
        results_by_category[category]['total'] += 1
        if is_correct:
            results_by_category[category]['correct'] += 1
        
        # Print result
        logger.info("")
        logger.info(f"[Test {i}] {category.upper()}")
        logger.info("")
        logger.info(f"Question: {test['query']}")
        logger.info(f"Expected: {test['answer'][:300]}...")
        logger.info(f"Got: {pred['result'][:300]}...")
        logger.info(f"Duration: {pred['duration']}")
        logger.info(f"Evaluation: {grade['results']}")
        logger.info(f"Status: {status}")
        logger.info("-" * 80)
    
    # Summary
    total = len(test_cases)
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY".center(80))
    logger.info("=" * 80)
    logger.info(f"Total Tests: {total}")
    logger.info(f"Correct: {correct_count} ({correct_count/total*100:.1f}%)")
    logger.info(f"Incorrect: {incorrect_count} ({incorrect_count/total*100:.1f}%)")
    # logger.info(f"Partial: {partial_count} ({partial_count/total*100:.1f}%)")
    logger.info(f"Errors: {error_count} ({error_count/total*100:.1f}%)")
    
    # Category breakdown
    if results_by_category:
        logger.info("-" * 80)
        logger.info("BY CATEGORY:")
        for category, stats in results_by_category.items():
            accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            logger.info(f"  {category}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    logger.info("=" * 80)


def export_results(test_cases: List[Dict], predictions: List[Dict], graded_outputs: List[Dict], filepath: str = "test_results.json"):
    """ Export test results to JSON file """
    
    # Loop through test results to create results list
    results = []
    for test, pred, grade in zip(test_cases, predictions, graded_outputs):
        results.append({
            "question": test['query'],
            "expected": test['answer'],
            "prediction": pred['result'],
            "evaluation": grade['results'],
            "category": test.get('category', 'uncategorized'),
            "duration": pred['duration']
        })
 
    # Print results list to file
    if False:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results exported to {filepath}")

    return results


def run_full_eval(
    model: str = MODEL_DEFAULT,
    temperature: float = MODEL_TEMPERATURE,
    eval_model: str = EVAL_MODEL,
    eval_model_temperature: str = EVAL_LLM_TEMPERATURE
):
    """ Run complete evaluation pipeline """

    logger.info("")
    logger.info("=" * 80)
    logger.info("RAG AGENT EVALUATION PIPELINE".center(80))
    logger.info("=" * 80)
    
    # Load test cases
    test_cases = load_test_dataset(EVAL_TEST_DATA_FILE_PATH)
    logger.info(f"Loaded {len(test_cases)} test cases")
    logger.info(f"Testing RAG agent with {model} (temp={temperature})")
    logger.info(f"Evaluation model: {eval_model}")
    
    # Initialize agent
    rag_chain = create_rag_chain(MODEL_DEFAULT, MODEL_TEMPERATURE)
    agent = create_insight_agent(rag_chain)
    
    # Run tests
    predictions = make_predictions(agent, test_cases, model, temperature)
    
    # Evaluate
    graded_outputs = evaluate_predictions(test_cases, predictions, eval_model, eval_model_temperature)
    
    # Report results
    print_eval_results(test_cases, predictions, graded_outputs)
    
    # Export results
    export_results(test_cases, predictions, graded_outputs)
    
    return test_cases, predictions, graded_outputs


def eval_single_test(test_case, eval_model: str = MODEL_GPT35):
    """Quick test for a single question"""

    # Initialize agent
    logger.debug("Initializing RAG agent")
    rag_chain = create_rag_chain(MODEL_DEFAULT, MODEL_TEMPERATURE)
    agent = create_insight_agent(rag_chain)
    
    # Run tests
    test_cases = []
    test_cases.append(test_case)
    predictions = make_predictions(agent, test_cases, eval_model, MODEL_TEMPERATURE)
    logger.debug(f" Predictions: {json.dumps(predictions, indent=2, default=str)}")

    # Evaluate
    graded_outputs = evaluate_predictions(test_cases, predictions, eval_model) 
    
    # Report results
    print_eval_results(test_cases, predictions, graded_outputs)
    
    # Export results
    results = export_results(test_cases, predictions, graded_outputs)
    logger.debug(f"Results:{json.dumps(results, indent=2, default=str)}")

    return results[0]


def get_answer_by_query(json_string: str, target_query: str) -> Optional[str]:
    """ Parses a JSON string containing QA pairs and returns the answer for a matching query """

    try:
        # Parse the JSON string into a Python list of dictionaries
        data_list: List[Dict[str, Any]] = json.loads(json_string)
        
        # Iterate through the list and return the 'answer' (ground truth) to the 'query'
        for item in data_list:
            if item.get("query") == target_query:
                return item.get("answer")
                
        # If the loop completes without finding a match, return None
        return None
        
    except json.JSONDecodeError as e:
        logger.error(f"🛑 ERROR: Invalid JSON format. {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return None


if __name__ == "__main__":
    # Run full evaluation
    run_full_eval(
        model=MODEL_DEFAULT,
        temperature=MODEL_TEMPERATURE,
        eval_model=EVAL_MODEL
    )