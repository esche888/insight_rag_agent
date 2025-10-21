import os
from dotenv import load_dotenv
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

load_dotenv()

CUTOFF_MARKER = "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"

@lru_cache(maxsize=None)
def load_prompt(key_name: str) -> str:
    """
    Loads and caches a prompt template from either:
      1. A text file (if *_PATH is defined in .env)
      2. A string value (if the prompt itself is in .env)
    
    Features:
    - Replaces literal '\\n' with real newlines for .env strings.
    - Strips whitespace.
    - Ignores any text after a line containing the cutoff marker.
    - Caches results for performance.

    Example:
        RECOMMENDATION_PROMPT_PATH=prompts/recommendation_prompt.txt
        or
        RECOMMENDATION_PROMPT=You are an expert.\nAnalyze: {question}

        load_prompt("RECOMMENDATION_PROMPT") -> str
    """
    path_key = f"{key_name}_PATH"
    path_value = os.getenv(path_key)

    if path_value and os.path.exists(path_value):
        with open(path_value, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Stop reading when marker line is found
        filtered_lines = []
        for line in lines:
            if CUTOFF_MARKER in line:
                break
            filtered_lines.append(line)
        text = "".join(filtered_lines).strip()
        logger.info(f"Loaded prompt from file: {path_value}")
        return text

    # Otherwise, try to load directly from .env
    text_value = os.getenv(key_name)
    if text_value:
        text = text_value.replace("\\n", "\n").strip()
        logger.info(f"Loaded prompt from environment variable: {key_name}")
        return text

    raise ValueError(
        f"⚠️ Neither {path_key} nor {key_name} found or valid in .env"
    )


if __name__ == "__main__":
    prompt = load_prompt("RECOMMENDATION_PROMPT")
    print(f"Prompt from file: {prompt}")