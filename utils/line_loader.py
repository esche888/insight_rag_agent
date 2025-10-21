import os
from dotenv import load_dotenv
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

load_dotenv()

CUTOFF_MARKER = "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"

@lru_cache(maxsize=None)
def load_lines(key_name: str) -> list[str]:
    """ Loads and caches a list of one-line questions from a file (if *_PATH is defined)
    or from an inline .env variable (comma-separated). """
    path_key = f"{key_name}_PATH"
    path_value = os.getenv(path_key)

    # Case 1: File-based
    if path_value and os.path.exists(path_value):
        with open(path_value, "r", encoding="utf-8") as f:
            lines = f.readlines()
        questions = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue  # skip blank or comment lines
            if CUTOFF_MARKER in stripped:
                logger.debug(f"Cutoff marker found at line {i+1} in {path_value}")
                break
            questions.append(stripped.strip(", "))  # remove trailing commas
        logger.debug(f"Loaded {len(questions)} lines from file: {path_value}")
        return questions

    # Case 2: Inline comma-separated list in .env
    inline_value = os.getenv(key_name)
    if inline_value:
        # Split by comma or semicolon
        parts = [q.strip().strip(",") for q in inline_value.replace(";", ",").split(",")]
        questions = [q for q in parts if q]
        logger.debug(f"Loaded {len(questions)} lines from environment variable: {key_name}")
        return questions

    raise ValueError(
        f"⚠️ Neither {path_key} nor {key_name} found or valid in .env"
    )

if __name__ == "__main__":
    prompt = load_lines("RECOMMENDATION_PROMPT")
    logger.info(f"Prompt from file: {prompt}")