import json

def clean_json_string(raw_string: str) -> str:
    """Removes markdown fences and returns a clean JSON string."""
    
    # Strip external whitespace/newlines from the whole string.
    cleaned = raw_string.strip()

    # Remove the starting fence and optional newline/whitespace.
    if cleaned.startswith('```json'):
        cleaned = cleaned.removeprefix('```json').lstrip()
    
    # Remove the ending fence.
    if cleaned.endswith('```'):
        cleaned = cleaned.removesuffix('```').rstrip()
        
    # Strip any remaining whitespace/newlines
    return cleaned.strip()