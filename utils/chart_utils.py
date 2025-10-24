"""
chart_utils.py
Utility functions for chart data validation and processing
"""

import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Supported chart types
SUPPORTED_CHART_TYPES = [
    "line chart",
    "bar chart",
    "stacked bar chart",
    "scatter plot",
    "area chart"
]

# Common field mappings
COMMON_FIELDS = {
    'temporal': ['month', 'year', 'date', 'quarter'],
    'categorical': ['product', 'region', 'category', 'gender'],
    'quantitative': ['total_sales', 'avg_satisfaction', 'sales', 'revenue', 'satisfaction', 'age']
}


def validate_chart_data(chart_data: Any, chart_type: str) -> Tuple[bool, str]:
    """
    Validate chart data structure and content.

    Args:
        chart_data: The chart data to validate (should be list of dicts)
        chart_type: The type of chart being created

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if data is a list
    if not isinstance(chart_data, list):
        return False, f"Chart data must be a list, got {type(chart_data).__name__}"

    # Check if list is empty
    if len(chart_data) == 0:
        return False, "Chart data is empty"

    # Check if items are dictionaries
    if not all(isinstance(item, dict) for item in chart_data):
        return False, "All chart data items must be dictionaries"

    # Check if all items have the same keys
    first_keys = set(chart_data[0].keys())
    for i, item in enumerate(chart_data[1:], 1):
        if set(item.keys()) != first_keys:
            return False, f"Item {i} has different keys than item 0"

    # Check for at least one numeric field
    has_numeric = any(
        isinstance(v, (int, float))
        for item in chart_data
        for v in item.values()
    )
    if not has_numeric:
        return False, "Chart data must contain at least one numeric field"

    # Chart-specific validations
    chart_type_lower = chart_type.lower()

    if chart_type_lower == "line chart":
        # Line charts need temporal data
        has_temporal = any(field in first_keys for field in COMMON_FIELDS['temporal'])
        if not has_temporal:
            return False, "Line chart requires temporal field (month, year, date, or quarter)"

    elif chart_type_lower == "scatter plot":
        # Scatter plots need at least 2 quantitative fields
        quantitative_count = sum(
            1 for field in first_keys
            if any(isinstance(chart_data[0][field], (int, float)) for _ in [None])
        )
        if quantitative_count < 2:
            return False, "Scatter plot requires at least 2 quantitative fields"

    elif chart_type_lower == "stacked bar chart":
        # Stacked bar charts need categorical grouping
        has_categorical = any(field in first_keys for field in COMMON_FIELDS['categorical'])
        if not has_categorical:
            return False, "Stacked bar chart requires categorical field (product, region, category, etc.)"

    return True, ""


def infer_chart_fields(chart_data: List[Dict[str, Any]], chart_type: str) -> Dict[str, Optional[str]]:
    """
    Infer appropriate x, y, and color fields based on data structure and chart type.

    Args:
        chart_data: List of data dictionaries
        chart_type: Type of chart

    Returns:
        Dictionary with 'x_field', 'y_field', 'color_field' keys
    """
    if not chart_data or len(chart_data) == 0:
        return {'x_field': None, 'y_field': None, 'color_field': None}

    first_item = chart_data[0]
    fields = list(first_item.keys())

    # Categorize fields
    temporal_fields = [f for f in fields if f in COMMON_FIELDS['temporal']]
    categorical_fields = [f for f in fields if f in COMMON_FIELDS['categorical']]
    quantitative_fields = [
        f for f in fields
        if isinstance(first_item[f], (int, float))
    ]

    chart_type_lower = chart_type.lower()
    result = {'x_field': None, 'y_field': None, 'color_field': None}

    if chart_type_lower in ["line chart", "area chart"]:
        # X: temporal, Y: quantitative, Color: categorical
        result['x_field'] = f"{temporal_fields[0]}:O" if temporal_fields else None
        result['y_field'] = f"{quantitative_fields[0]}:Q" if quantitative_fields else None
        result['color_field'] = f"{categorical_fields[0]}:N" if categorical_fields else None

    elif chart_type_lower in ["bar chart", "stacked bar chart"]:
        # X: temporal or categorical, Y: quantitative, Color: categorical
        if temporal_fields:
            result['x_field'] = f"{temporal_fields[0]}:O"
        elif categorical_fields:
            result['x_field'] = f"{categorical_fields[0]}:N"

        result['y_field'] = f"{quantitative_fields[0]}:Q" if quantitative_fields else None

        # For color, prefer a different categorical field than x
        for cat_field in categorical_fields:
            if result['x_field'] and cat_field not in result['x_field']:
                result['color_field'] = f"{cat_field}:N"
                break

    elif chart_type_lower == "scatter plot":
        # X: quantitative, Y: quantitative (different), Color: categorical
        if len(quantitative_fields) >= 2:
            result['x_field'] = f"{quantitative_fields[0]}:Q"
            result['y_field'] = f"{quantitative_fields[1]}:Q"
        result['color_field'] = f"{categorical_fields[0]}:N" if categorical_fields else None

    logger.debug(f"Inferred fields for {chart_type}: {result}")
    return result


def get_default_labels(field_spec: Optional[str]) -> str:
    """
    Get default label for a field specification.

    Args:
        field_spec: Field specification like 'month:O' or 'total_sales:Q'

    Returns:
        Human-readable label
    """
    if not field_spec:
        return ""

    # Extract field name (before the colon)
    field_name = field_spec.split(':')[0] if ':' in field_spec else field_spec

    # Convert snake_case to Title Case
    label = field_name.replace('_', ' ').title()

    # Special cases
    label_map = {
        'Total Sales': 'Total Sales ($)',
        'Avg Satisfaction': 'Average Satisfaction',
        'Age': 'Age (years)'
    }

    return label_map.get(label, label)


def suggest_chart_type(chart_data: List[Dict[str, Any]]) -> str:
    """
    Suggest an appropriate chart type based on data characteristics.

    Args:
        chart_data: List of data dictionaries

    Returns:
        Suggested chart type
    """
    if not chart_data or len(chart_data) == 0:
        return "bar chart"

    first_item = chart_data[0]
    fields = list(first_item.keys())

    # Categorize fields
    has_temporal = any(f in COMMON_FIELDS['temporal'] for f in fields)
    has_categorical = any(f in COMMON_FIELDS['categorical'] for f in fields)
    quantitative_count = sum(1 for f in fields if isinstance(first_item[f], (int, float)))

    # Decision logic
    if has_temporal and quantitative_count >= 1:
        # Time series data -> line chart or area chart
        return "line chart"

    elif quantitative_count >= 2 and not has_temporal:
        # Multiple quantitative without time -> scatter plot
        return "scatter plot"

    elif has_categorical and quantitative_count >= 1:
        # Categorical comparison -> bar chart
        return "bar chart"

    else:
        # Default
        return "bar chart"


def format_chart_data(chart_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format and clean chart data.

    Args:
        chart_data: Raw chart data

    Returns:
        Cleaned chart data
    """
    if not chart_data:
        return []

    cleaned_data = []
    for item in chart_data:
        cleaned_item = {}
        for key, value in item.items():
            # Convert numeric strings to numbers
            if isinstance(value, str):
                try:
                    # Try integer first
                    cleaned_item[key] = int(value)
                except ValueError:
                    try:
                        # Try float
                        cleaned_item[key] = float(value)
                    except ValueError:
                        # Keep as string
                        cleaned_item[key] = value
            else:
                cleaned_item[key] = value

        cleaned_data.append(cleaned_item)

    return cleaned_data
