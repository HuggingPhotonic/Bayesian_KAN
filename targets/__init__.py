"""
Target function registry and utilities.
"""

from .functions import (
    TARGETS,
    TARGET_DIMS,
    get_target_function,
    resolve_target,
    get_function_info,
    list_functions,
    generate_data,
)

__all__ = [
    "TARGETS",
    "TARGET_DIMS",
    "get_target_function",
    "resolve_target",
    "get_function_info",
    "list_functions",
    "generate_data",
]
