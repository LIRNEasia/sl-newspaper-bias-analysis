"""Shared utility for loading LLM prompt templates from the prompts/ directory."""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROMPTS_DIR = os.path.join(_PROJECT_ROOT, "prompts")


def load_prompt(path: str, **kwargs) -> str:
    """Load a prompt template and substitute {{variable}} placeholders.

    Args:
        path: Relative path within the prompts/ directory (e.g. "ditwah/stance_system.md")
        **kwargs: Variable substitutions; each key maps to {{key}} in the template.

    Returns:
        Rendered prompt string.
    """
    full_path = os.path.join(_PROMPTS_DIR, path)
    with open(full_path, "r") as f:
        template = f.read()
    for key, value in kwargs.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template
