import os
import jinja2


def load_prompt(prompt_name: str, prompt_version: str) -> jinja2.Template:
    """Load a Jinja2 prompt template from the local prompt_library.

    Args:
        prompt_name: Directory name under `prompt_library/` (e.g. "user_simulator").
        prompt_version: File version name (e.g. "v0.0.1").

    Returns:
        A compiled Jinja2 Template.
    """

    prompt_path = f"../prompt_library/{prompt_name}/{prompt_version}.jinja"
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt {prompt_path} not found")

    with open(prompt_path, "r") as f:
        return jinja2.Template(f.read())
