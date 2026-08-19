import os
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]


def run_case(env_updates):
    code = """
from unittest.mock import patch
from app.core import llm
llm.get_llm.cache_clear()
with patch("app.core.llm.ChatGroq") as groq, patch("app.core.llm.ChatOpenAI") as openai:
    model = llm.get_llm()
    print(type(model).__name__)
    if groq.called:
        print(groq.call_args.kwargs.get('model'))
    if openai.called:
        print(openai.call_args.kwargs.get('model'))
"""
    env = os.environ.copy()
    env.update(env_updates)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip().splitlines()


groq_case = run_case({
    "LLM_PROVIDER": "auto",
    "GROQ_API_KEY": "groq-test-key",
    "GROQ_MODEL": "openai/gpt-oss-120b",
    "OPENAI_API_KEY": "openai-test-key",
})
assert groq_case == ["MagicMock", "openai/gpt-oss-120b"], groq_case

openai_case = run_case({
    "LLM_PROVIDER": "openai",
    "GROQ_API_KEY": "",
    "OPENAI_API_KEY": "openai-test-key",
    "OPENAI_MODEL": "gpt-5-mini",
})
assert openai_case == ["MagicMock", "gpt-5-mini"], openai_case

print("llm_provider_selection=passed")
