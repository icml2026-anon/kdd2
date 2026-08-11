"""Test fixtures. Sets a dummy API key so LLMBackbone can be constructed in
unit tests without a real key. Tests here do not make real API calls."""
import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def _dummy_api_key():
    os.environ.setdefault("DEEPSEEK_API_KEY", "test-dummy-key-not-real")
    yield
