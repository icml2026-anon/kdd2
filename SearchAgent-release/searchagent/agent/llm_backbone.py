"""LLM backbone for SearchAgent using DeepSeek API."""

from __future__ import annotations

import logging
import os
import threading
import time as _time
import requests
import numpy as np
from typing import Dict, List, Optional, Tuple

from searchagent.config import LLMConfig

logger = logging.getLogger(__name__)

# Process-wide request throttle. The proxy 429s on bursty traffic but serves
# spaced-out calls fine (measured: 5/5 OK at 10s spacing, 0 at full speed).
# A min gap between *starts* of API calls smooths the burst. Tunable via
# SEARCHAGENT_MIN_REQUEST_GAP (seconds); 0 disables.
_MIN_REQUEST_GAP = float(os.environ.get("SEARCHAGENT_MIN_REQUEST_GAP", "0") or 0)
_throttle_lock = threading.Lock()
_last_request_ts = [0.0]


def _throttle():
    if _MIN_REQUEST_GAP <= 0:
        return
    with _throttle_lock:
        wait = _last_request_ts[0] + _MIN_REQUEST_GAP - _time.monotonic()
        if wait > 0:
            _time.sleep(wait)
        _last_request_ts[0] = _time.monotonic()


class LLMBackbone:
    """Manages the DeepSeek API client for agent decision-making and text generation.

    Supports:
        - Forced-choice action selection via token logprobs
        - Free-form text generation (query generation, reasoning)
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._loaded = False
        self.tokenizer = None
        self.model = None
        # API credentials are read from environment variables. Set DEEPSEEK_API_KEY
        # (and optionally DEEPSEEK_API_URL) before running; no key is hardcoded.
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_url = os.environ.get(
            "DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is not set. "
                "Export it before running, e.g. `export DEEPSEEK_API_KEY=sk-...`."
            )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def load(self) -> None:
        """Load API client (no-op, but kept for backward compatibility)."""
        if self._loaded:
            return
        logger.info("Initializing DeepSeek API backbone...")
        self._loaded = True

    def get_action_logprobs(
        self, prompt: str, action_labels: List[str]
    ) -> Dict[str, float]:
        """Extract log-probabilities for each action label token.

        Args:
            prompt: The full prompt ending with a cue for the model to choose.
            action_labels: List of single-token labels (e.g., ["A", "B", "C"]).

        Returns:
            Dict mapping each label to its normalized probability.
        """
        self.load()
        payload = {
            "model": getattr(self.config, "model_name", "deepseek-v4-flash"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 5,
            "temperature": getattr(self.config, "temperature", 0.1),
            "thinking": {"type": "disabled"},
        }
        
        # Default very low logprob for each label
        logprobs_dict = {label: -20.0 for label in action_labels}
        
        import time
        max_retries = 8
        backoff = 2.0
        
        for attempt in range(max_retries):
            try:
                _throttle()
                response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=30)
                if response.status_code == 200:
                    response_json = response.json()
                    # Proxy intermittently returns logprobs: null (explicit None,
                    # not missing) — `.get(k, {})` returns None in that case, so
                    # coerce to {} to avoid 'NoneType' has no attribute 'get'.
                    choice = response_json["choices"][0]
                    logprobs_data = choice.get("logprobs") or {}
                    content_logprobs = logprobs_data.get("content") or logprobs_data.get("reasoning_content")
                    if not content_logprobs:
                        logger.error(f"Failed to parse logprobs. Response: {response_json}")
                        raise KeyError("content")
                    if content_logprobs:
                        top_logprobs = content_logprobs[0].get("top_logprobs", [])
                        for item in top_logprobs:
                            token = item.get("token", "").strip()
                            logprob = item.get("logprob", -20.0)
                            for label in action_labels:
                                if token.upper() == label.upper() or token == label:
                                    logprobs_dict[label] = max(logprobs_dict[label], logprob)
                    break
                elif response.status_code in [429, 500, 502, 503, 504]:
                    logger.warning(f"DeepSeek API returned status {response.status_code} on attempt {attempt+1}. Retrying in {backoff:.1f}s...")
                else:
                    logger.error(f"DeepSeek API returned unexpected status {response.status_code}: {response.text}")
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error("DeepSeek API logprobs failed after %d retries: %s", max_retries, e)
                    break
                logger.warning(f"DeepSeek API connection error on attempt {attempt+1}: {e}. Retrying in {backoff:.1f}s...")
            
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 8.0)

        # Convert logprobs to normalized probabilities (softmax)
        log_probs_array = np.array([logprobs_dict[label] for label in action_labels])
        probs = np.exp(log_probs_array - np.max(log_probs_array))
        probs /= probs.sum()
        
        return {label: float(probs[i]) for i, label in enumerate(action_labels)}

    def generate_text(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate free-form text from a prompt.

        Used for query generation, reasoning, and reformulation.
        """
        self.load()
        payload = {
            "model": getattr(self.config, "model_name", "deepseek-v4-flash"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens or getattr(self.config, "max_new_tokens", 128),
            "temperature": getattr(self.config, "temperature", 0.7),
            "top_p": getattr(self.config, "top_p", 0.95),
            "thinking": {"type": "disabled"},  # flash/pro are thinking models;
            # without this, output goes to reasoning_content and content is empty.
        }
        
        import time
        max_retries = 8
        backoff = 2.0
        
        for attempt in range(max_retries):
            try:
                _throttle()
                response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=30)
                if response.status_code == 200:
                    msg = response.json()["choices"][0].get("message") or {}
                    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
                    return text
                elif response.status_code in [429, 500, 502, 503, 504]:
                    logger.warning(f"DeepSeek API returned status {response.status_code} on attempt {attempt+1}. Retrying in {backoff:.1f}s...")
                else:
                    logger.error(f"DeepSeek API returned unexpected status {response.status_code}: {response.text}")
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error("DeepSeek API generation failed after %d retries: %s", max_retries, e)
                    break
                logger.warning(f"DeepSeek API connection error on attempt {attempt+1}: {e}. Retrying in {backoff:.1f}s...")
            
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 8.0)
            
        return ""

    def build_chat_prompt(self, system: str, user: str) -> str:
        """Build a chat-formatted prompt (kept for backward compatibility)."""
        prompt = ""
        if system:
            prompt += f"System: {system}\n\n"
        prompt += f"User: {user}"
        return prompt

    def unload(self) -> None:
        """Free memory (no-op)."""
        self._loaded = False
        logger.info("DeepSeek API backbone unloaded.")
