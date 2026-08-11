from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

from searchagent.agent.prompts import FINITE_LABEL_SYSTEM_PROMPT, QUERY_SYSTEM_PROMPT
from searchagent.config import LLMConfig


logger = logging.getLogger(__name__)
_MIN_REQUEST_GAP = float(os.environ.get("SEARCHAGENT_MIN_REQUEST_GAP", "0") or 0)
_throttle_lock = threading.Lock()
_manifest_lock = threading.Lock()
_last_request_ts = [0.0]


def _throttle() -> None:
    if _MIN_REQUEST_GAP <= 0:
        return
    with _throttle_lock:
        wait = _last_request_ts[0] + _MIN_REQUEST_GAP - _time.monotonic()
        if wait > 0:
            _time.sleep(wait)
        _last_request_ts[0] = _time.monotonic()


def _rotated(labels: List[str], key: str) -> List[str]:
    if not labels:
        return []
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    shift = int.from_bytes(digest[:8], "big") % len(labels)
    return labels[shift:] + labels[:shift]


class LLMBackbone:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._loaded = False
        self.tokenizer = None
        self.model = None
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_url = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
        self.manifest_path = os.environ.get("SEARCHAGENT_RUN_MANIFEST", "")
        self.request_records: List[dict] = []
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def load(self) -> None:
        if not self._loaded:
            self._loaded = True

    def _post(self, payload: dict) -> dict:
        _throttle()
        response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"API status {response.status_code}: {response.text[:500]}")
        return response.json()

    @staticmethod
    def _message_text(response_json: dict) -> str:
        choice = response_json.get("choices", [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or msg.get("reasoning_content") or ""
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        return str(content).strip()

    @staticmethod
    def _parse_probability_json(text: str, labels: List[str], epsilon: float = 1e-8) -> Dict[str, float]:
        obj = json.loads(text)
        if not isinstance(obj, dict) or set(obj.keys()) != {"probabilities"}:
            raise ValueError("invalid probability object")
        probs = obj["probabilities"]
        if not isinstance(probs, dict) or set(probs.keys()) != set(labels):
            raise ValueError("invalid probability labels")
        values = []
        for label in labels:
            value = probs[label]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("invalid probability value")
            values.append(max(float(value), 0.0) + epsilon)
        arr = np.asarray(values, dtype=np.float64)
        arr /= arr.sum()
        return {label: float(arr[i]) for i, label in enumerate(labels)}

    def _record(self, payload: dict, response_json: Optional[dict], prompt: str, retry_count: int, parser_status: str) -> None:
        response_text = "" if response_json is None else json.dumps(response_json, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        record = {
            "requested_model": payload.get("model", ""),
            "returned_model": "" if response_json is None else response_json.get("model", ""),
            "request_id": "" if response_json is None else response_json.get("id", ""),
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "canonical_prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_hash": hashlib.sha256(response_text.encode("utf-8")).hexdigest() if response_text else "",
            "usage": {} if response_json is None else response_json.get("usage", {}),
            "retry_count": retry_count,
            "parser_status": parser_status,
            "request_body": payload,
        }
        self.request_records.append(record)
        if self.manifest_path:
            path = Path(self.manifest_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _manifest_lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def get_action_logprobs(self, prompt: str, action_labels: List[str], rotation_key: Optional[str] = None) -> Dict[str, float]:
        self.load()
        labels = list(action_labels)
        if not labels:
            return {}
        order = _rotated(labels, rotation_key or prompt)
        for attempt in range(2):
            content = prompt if attempt == 0 else prompt + "\nFormat reminder: return only the required JSON object with every allowed label exactly once."
            payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": FINITE_LABEL_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "max_tokens": 256,
                "temperature": 0,
                "top_p": 1,
                "reasoning_effort": self.config.reasoning_effort,
                "response_format": {"type": "json_object"},
            }
            response_json = None
            try:
                response_json = self._post(payload)
                text = self._message_text(response_json)
                parsed = self._parse_probability_json(text, labels)
                self._record(payload, response_json, content, attempt, "ok")
                return parsed
            except Exception as exc:
                self._record(payload, response_json, content, attempt, f"error:{type(exc).__name__}")
                logger.warning("probability request failed on attempt %d: %s", attempt + 1, exc)
        logger.error("probability_parse_failure=true")
        uniform = 1.0 / len(labels)
        return {label: uniform for label in labels}

    def generate_query(self, prompt: str, current_query: str) -> str:
        self.load()
        for attempt in range(2):
            content = prompt if attempt == 0 else prompt + "\nFormat reminder: return exactly one JSON object with the sole key query."
            payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "max_tokens": 64,
                "temperature": 0,
                "top_p": 1,
                "reasoning_effort": self.config.reasoning_effort,
                "response_format": {"type": "json_object"},
            }
            response_json = None
            try:
                response_json = self._post(payload)
                obj = json.loads(self._message_text(response_json))
                if not isinstance(obj, dict) or set(obj.keys()) != {"query"}:
                    raise ValueError("invalid query object")
                query = obj["query"]
                if not isinstance(query, str) or not query.strip() or "\n" in query or len(query.split()) > 32:
                    raise ValueError("invalid query value")
                self._record(payload, response_json, content, attempt, "ok")
                return query.strip()
            except Exception as exc:
                self._record(payload, response_json, content, attempt, f"error:{type(exc).__name__}")
                logger.warning("query request failed on attempt %d: %s", attempt + 1, exc)
        return current_query

    def generate_text(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        self.load()
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "reasoning_effort": self.config.reasoning_effort,
        }
        for attempt in range(2):
            response_json = None
            try:
                response_json = self._post(payload)
                text = self._message_text(response_json)
                self._record(payload, response_json, prompt, attempt, "ok")
                return text
            except Exception as exc:
                self._record(payload, response_json, prompt, attempt, f"error:{type(exc).__name__}")
                logger.warning("generation request failed on attempt %d: %s", attempt + 1, exc)
        return ""

    def build_chat_prompt(self, system: str, user: str) -> str:
        return (f"System: {system}\n\n" if system else "") + f"User: {user}"

    def unload(self) -> None:
        self._loaded = False
