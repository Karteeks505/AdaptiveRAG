from __future__ import annotations

import json
from typing import Any

import httpx


def ollama_generate(base_url: str, model: str, prompt: str, temperature: float) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return str(data.get("response", ""))


def ollama_judge_compliance(
    base_url: str,
    model: str,
    question: str,
    answer: str,
    gold_v1: str,
    temperature: float,
) -> bool:
    prompt = (
        "You are a strict compliance reviewer. Reply with JSON only: {\"aligned\": true|false}.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Authoritative v1 reference: {gold_v1}\n"
        "Aligned means the answer is consistent with the reference for regulatory/policy wording."
    )
    raw = ollama_generate(base_url, model, prompt, temperature)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        obj = json.loads(raw[start:end])
        return bool(obj.get("aligned"))
    except Exception:
        return False
