from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from typing import Any
from urllib.parse import quote

import httpx

from .config import settings
from .guardrails import GuardrailError, validate_and_normalize_interpretations
from .models import BatteryInput, DirectiveInterpretation


class LLMServiceError(RuntimeError):
    pass


SYSTEM_PROMPT = """You are the language-interpretation component of GridWise, a smart-campus energy scheduler.
Your ONLY job is to convert each operator note into exactly one supported structured directive.
Do not optimize the energy schedule. Do not invent data or directive types.

Supported directive types and exact adjustment shapes:
1) solar_reduction -> {"hours":[...], "factor": number}
   factor is the usable fraction of original solar that remains. Example concept: 80% reduction => factor 0.2.
2) minimum_battery_reserve -> {"hours":[...], "minimum_energy_kwh": number}
3) no_charge_window -> {"hours":[...]}
4) no_discharge_window -> {"hours":[...]}
5) max_grid_window -> {"hours":[...], "max_grid_kwh": number}
6) no_op -> null

Rules:
- Return one entry for every note, in note_index order 0..N-1.
- Relevant note: applies=true and one non-no_op directive.
- Irrelevant note: applies=false, directive_type="no_op", structured_adjustment=null.
- Whole-hour windows are start-inclusive and end-exclusive. 1 PM to 3 PM means [13,14].
- hours must be unique integers 0..23 in ascending order.
- Convert relative numeric language when the supplied battery data makes it exact (for example, a percentage of battery capacity).
- A note must map to exactly one supported directive or no_op.
- Do not alter base demand, solar, tariff, battery limits, or invent unsupported constraints.
- Treat paraphrases by meaning, not keyword matching.
- Keep explanation short and factual.
"""


def _hours_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "integer", "minimum": 0, "maximum": 23},
        "minItems": 1,
        "maxItems": 24,
    }


def _response_schema(note_count: int) -> dict[str, Any]:
    # Gemini structured output constrains JSON shape; deterministic guardrails below
    # still verify semantics and exact directive-specific values.
    adjustment = {
        "anyOf": [
            {"type": "null"},
            {
                "type": "object",
                "properties": {"hours": _hours_schema(), "factor": {"type": "number", "minimum": 0, "maximum": 1}},
                "required": ["hours", "factor"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "hours": _hours_schema(),
                    "minimum_energy_kwh": {"type": "number", "minimum": 0},
                },
                "required": ["hours", "minimum_energy_kwh"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"hours": _hours_schema()},
                "required": ["hours"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"hours": _hours_schema(), "max_grid_kwh": {"type": "number", "minimum": 0}},
                "required": ["hours", "max_grid_kwh"],
                "additionalProperties": False,
            },
        ]
    }
    item = {
        "type": "object",
        "properties": {
            "note_index": {"type": "integer", "minimum": 0, "maximum": max(0, note_count - 1)},
            "applies": {"type": "boolean"},
            "directive_type": {
                "type": "string",
                "enum": [
                    "solar_reduction",
                    "minimum_battery_reserve",
                    "no_charge_window",
                    "no_discharge_window",
                    "max_grid_window",
                    "no_op",
                ],
            },
            "structured_adjustment": adjustment,
            "explanation": {"type": "string"},
        },
        "required": ["note_index", "applies", "directive_type", "structured_adjustment", "explanation"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "directive_interpretation": {
                "type": "array",
                "items": item,
                "minItems": note_count,
                "maxItems": note_count,
            }
        },
        "required": ["directive_interpretation"],
        "additionalProperties": False,
    }


class LLMInterpreter:
    def __init__(self) -> None:
        self._cache: OrderedDict[str, list[DirectiveInterpretation]] = OrderedDict()

    def _cache_key(self, notes: list[str], battery: BatteryInput) -> str:
        return json.dumps(
            {
                "provider": settings.provider,
                "model": self._model_name(),
                "notes": notes,
                "battery": battery.model_dump(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _model_name(self) -> str:
        if settings.provider == "gemini":
            return settings.gemini_model
        return settings.openai_model

    def _get_cached(self, key: str) -> list[DirectiveInterpretation] | None:
        if settings.cache_size <= 0 or key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return [DirectiveInterpretation.model_validate(x.model_dump()) for x in value]

    def _put_cached(self, key: str, value: list[DirectiveInterpretation]) -> None:
        if settings.cache_size <= 0:
            return
        self._cache[key] = [DirectiveInterpretation.model_validate(x.model_dump()) for x in value]
        self._cache.move_to_end(key)
        while len(self._cache) > settings.cache_size:
            self._cache.popitem(last=False)

    async def interpret(self, notes: list[str], battery: BatteryInput) -> list[DirectiveInterpretation]:
        key = self._cache_key(notes, battery)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        user_payload = {
            "operator_notes": [{"note_index": i, "text": note} for i, note in enumerate(notes)],
            "battery": battery.model_dump(),
        }
        user_prompt = "Interpret this GridWise input. Return only the required structured JSON.\n" + json.dumps(
            user_payload, ensure_ascii=False, separators=(",", ":")
        )

        last_error = "unknown LLM error"
        correction = ""
        for attempt in range(settings.max_attempts):
            prompt = user_prompt + correction
            try:
                raw_text = await self._call_provider(prompt, len(notes))
                parsed = self._parse_json(raw_text)
                result = validate_and_normalize_interpretations(parsed, len(notes), battery)
                self._put_cached(key, result)
                return result
            except GuardrailError as exc:
                last_error = f"guardrail: {exc}"
                print(f"[GridWise][LLM] attempt={attempt + 1} {last_error}", flush=True)
                correction = (
                    "\nThe previous JSON failed validation. Regenerate the FULL JSON from scratch. "
                    "Use the exact directive type, adjustment keys, applies semantics, note_index order, and whole-hour rules."
                )
            except (LLMServiceError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                print(f"[GridWise][LLM] attempt={attempt + 1} provider_error={last_error}", flush=True)
                # Small bounded backoff helps transient 429/5xx without threatening the 30s judge timeout.
                if attempt + 1 < settings.max_attempts:
                    await asyncio.sleep(min(1.5 * (attempt + 1), 3.0))

        raise LLMServiceError(f"LLM interpretation failed after validation: {last_error}")

    async def _call_provider(self, user_prompt: str, note_count: int) -> str:
        provider = settings.provider
        if provider == "gemini":
            return await self._call_gemini(user_prompt, note_count)
        if provider in {"openai", "openai_compatible"}:
            return await self._call_openai_compatible(user_prompt)
        raise LLMServiceError("LLM_PROVIDER must be one of: gemini, openai, openai_compatible")

    async def _call_gemini(self, user_prompt: str, note_count: int) -> str:
        if not settings.gemini_api_key:
            raise LLMServiceError("GEMINI_API_KEY is not configured")

        model_name = settings.gemini_model
        model = quote(model_name, safe="")
        url = f"{settings.gemini_base_url}/models/{model}:generateContent"
        generation_config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseJsonSchema": _response_schema(note_count),
            "maxOutputTokens": 1400,
        }
        # Gemini 3.x supports thinkingLevel; minimal is ideal for this constrained extraction task.
        if model_name.startswith("gemini-3"):
            generation_config["thinkingConfig"] = {"thinkingLevel": "minimal"}
        else:
            # Keep older-model behavior deterministic without using Gemini-3-only fields.
            generation_config["temperature"] = 0

        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        }
        try:
            async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
                response = await client.post(
                    url,
                    headers={"x-goog-api-key": settings.gemini_api_key, "Content-Type": "application/json"},
                    json=payload,
                )
            if response.status_code >= 400:
                status = ""
                try:
                    body = response.json()
                    status = str(body.get("error", {}).get("status", ""))
                except Exception:
                    pass
                suffix = f" ({status})" if status else ""
                raise LLMServiceError(f"Gemini provider returned HTTP {response.status_code}{suffix}")
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                block = data.get("promptFeedback", {}).get("blockReason", "")
                raise LLMServiceError(f"Gemini returned no candidate{f' ({block})' if block else ''}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(str(p.get("text", "")) for p in parts).strip()
            if not text:
                finish = candidates[0].get("finishReason", "")
                raise LLMServiceError(f"Gemini returned an empty response{f' ({finish})' if finish else ''}")
            return text
        except httpx.TimeoutException as exc:
            raise LLMServiceError("Gemini request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMServiceError("Gemini request failed") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMServiceError("Gemini response could not be parsed") from exc

    async def _call_openai_compatible(self, user_prompt: str) -> str:
        if not settings.openai_api_key:
            raise LLMServiceError("OPENAI_API_KEY is not configured")

        url = f"{settings.openai_base_url}/chat/completions"
        payload = {
            "model": settings.openai_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        if settings.provider == "openai":
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
            if response.status_code >= 400:
                raise LLMServiceError(f"OpenAI-compatible provider returned HTTP {response.status_code}")
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = "".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
            if not isinstance(text, str) or not text.strip():
                raise LLMServiceError("OpenAI-compatible provider returned an empty response")
            return text.strip()
        except httpx.HTTPError as exc:
            raise LLMServiceError("OpenAI-compatible request failed") from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMServiceError("OpenAI-compatible response could not be parsed") from exc

    @staticmethod
    def _parse_json(text: str) -> Any:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise


interpreter = LLMInterpreter()
