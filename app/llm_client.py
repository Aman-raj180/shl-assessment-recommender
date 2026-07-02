"""
LLM client wrapper -- Groq (LLaMA 3.3 70B), OpenAI-compatible API.

Switched from Gemini to Groq because Groq's free tier uses a per-minute
request cap with no daily ceiling, and no billing/card is required. That
matters specifically because the automated evaluator will run many 8-turn
conversations back to back -- a provider with a *daily* free cap (e.g.
OpenRouter's free-tagged models) risks getting blocked mid-evaluation,
which would fail the "schema compliance on every response" hard eval
outright. Groq's per-minute cap resets continuously instead.

Groq's JSON mode (response_format={"type": "json_object"}) guarantees
syntactically valid JSON but, unlike Gemini's response_schema, does not
enforce a specific schema -- so the required field structure is spelled
out explicitly in the system prompt, and the response is defensively
validated/coerced in code before it's trusted.
"""
import os
import re
import json
import time
from groq import Groq
from groq import RateLimitError, APIError, APIConnectionError, APITimeoutError

MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Hard wall-clock budget for the whole generate_decision() call, including
# all retries. The assignment caps each /chat call at 30s; we leave a wide
# safety margin for FastAPI/network overhead and the retrieval step that
# runs before this is called, so retries never cause us to blow the caller's
# timeout -- if we can't get a good answer inside this budget we fail fast
# and let the agent's fallback take over instead.
DEADLINE_SECONDS = 18
PER_REQUEST_TIMEOUT = 10  # Groq client-level timeout per individual attempt
MAX_BACKOFF_SECONDS = 4

_client = None


class LLMUnavailableError(Exception):
    """Raised when the LLM could not be reached after retries (rate limit, 5xx, network)."""


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Export it or add it to your .env / "
                "hosting provider's environment variables."
            )
        _client = Groq(api_key=api_key, timeout=PER_REQUEST_TIMEOUT)
    return _client


JSON_SCHEMA_INSTRUCTIONS = """
Respond with ONLY this JSON object, no markdown fences, no extra text:
{"action": "clarify"|"recommend"|"refine"|"compare"|"refuse", "reply": "...", "recommended_names": ["exact name", ...] (0-10 items), "end_of_conversation": true|false}
"""


def _coerce_decision(raw: dict) -> dict:
    action = raw.get("action")
    if action not in ("clarify", "recommend", "refine", "compare", "refuse"):
        action = "clarify"
    reply = raw.get("reply")
    if not isinstance(reply, str):
        reply = ""
    names = raw.get("recommended_names")
    if not isinstance(names, list):
        names = []
    names = [n for n in names if isinstance(n, str)][:10]
    eoc = raw.get("end_of_conversation")
    if not isinstance(eoc, bool):
        eoc = False
    return {
        "action": action,
        "reply": reply,
        "recommended_names": names,
        "end_of_conversation": eoc,
    }


def _mock_decision(user_prompt: str) -> dict:
    """
    Deterministic stand-in used only when MOCK_LLM=1, so the retrieval ->
    prompt -> schema -> FastAPI wiring can be smoke-tested end-to-end
    without hitting a real API (e.g. this sandbox, which cannot reach
    api.groq.com either). Not a substitute for real quality testing.
    """
    names = re.findall(r"^- (.+?) \[", user_prompt, re.MULTILINE)
    if "this is user turn 1" in user_prompt:
        return _coerce_decision({
            "action": "clarify",
            "reply": "[MOCK] Could you tell me more about the role and what the assessment should measure?",
            "recommended_names": [],
            "end_of_conversation": False,
        })
    return _coerce_decision({
        "action": "recommend",
        "reply": "[MOCK] Here is a shortlist based on what you've shared.",
        "recommended_names": names[:3],
        "end_of_conversation": False,
    })


def _extract_json(text: str) -> dict:
    text = text.strip()
    # strip markdown fences if the model added them despite instructions
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def generate_decision(system_prompt: str, user_prompt: str) -> dict:
    if os.environ.get("MOCK_LLM") == "1":
        return _mock_decision(user_prompt)

    client = get_client()
    full_system_prompt = system_prompt + "\n\n" + JSON_SCHEMA_INSTRUCTIONS
    last_error = None
    start_time = time.monotonic()
    attempt = 0

    while time.monotonic() - start_time < DEADLINE_SECONDS:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            raw = _extract_json(response.choices[0].message.content)
            return _coerce_decision(raw)

        except RateLimitError as e:
            last_error = e
            backoff = _retry_delay_from_error(e)
        except (APIConnectionError, APITimeoutError, APIError) as e:
            last_error = e
            backoff = 1.5
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            backoff = 0.5

        elapsed = time.monotonic() - start_time
        remaining = DEADLINE_SECONDS - elapsed
        if remaining <= 0.5:
            break
        time.sleep(min(backoff, remaining - 0.5, MAX_BACKOFF_SECONDS))

    raise LLMUnavailableError(f"{last_error} (gave up after {attempt} attempt(s), "
                               f"{time.monotonic() - start_time:.1f}s elapsed)")


def _retry_delay_from_error(e: RateLimitError, default: float = 3.0) -> float:
    # Groq returns rate-limit headers on 429s; prefer the token-reset window
    # since TPM (not RPM) is the binding constraint for this workload.
    try:
        headers = getattr(e, "response", None).headers  # type: ignore[union-attr]
        for header_name in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
            val = headers.get(header_name)
            if val:
                match = re.match(r"([\d.]+)", val)
                if match:
                    return min(float(match.group(1)), MAX_BACKOFF_SECONDS)
    except Exception:
        pass
    return default
