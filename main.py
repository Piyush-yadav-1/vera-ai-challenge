import os
import json
import re
import uuid
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vera")

app = FastAPI(title="Vera AI Engine", version="2.0.0")

# --- OpenRouter client ---
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"
)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    default_headers={
        "HTTP-Referer": "http://localhost:8080",
        "X-Title": "Vera-Engine",
    },
)

# --- ATOMIC STATE CACHE ---
CONTEXT_STORE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "merchant": {},
    "category": {},
    "customer": {},
    "trigger": {},
}

# --- REQUEST SCHEMAS (Pydantic V2 — extra="allow" to tolerate unknown fields) ---
class ContextPayload(BaseModel):
    model_config = {"extra": "allow"}
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None

class TickPayload(BaseModel):
    model_config = {"extra": "allow"}
    now: Optional[str] = None
    available_triggers: Optional[List[str]] = []
    # Legacy single-trigger fields the old code accepted
    merchant_id: Optional[str] = None
    trigger: Optional[Dict[str, Any]] = None

class ReplyPayload(BaseModel):
    model_config = {"extra": "allow"}
    conversation_id: Optional[str] = None
    merchant_id: str = "default_merchant"
    customer_id: Optional[str] = None
    from_role: Optional[str] = "merchant"
    message: str = ""
    received_at: Optional[str] = None
    turn_number: Optional[int] = 1
    thread_history: Optional[list] = []

# --- AUTO-REPLY PATTERNS ---
AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"get back to you",
    r"currently closed",
    r"business hours",
    r"automated reply",
    r"auto-generated",
    r"auto-reply",
    r"out of office",
    r"away right now",
    r"currently unavailable",
    r"automated assistant",
    r"our team will respond",
]

# --- HOSTILE PATTERNS ---
HOSTILE_PATTERNS = [
    r"stop messaging",
    r"useless spam",
    r"don't contact",
    r"do not contact",
    r"stop sending",
    r"unsubscribe",
    r"leave me alone",
    r"not interested",
    r"block you",
    r"report spam",
]

# --- INTENT / COMMITMENT PATTERNS ---
INTENT_PATTERNS = [
    r"lets do it",
    r"let's do it",
    r"go ahead",
    r"yes.*do it",
    r"whats next",
    r"what's next",
    r"i want to join",
    r"sign me up",
    r"ok.*proceed",
    r"sounds good.*next",
    r"i'm in",
    r"im in",
    r"yes please",
    r"yes i want",
    r"mujhe.*judna",
    r"haan.*karo",
]

# --- SYSTEM PROMPT (tick — composition) ---
TICK_SYSTEM_PROMPT = """You are Vera, an AI growth assistant for Indian merchants on WhatsApp.
You compose concise, high-converting messages based on provided context.

CRITICAL RULES:
1. STRICT GROUNDING: Use ONLY exact catalog items, prices (₹), metrics, and data from the context. NEVER fabricate.
2. SINGLE CTA: Always end with ONE clear, low-friction call-to-action.
3. SPECIFICITY WINS: Anchor every message on a verifiable fact (number, date, source).
4. VOICE MATCH: Match the category voice (dentists=clinical/peer, salons=warm/practical, etc).
5. Hindi-English code-mix is fine and often preferred.

Respond ONLY with a JSON object (no markdown fences, no explanation) in this exact format:
{"body": "the WhatsApp message text", "cta": "open_ended" or "binary_yes_stop" or "none", "send_as": "vera" or "merchant_on_behalf", "suppression_key": "unique_key", "rationale": "1 sentence why"}"""

# --- SYSTEM PROMPT (reply — conversation handling) ---
REPLY_SYSTEM_PROMPT = """You are Vera, an AI growth assistant for Indian merchants on WhatsApp.
You are in a live conversation. Analyze the merchant's reply and decide the correct action.

ACTIONS:
- "send": Reply with a message. Use when the merchant is engaged, asks a question, or shows interest.
- "end": Gracefully exit. Use when the merchant is hostile, says stop, or is not interested.
- "wait": Back off temporarily. Use when the merchant needs time or hasn't responded meaningfully.

RULES:
1. If the merchant commits ("let's do it", "yes", "go ahead") → switch to ACTION mode. Don't re-qualify. Respond with concrete next steps: "Done. Sending the draft now..." / "Confirmed. Here's what happens next..."
2. If the merchant is hostile or says stop → end gracefully with a short apology.
3. Use ONLY facts from context. NEVER fabricate.
4. Keep responses concise for WhatsApp.

Respond ONLY with a JSON object (no markdown fences) in this exact format:
{"action": "send" or "end" or "wait", "body": "message text (empty if action is end/wait)", "cta": "open_ended" or "binary_yes_stop" or "none", "rationale": "1 sentence why"}"""


def get_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    # Strip markdown ```json ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    # Strip <think>...</think> blocks some models produce
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    # Find the first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)


def call_llm(system_prompt: str, user_prompt: str, retries: int = 2) -> str:
    """Call OpenRouter with retry logic for rate limits. Raises on failure."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("LLM returned empty content")
            return content
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited (attempt {attempt+1}/{retries+1}), waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"LLM call failed (attempt {attempt+1}): {e}")
                break
    raise RuntimeError(f"LLM call failed after {retries+1} attempts: {last_error}")


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/v1/healthz")
def healthz():
    return {
        "status": "ok",
        "timestamp": get_utc_now(),
        "uptime_seconds": 0,
        "contexts_loaded": {
            scope: len(items) for scope, items in CONTEXT_STORE.items()
        },
    }


@app.get("/v1/metadata")
def metadata():
    return {
        "team_name": "Vera-Engine-Pro",
        "team_members": ["Piyush Yadav"],
        "model": OPENROUTER_MODEL,
        "approach": "4-context fusion with LLM composition, rule-based routing for auto-reply/hostile/intent",
        "version": "2.0.0",
    }


@app.post("/v1/context")
def ingest_context(data: ContextPayload):
    scope = data.scope
    if scope not in CONTEXT_STORE:
        raise HTTPException(status_code=400, detail=f"invalid_scope: {scope}")

    current = CONTEXT_STORE[scope].get(data.context_id)
    if current and current["version"] >= data.version:
        return {"accepted": True, "status": "ignored_older", "current_version": current["version"]}

    CONTEXT_STORE[scope][data.context_id] = {
        "version": data.version,
        "payload": data.payload,
    }
    return {
        "accepted": True,
        "ack_id": f"ack_{uuid.uuid4().hex[:8]}",
        "stored_at": get_utc_now(),
    }


@app.post("/v1/tick")
def process_tick(data: TickPayload):
    """Compose proactive messages for available triggers."""
    actions = []

    # Get the list of trigger IDs to process
    trigger_ids = data.available_triggers or []

    # Fallback: legacy single-trigger format
    if not trigger_ids and data.merchant_id and data.trigger:
        return _compose_single_tick(data.merchant_id, data.trigger)

    for tid in trigger_ids:
        trigger_data = CONTEXT_STORE["trigger"].get(tid, {}).get("payload", {})
        if not trigger_data:
            continue

        merchant_id = trigger_data.get("merchant_id", "")
        merchant = CONTEXT_STORE["merchant"].get(merchant_id, {}).get("payload", {})
        cat_slug = merchant.get("category_slug") or trigger_data.get("category", "general")
        category = CONTEXT_STORE["category"].get(cat_slug, {}).get("payload", {})

        customer_id = trigger_data.get("customer_id")
        customer = CONTEXT_STORE["customer"].get(customer_id, {}).get("payload", {}) if customer_id else None

        fusion = {
            "category": category,
            "merchant": merchant,
            "trigger": trigger_data,
        }
        if customer:
            fusion["customer"] = customer

        try:
            raw = call_llm(TICK_SYSTEM_PROMPT, f"Compose a message for this context:\n{json.dumps(fusion, default=str)}")
            parsed = extract_json(raw)
            actions.append({
                "conversation_id": f"conv_{uuid.uuid4().hex[:8]}",
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "trigger_id": tid,
                "send_as": parsed.get("send_as", "vera"),
                "body": parsed.get("body", ""),
                "cta": parsed.get("cta", "none"),
                "suppression_key": parsed.get("suppression_key", f"{tid}:sent"),
                "rationale": parsed.get("rationale", ""),
            })
        except Exception as e:
            logger.error(f"Tick composition failed for trigger {tid}: {e}")
            # Skip this trigger rather than crashing the whole tick
            continue

    return {"actions": actions}


def _compose_single_tick(merchant_id: str, trigger: Dict[str, Any]) -> dict:
    """Legacy handler for the old single-trigger tick format."""
    merchant = CONTEXT_STORE["merchant"].get(merchant_id, {}).get("payload", {})
    cat_slug = merchant.get("category_slug") or merchant.get("category", "general")
    category = CONTEXT_STORE["category"].get(cat_slug, {}).get("payload", {})
    fusion = {"merchant": merchant, "category": category, "trigger": trigger}

    try:
        raw = call_llm(TICK_SYSTEM_PROMPT, f"Compose a message for this context:\n{json.dumps(fusion, default=str)}")
        parsed = extract_json(raw)
        return {
            "actions": [{
                "conversation_id": f"conv_{uuid.uuid4().hex[:8]}",
                "merchant_id": merchant_id,
                "customer_id": None,
                "send_as": parsed.get("send_as", "vera"),
                "body": parsed.get("body", ""),
                "cta": parsed.get("cta", "none"),
                "suppression_key": parsed.get("suppression_key", f"{merchant_id}:tick"),
                "rationale": parsed.get("rationale", ""),
            }],
        }
    except Exception as e:
        logger.error(f"Single tick composition failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM composition failed: {e}")


@app.post("/v1/reply")
def process_reply(data: ReplyPayload):
    """Handle an incoming merchant/customer reply."""
    logger.info(f"[REPLY] merchant={data.merchant_id} turn={data.turn_number} message=\"{data.message[:80]}\"")
    text = data.message.lower().strip()

    # --- RULE 1: Auto-reply detection → end immediately ---
    if any(re.search(p, text) for p in AUTO_REPLY_PATTERNS):
        logger.info("[RULE] Auto-reply detected → ending conversation")
        return {
            "action": "end",
            "body": "",
            "rationale": "Auto-reply detected. Ending to prevent loop.",
        }

    # --- RULE 2: Hostile detection → end gracefully ---
    if any(re.search(p, text) for p in HOSTILE_PATTERNS):
        logger.info("[RULE] Hostile message detected → ending gracefully")
        return {
            "action": "end",
            "body": "Sorry for the inconvenience. I won't message you again. Best wishes!",
            "rationale": "Hostile message detected. Graceful exit.",
        }

    # --- RULE 3: Intent / commitment detected → action mode ---
    if any(re.search(p, text) for p in INTENT_PATTERNS):
        logger.info("[RULE] Intent commitment detected → action mode")
        merchant = CONTEXT_STORE["merchant"].get(data.merchant_id, {}).get("payload", {})
        merchant_name = merchant.get("identity", {}).get("name", "")

        return {
            "action": "send",
            "body": f"Done! Sending the next steps now. Here's what happens next — I'll draft everything and share it with you for confirmation. No extra effort needed from your side.",
            "cta": "none",
            "rationale": "Merchant committed. Switched to action mode with concrete next steps.",
        }

    # --- RULE 4: LLM-powered reply ---
    merchant = CONTEXT_STORE["merchant"].get(data.merchant_id, {}).get("payload", {})
    cat_slug = merchant.get("category_slug", "general")
    category = CONTEXT_STORE["category"].get(cat_slug, {}).get("payload", {})

    prompt_context = {
        "merchant": merchant,
        "category_voice": category.get("voice", {}),
        "incoming_message": data.message,
        "turn_number": data.turn_number,
        "from_role": data.from_role,
    }

    try:
        raw = call_llm(
            REPLY_SYSTEM_PROMPT,
            f"Handle this conversation turn:\n{json.dumps(prompt_context, default=str)}",
        )
        parsed = extract_json(raw)

        # Validate the action field
        action = parsed.get("action", "send")
        if action not in ("send", "end", "wait"):
            action = "send"

        result = {
            "action": action,
            "body": parsed.get("body", ""),
            "rationale": parsed.get("rationale", ""),
        }
        if action == "send":
            result["cta"] = parsed.get("cta", "none")
        if action == "wait":
            result["wait_seconds"] = parsed.get("wait_seconds", 1800)

        return result
    except Exception as e:
        logger.error(f"Reply LLM call failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed: {e}. Cannot produce a valid response.",
        )
