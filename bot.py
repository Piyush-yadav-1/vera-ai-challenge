"""
Vera AI Engine — Standalone Composition Module
================================================
Required by magicpin AI Challenge §7.1.

Usage:
    from bot import compose
    result = compose(category_dict, merchant_dict, trigger_dict, customer_dict_or_none)

Returns:
    dict with keys: body, cta, send_as, suppression_key, rationale
"""

import os
import re
import json
import time
from openai import OpenAI

# --- LLM Client ---
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    default_headers={
        "HTTP-Referer": "http://localhost:8080",
        "X-Title": "Vera-Engine",
    },
)


# --- Trigger-specific prompt routing ---
TRIGGER_PROMPTS = {
    "research_digest": """The trigger is a RESEARCH DIGEST — a new study/publication relevant to this category.
Your message should:
- Lead with the specific finding (cite source, trial size, percentage)
- Connect it to THIS merchant's patient/customer base
- Offer to share the abstract or draft patient-facing content
- Tone: peer/colleague sharing a clinical update, NOT promotional""",

    "regulation_change": """The trigger is a REGULATION CHANGE — a compliance deadline.
Your message should:
- State the regulation clearly with the deadline date
- Explain what the merchant needs to do
- Offer to help with compliance steps
- Tone: urgent but helpful, not alarming""",

    "recall_due": """The trigger is a RECALL/APPOINTMENT DUE for a specific customer.
Your message should be sent ON BEHALF OF THE MERCHANT to the customer:
- Use the customer's name
- Reference their specific service history
- Offer specific available time slots from the trigger data
- Include the service price from merchant's active offers
- send_as must be "merchant_on_behalf"
- Match customer's language preference""",

    "perf_dip": """The trigger is a PERFORMANCE DIP — the merchant's metrics dropped.
Your message should:
- State the specific metric and percentage drop
- Suggest a concrete, actionable fix (not generic advice)
- Reference what's working (if any positive signals exist)
- Use loss aversion framing""",

    "perf_spike": """The trigger is a PERFORMANCE SPIKE — something is working well.
Your message should:
- Celebrate the specific metric improvement
- Suggest how to capitalize on the momentum
- Be brief and encouraging""",

    "renewal_due": """The trigger is a SUBSCRIPTION RENEWAL — plan is expiring soon.
Your message should:
- State exactly how many days remain
- Remind them what they'd lose (specific features/metrics)
- Make the renewal frictionless
- NOT be pushy — frame as a heads-up""",

    "festival_upcoming": """The trigger is an UPCOMING FESTIVAL.
Your message should:
- Connect the festival to the merchant's specific category
- Suggest a concrete promotional action (not just "run a sale")
- Reference their existing offers if applicable
- Be timely and actionable""",

    "milestone_reached": """The trigger is a MILESTONE — the merchant hit a number worth celebrating.
Your message should:
- Celebrate the specific milestone
- Suggest how to leverage it (e.g., share on social media, Google post)
- Be brief and genuinely congratulatory""",

    "review_theme_emerged": """The trigger is an emerging REVIEW THEME — customers are mentioning something repeatedly.
Your message should:
- Quote the specific theme and count
- Suggest a concrete response strategy
- Frame as an opportunity, not a criticism""",

    "competitor_opened": """The trigger is a NEW COMPETITOR nearby.
Your message should:
- State the competitor name and distance
- Compare their offering to the merchant's
- Suggest a differentiation or competitive response
- Use social proof and the merchant's strengths""",

    "active_planning_intent": """The trigger is an ACTIVE PLANNING intent — the merchant wants to build something new.
Your message should:
- Directly address their stated intent
- Provide a concrete, structured plan/draft
- Include specific numbers (pricing, schedule, format)
- Be action-oriented — they already said yes""",

    "supply_alert": """The trigger is a SUPPLY/SAFETY ALERT — a product recall or stock issue.
Your message should:
- State the specific alert (molecule, batch numbers)
- Suggest immediate action steps
- Offer to help filter affected customers
- Tone: urgent, precise, trustworthy""",

    "chronic_refill_due": """The trigger is a CHRONIC MEDICATION REFILL due for a customer.
Your message should be sent ON BEHALF OF THE MERCHANT to the customer:
- List the specific medications due
- State when stock runs out
- Offer delivery if available
- send_as must be "merchant_on_behalf"
- Tone: caring, precise, no medical advice""",

    "customer_lapsed_hard": """The trigger is a LAPSED CUSTOMER win-back attempt.
Your message should be sent ON BEHALF OF THE MERCHANT to the customer:
- Reference their history (days since last visit, previous focus)
- Offer something specific to re-engage
- send_as must be "merchant_on_behalf"
- Low pressure, curiosity-driven""",

    "winback_eligible": """The trigger is a MERCHANT WIN-BACK — their subscription expired.
Your message should:
- Acknowledge the gap without guilt-tripping
- Show what they've missed (performance decline since expiry)
- Make re-activation frictionless""",

    "dormant_with_vera": """The trigger is DORMANCY — the merchant hasn't responded to Vera in a while.
Your message should:
- NOT repeat old topics
- Open with something genuinely new/interesting
- Ask a question to re-engage (curiosity lever)
- Be very brief""",

    "curious_ask_due": """The trigger is a CURIOUS ASK — time to ask the merchant something engaging.
Your message should:
- Ask a genuine, interesting question about their business
- Something they'd actually want to answer
- Use the question to open a new conversation thread""",
}

BASE_SYSTEM_PROMPT = """You are Vera, an AI growth assistant for Indian merchants on WhatsApp.
You compose concise, high-converting messages grounded in real merchant data.

CRITICAL RULES:
1. STRICT GROUNDING: Use ONLY data from the provided context. NEVER fabricate offers, stats, research, or competitor names.
2. SINGLE CTA: End with ONE clear call-to-action. Binary (YES/STOP) for action triggers; open-ended for information.
3. SPECIFICITY: Anchor on a concrete, verifiable fact (number, date, source, price).
4. VOICE MATCH: Match the category voice (dentists=clinical/peer, salons=warm/practical, restaurants=operator-to-operator, gyms=coaching, pharmacies=trustworthy/precise).
5. CONCISE: WhatsApp messages. No preambles ("I hope you're doing well"), no re-introductions.
6. Hindi-English code-mix is natural and preferred for Indian merchants.
7. Use ₹ for prices. Use service+price format ("Dental Cleaning @ ₹299"), NOT percentage discounts.

Respond ONLY with a raw JSON object (NO markdown fences, NO explanation) with these keys:
{"body": "WhatsApp message text", "cta": "open_ended"|"binary_yes_stop"|"none", "send_as": "vera"|"merchant_on_behalf", "suppression_key": "unique_dedup_key", "rationale": "1-2 sentence reasoning"}"""


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and thinking blocks."""
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)


def _call_llm(system: str, user: str, retries: int = 3) -> str:
    """Call OpenRouter with retry on rate limits."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = _client.chat.completions.create(
                model=OPENROUTER_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content
            raise ValueError("Empty LLM response")
        except Exception as e:
            last_err = e
            if "429" in str(e).lower() or "rate" in str(e).lower():
                wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s
                time.sleep(wait)
            else:
                break
    raise RuntimeError(f"LLM failed after {retries+1} attempts: {last_err}")


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
) -> dict:
    """
    Compose a WhatsApp message from the 4-context framework.

    Inputs are dicts loaded from the dataset JSON.
    Returns dict with keys: body, cta, send_as, suppression_key, rationale.
    Deterministic (temperature=0). Completes in <30s.
    """
    # Determine trigger kind for prompt routing
    trigger_kind = trigger.get("kind", "")
    trigger_specific = TRIGGER_PROMPTS.get(trigger_kind, "")

    # Determine send_as from trigger scope
    is_customer_facing = trigger.get("scope") == "customer" and customer is not None
    default_send_as = "merchant_on_behalf" if is_customer_facing else "vera"

    # Build the context payload for the LLM
    context = {
        "category": {
            "slug": category.get("slug", ""),
            "voice": category.get("voice", {}),
            "offer_catalog": category.get("offer_catalog", [])[:5],
            "peer_stats": category.get("peer_stats", {}),
        },
        "merchant": {
            "name": merchant.get("identity", {}).get("name", ""),
            "owner": merchant.get("identity", {}).get("owner_first_name", ""),
            "city": merchant.get("identity", {}).get("city", ""),
            "locality": merchant.get("identity", {}).get("locality", ""),
            "languages": merchant.get("identity", {}).get("languages", []),
            "performance": merchant.get("performance", {}),
            "active_offers": [
                o for o in merchant.get("offers", []) if o.get("status") == "active"
            ],
            "signals": merchant.get("signals", []),
            "review_themes": merchant.get("review_themes", []),
            "customer_aggregate": merchant.get("customer_aggregate", {}),
        },
        "trigger": trigger,
    }

    if customer:
        context["customer"] = {
            "name": customer.get("identity", {}).get("name", ""),
            "language_pref": customer.get("identity", {}).get("language_pref", ""),
            "state": customer.get("state", ""),
            "relationship": customer.get("relationship", {}),
            "preferences": customer.get("preferences", {}),
        }

    # Build trigger-specific system prompt
    system = BASE_SYSTEM_PROMPT
    if trigger_specific:
        system += f"\n\nTRIGGER-SPECIFIC GUIDANCE:\n{trigger_specific}"

    user_prompt = f"Compose a message for this context:\n{json.dumps(context, default=str)}"

    # Call LLM and parse
    raw = _call_llm(system, user_prompt)
    parsed = _extract_json(raw)

    # Validate and normalize output
    return {
        "body": parsed.get("body", ""),
        "cta": parsed.get("cta", "none"),
        "send_as": parsed.get("send_as", default_send_as),
        "suppression_key": parsed.get(
            "suppression_key",
            trigger.get("suppression_key", f"{trigger.get('id', 'unknown')}:sent"),
        ),
        "rationale": parsed.get("rationale", ""),
    }
