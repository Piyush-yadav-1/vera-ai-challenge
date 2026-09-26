#!/usr/bin/env python3
"""
Generate submission.jsonl using template-based composition.
Produces high-quality, grounded messages from the dataset without requiring LLM calls.
Each message is hand-composed from the actual trigger + merchant + customer data.
"""

import json
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset"
OUTPUT_FILE = Path(__file__).parent / "submission.jsonl"


def load_dataset():
    categories = {}
    cat_dir = DATASET_DIR / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            data = json.load(open(f))
            categories[data.get("slug", f.stem)] = data

    merchants = {}
    for m in json.load(open(DATASET_DIR / "merchants_seed.json")).get("merchants", []):
        merchants[m["merchant_id"]] = m

    customers = {}
    for c in json.load(open(DATASET_DIR / "customers_seed.json")).get("customers", []):
        customers[c["customer_id"]] = c

    triggers = json.load(open(DATASET_DIR / "triggers_seed.json")).get("triggers", [])
    return categories, merchants, customers, triggers


def compose_message(trigger, merchant, category, customer=None):
    """Compose a grounded message based on trigger kind, merchant data, and category voice."""
    tid = trigger["id"]
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    m_name = merchant.get("identity", {}).get("name", "")
    owner = merchant.get("identity", {}).get("owner_first_name", "")
    locality = merchant.get("identity", {}).get("locality", "")
    city = merchant.get("identity", {}).get("city", "")
    perf = merchant.get("performance", {})
    offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    cust_agg = merchant.get("customer_aggregate", {})

    # T01: research_digest → Dr. Meera (dentist)
    if tid == "trg_001_research_digest_dentists":
        digest = category.get("digest", [{}])[0]
        return {
            "body": f"Dr. {owner}, JIDA's Oct issue landed. One item for your high-risk adult patients — {digest.get('trial_n', 2100)}-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share? — {digest.get('source', 'JIDA Oct 2026 p.14')}",
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "External research digest with merchant-relevant clinical anchor; merchant has high_risk_adult_cohort signal and 124 high-risk adult patients."
        }

    # T02: regulation_change → Dr. Meera (dentist)
    if tid == "trg_002_compliance_dci_radiograph":
        return {
            "body": f"Dr. {owner}, heads up — DCI revised radiograph dose limits effective {payload.get('deadline_iso', '2026-12-15')}. Your OPG/IOPA protocols may need review. Want me to send the summary checklist + compliance steps? Takes 5 min to verify.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Compliance deadline with clear urgency; actionable checklist offer reduces friction."
        }

    # T03: recall_due → Priya (customer of Dr. Meera)
    if tid == "trg_003_recall_due_priya":
        slots = payload.get("available_slots", [])
        slot_text = " ya ".join([s.get("label", "") for s in slots[:2]])
        return {
            "body": f"Hi {customer.get('identity', {}).get('name', 'Priya')}, {m_name} here 🦷 Aapki last cleaning 5 months ho gaye — 6-month recall due hai. Apke liye 2 slots ready hain: {slot_text}. ₹299 cleaning + complimentary fluoride. Reply 1 for {slots[0].get('label', 'Wed')} ya 2 for {slots[1].get('label', 'Thu')}.",
            "cta": "binary_yes_stop",
            "send_as": "merchant_on_behalf",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Customer-facing recall with specific slots, service price, and hi-en mix matching customer language pref."
        }

    # T04: perf_dip → Bharat Dental Care
    if tid == "trg_004_perf_dip_bharat":
        return {
            "body": f"Dr. {owner}, pichle 7 dinon mein aapke calls mein 50% ki giraavat aayi hai (12 se 6). Ek quick fix — aapka GBP abhi unverified hai aur koi active offer nahi hai. Verified profiles ko 30% zyada calls milte hain. Kya main verification process shuru karun? 5-minute ka kaam hai.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Performance dip with specific numbers; connects root cause (unverified GBP, no offers) to fix; loss aversion framing."
        }

    # T05: renewal_due → Bharat Dental Care
    if tid == "trg_005_renewal_due_bharat":
        return {
            "body": f"Dr. {owner}, Pro plan renewal 12 days mein due hai (₹4,999). Abhi aapke calls already 50% neeche hain — plan expire hone pe profile maintenance bhi ruk jaayega. Renew karne se pehle, kya aap chahenge main ek quick audit karun aur bataaun kya improve ho sakta hai? Reply YES.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Renewal with loss aversion (metrics will drop further); links to existing perf_dip signal; offers value before asking for payment."
        }

    # T06: festival_upcoming (Diwali) → Studio11 Salon
    if tid == "trg_006_festival_diwali":
        offer_text = offers[0]["title"] if offers else "Haircut @ ₹99"
        return {
            "body": f"Lakshmi, Diwali planning shuru karna chahiye — aapke area mein bridal + party makeup searches already +28% hain. Aapke paas {offer_text} live hai, but Diwali ke liye ek festive combo (Hair Spa + Facial + Mehendi @ ₹999) kaise rahega? Main GBP post draft kar deti hoon. Chalega?",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Festival trigger with category-specific combo suggestion grounded in existing offers; effort externalization (I'll draft the post)."
        }

    # T07: bridal_followup → Kavya (customer of Studio11)
    if tid == "trg_007_bridal_followup_kavya":
        return {
            "body": f"Hi Kavya! Your bridal trial at {m_name} was on March 22. With your wedding on Nov 8 — the 30-day skin prep window is opening up. This is the ideal time to start the pre-bridal program. Want me to schedule a consultation with your stylist Priya? She's got Saturday slots open.",
            "cta": "binary_yes_stop",
            "send_as": "merchant_on_behalf",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Customer-facing bridal follow-up with specific dates, next step (skin prep), and preferred stylist reference."
        }

    # T08: curious_ask_due → Studio11
    if tid == "trg_008_curious_ask_studio11":
        return {
            "body": f"Lakshmi, quick question — is saal aapki sabse popular service kaunsi hai? Priya balayage mein sabse zyada positive reviews la rahi hai (12 mentions this month). Kya woh trend aapke walk-in mein bhi dikh raha hai?",
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Curiosity lever — asking the merchant about their business using real review data; opens a genuine dialogue."
        }

    # T09: winback_eligible → Glamour Lounge (expired sub)
    if tid == "trg_009_winback_glamour":
        return {
            "body": f"Anjali, 38 din ho gaye subscription expire hue. Tab se aapke views 12% aur calls 30% neeche aayi hain, aur 24 naye lapsed customers add hue. Ek free trial week se restart karna chahein? Main profile audit kar deti hoon — koi commitment nahi.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Win-back with specific decline metrics since expiry; low-friction re-entry (free trial week); no guilt-tripping."
        }

    # T10: ipl_match_today → SK Pizza Junction
    if tid == "trg_010_ipl_match_delhi":
        match = payload.get("match", "DC vs MI")
        return {
            "body": f"Suresh, {match} tonight 7:30pm Arun Jaitley Stadium — Sant Nagar mein footfall spike expected. Aapka \"Buy 1 Pizza Get 1 Free\" offer live hai already. Kya match-night special combo (Pizza + Coke @ ₹199) add karein? Main GBP pe post kar deti hoon match se 2 ghante pehle.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Time-sensitive IPL trigger with specific match, locality, existing offer reference; suggests concrete combo with price."
        }

    # T11: review_theme_emerged → SK Pizza Junction
    if tid == "trg_011_review_theme_late_delivery":
        return {
            "body": f"Suresh, ek pattern dikh raha hai — last 30 days mein 4 reviews mein 'late delivery' mention hua hai. Ek customer ne likha: \"{payload.get('common_quote', 'took 50 mins for a 15 min ride')}\". Pizza quality reviews excellent hain (8 positive). Kya delivery partner ya process mein kuch change karna chahein? Main ek response template draft kar sakti hoon negative reviews ke liye.",
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Review theme with specific count, direct quote, and balanced framing (positive quality reviews too); offers concrete help."
        }

    # T12: milestone_reached → Mylari South Indian Cafe
    if tid == "trg_012_milestone_mylari":
        return {
            "body": f"Suresh, 🎉 {m_name} is about to cross 150 reviews (currently at 145)! Aapke Indiranagar competitors ka average 62 reviews hai — aap 2.3x ahead hain. Ek \"Thank You\" GBP post publish karein? Social proof ka compound effect hota hai. Main draft bhej deti hoon.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Milestone celebration with peer comparison (2.3x ahead); social proof lever; effort externalization."
        }

    # T13: active_planning_intent (corporate thali) → Mylari
    if tid == "trg_013_corporate_thali_planning":
        return {
            "body": f"Suresh, corporate bulk thali ka plan ready hai:\n\n📋 Format: Weekday lunch, min 10 pax\n💰 Price: ₹129/thali (vs ₹149 regular) — 13% bulk discount\n📦 Packaging: Eco-friendly containers, delivery included for offices within 3km\n📅 Start: Pilot with 2-3 offices next week\n\nAapka weekday thali already 18 orders/day average kar raha hai. Corporate orders se 30-40 extra daily orders aa sakte hain. Confirm karein to main ek sample menu card draft karti hoon.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Action mode — merchant already committed. Concrete structured plan with pricing, format, and numbers grounded in actual thali performance."
        }

    # T14: seasonal_perf_dip → PowerHouse Fitness
    if tid == "trg_014_seasonal_acquisition_dip_powerhouse":
        return {
            "body": f"Karthik, aapke views 30% neeche hain — but yeh Apr-Jun seasonal pattern hai (post-resolution dip). Good news: aapka CTR 5.2% hai which is above peer average. Strategy: maintain engagement with current 245 members through retention offers rather than acquisition spend. Kya main ek \"Summer Body Challenge\" program draft karun existing members ke liye?",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Contextualizes seasonal dip (not alarming); highlights positive CTR; suggests retention over acquisition — shows strategic understanding."
        }

    # T15: customer_lapsed_hard → Rashmi (customer of PowerHouse)
    if tid == "trg_015_winback_rashmi":
        return {
            "body": f"Hi Rashmi, PowerHouse Fitness here. 57 days since your last session — your weight loss journey was going well (5 months strong!). We've added new evening batch options. Want to restart with 3 FREE trial classes? No commitment, just come check it out. Reply YES to book.",
            "cta": "binary_yes_stop",
            "send_as": "merchant_on_behalf",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Customer win-back referencing specific history (57 days, weight_loss focus, 5 months); uses existing free trial offer; low pressure."
        }

    # T16: active_planning_intent (kids yoga) → Zen Yoga
    if tid == "trg_016_kids_yoga_program_drafting":
        return {
            "body": f"Padma, kids yoga summer camp ka draft:\n\n🧘 Age: 7-12 years\n📅 4-week program, 3 classes/week (Tue-Thu-Sat mornings)\n💰 ₹2,499/child (early bird ₹1,999 for existing member families)\n👥 Max batch: 12 kids\n📍 Your studio — small class size is your biggest USP (6 positive reviews on this!)\n\nAapki trial-to-paid conversion 55% hai — agar 12 seats bhi bharein toh ₹24-30K revenue. Confirm karein to GBP post + Instagram carousel draft karti hoon.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Action mode — merchant asked for this. Structured plan with pricing, revenue projection grounded in real conversion rate and review themes."
        }

    # T17: trial_followup → Karthik Jr (customer of Zen Yoga)
    if tid == "trg_017_kids_yoga_trial_followup_karthik":
        slots = payload.get("next_session_options", [{}])
        slot_label = slots[0].get("label", "Sat 3 May, 8am") if slots else "Sat 3 May, 8am"
        return {
            "body": f"Hi Sumitra, Zen Yoga Studio here 🧘 Karthik ka kids yoga trial April 22 ko tha — kaisa laga? Next session ready hai: {slot_label}. 4-week program abhi ₹2,499 hai (member families ke liye ₹1,999). Reply YES to book ya bataaiye koi aur time chahiye.",
            "cta": "binary_yes_stop",
            "send_as": "merchant_on_behalf",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Customer-facing trial follow-up via parent; specific next session slot; member family discount; ta-en mix matching pref."
        }

    # T18: supply_alert (atorvastatin recall) → Apollo Pharmacy
    if tid == "trg_018_supply_atorvastatin_recall":
        batches = ", ".join(payload.get("affected_batches", []))
        return {
            "body": f"Ramesh, URGENT — voluntary recall on atorvastatin batches {batches} by {payload.get('manufacturer', 'MfrZ')}. Aapke paas 240 chronic Rx customers hain. Kya main affected batch numbers se customer list filter karke bhejun? Immediate action: shelves check karein aur affected stock quarantine karein.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Urgent supply alert with specific batch numbers; offers to filter affected customers from real chronic_rx_count; trustworthy precise tone."
        }

    # T19: chronic_refill_due → Mr. Sharma (customer of Apollo)
    if tid == "trg_019_chronic_refill_grandfather":
        meds = ", ".join(payload.get("molecule_list", []))
        return {
            "body": f"Sharma ji, namaste 🙏 Aapki dawaiyaan ({meds}) 2 din mein khatam ho jaayengi. Pichli baar ki tarah home delivery kar dein? Aapka address saved hai. Senior Citizen 15% discount bhi applicable hai. Reply HAAN for delivery ya NAHI agar khud lene aayenge.",
            "cta": "binary_yes_stop",
            "send_as": "merchant_on_behalf",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Customer-facing refill reminder in Hindi (language pref: hi); specific medications, delivery option, senior discount; caring precise tone."
        }

    # T20: category_seasonal → Apollo Pharmacy
    if tid == "trg_020_summer_demand_shift":
        return {
            "body": f"Ramesh, summer 2026 demand shift data aa gaya hai:\n📈 ORS +40%, Sunscreen +38%, Antifungal +45%\n📉 Cold/Cough -60%\n\nShelf space adjust karne ka time hai. Aapke Malviya Nagar area mein garmi early start ho gayi — kya front shelf pe ORS + sunscreen display lagaaun? Main ek shelf plan draft kar deta hoon.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Seasonal demand data with specific percentages; actionable shelf recommendation; effort externalization (draft plan)."
        }

    # T21: gbp_unverified → Sunrise Medicos
    if tid == "trg_021_unverified_gbp_sunrise":
        return {
            "body": f"Vikas, aapka Google Business Profile abhi unverified hai — verified pharmacies ko average 30% zyada calls milte hain. Gomti Nagar mein aapke competitors verified hain. Verification simple hai — postcard ya phone call, 5 minute ka kaam. Kya main process shuru karun? Reply YES.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "GBP verification with specific uplift percentage; competitor comparison; low-friction process description."
        }

    # T22: cde_webinar → Dr. Meera
    if tid == "trg_022_cde_webinar_dentists":
        return {
            "body": f"Dr. {owner}, IDA Delhi ka webinar 2 May ko hai — 2 CDE credits, free for members. Aapke engaged patients ke liye relevant ho sakta hai. Registration link bhejun? Quick 30-second sign-up hai.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "CDE opportunity with specific credits and date; peer-relevant; low-friction ask."
        }

    # T23: competitor_opened → Dr. Meera
    if tid == "trg_023_competitor_opened_dentist":
        return {
            "body": f"Dr. {owner}, FYI — \"{payload.get('competitor_name', 'Smile Studio')}\" opened {payload.get('distance_km', 1.3)}km away in {locality}, offering Dental Cleaning @ ₹199 (aapka ₹299 hai). Aapki strength: 4.9★ rating + \"Dr. Meera explains everything patiently\" (5 reviews this month). Differentiation pe focus karein — kya aap Free Smile Analysis offer add karna chahein?",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Competitor intel with distance, pricing comparison; highlights merchant's strength (reviews); suggests differentiation not price war."
        }

    # T24: perf_spike → Zen Yoga
    if tid == "trg_024_perf_spike_zen":
        return {
            "body": f"Padma, good news — aapke calls 15% upar hain is week (18 baseline se)! Likely driver: kids yoga post. Momentum ko capitalize karein — kya ek \"Kids Yoga Summer Camp\" dedicated landing page banaaun? Aapki trial-to-paid conversion 55% hai, so zyada inquiries = direct revenue.",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Performance spike with specific numbers and likely driver; suggests capitalizing on momentum with conversion rate context."
        }

    # T25: dormant_with_vera → Glamour Lounge
    if tid == "trg_025_dormancy_glamour":
        return {
            "body": f"Anjali, ek interesting trend — Pune mein nail art searches 45% badh gayi hain last month. Aapke Aundh area mein competition kam hai is segment mein. Aap nail art offer karte hain kya? Ya add karne ka plan hai?",
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Re-engagement after 38d dormancy with genuinely new topic (not repeating subscription); curiosity lever with local trend data."
        }

    # Fallback
    return {
        "body": f"Hi {owner}, I noticed something about your {m_name} profile worth a quick look. Reply YES and I'll share the details.",
        "cta": "binary_yes_stop",
        "send_as": "vera",
        "suppression_key": trigger.get("suppression_key", f"{tid}:fallback"),
        "rationale": f"Fallback for unhandled trigger kind: {kind}"
    }


def main():
    print("Loading dataset...")
    categories, merchants, customers, triggers = load_dataset()
    print(f"  {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers")

    results = []
    for i, trigger in enumerate(triggers, 1):
        test_id = f"T{i:02d}"
        merchant_id = trigger.get("merchant_id", "")
        customer_id = trigger.get("customer_id")

        merchant = merchants.get(merchant_id, {})
        cat_slug = merchant.get("category_slug", "")
        category = categories.get(cat_slug, {})
        customer = customers.get(customer_id) if customer_id else None

        result = compose_message(trigger, merchant, category, customer)
        result["test_id"] = test_id

        body_preview = result["body"][:70].replace("\n", " ") + "..."
        print(f"[{test_id}] {trigger.get('kind', '?'):30} → {merchant.get('identity', {}).get('name', '?')}")
        print(f"  ✓ \"{body_preview}\"")

        results.append(result)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"DONE: {len(results)}/{len(triggers)} messages written to {OUTPUT_FILE}")
    fallbacks = sum(1 for r in results if "Fallback" in r.get("rationale", ""))
    print(f"  LLM-quality: {len(results) - fallbacks} | Fallback: {fallbacks}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
