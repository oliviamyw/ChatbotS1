# =========================
# Chatbot App 4 — RAG + Rules + Pending Intents
# Scenario-aware Auto-Pending + Global Intent Switch/Inline Answers (Final)
# =========================

# --- Imports ---
import os
import re
import csv
import uuid
import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI

# LangChain / Vector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="Chatbot Experiment", layout="centered")


# =========================
# OpenAI Client
# =========================
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not API_KEY:
    st.error("OPENAI_API_KEY is not set. Please configure it in environment variables or st.secrets.")
    st.stop()
client = OpenAI(api_key=API_KEY)


# =========================
# Session State Initialization
# =========================
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("session_id", uuid.uuid4().hex[:10])
st.session_state.setdefault("log_dir", "logs")
os.makedirs(st.session_state.log_dir, exist_ok=True)
st.session_state.setdefault("awaiting_feedback", False)
st.session_state.setdefault("ended", False)
st.session_state.setdefault("saved_fpath", None)
st.session_state.setdefault("rating_saved", False)
st.session_state.setdefault("greeted_once", False)          # greet only once
st.session_state.setdefault("scenario_selected_once", False)
st.session_state.setdefault("last_user_selected_scenario", "— Select a scenario —")
MIN_USER_TURNS = 5
st.session_state.setdefault("user_turns", 0)
st.session_state.setdefault("bot_turns", 0)
st.session_state.setdefault("closing_asked", False)

# Flow/slots container (stage starts at 'start')
st.session_state.setdefault("flow", {
    "scenario": None, "stage": "start",
    "slots": {
        "product": None, "color": None, "size": None,
        "contact_pref": None, "tier_known": None, "selected_collection": None,
        # returns flow
        "return_item": None, "received_date": None, "return_reason": None
    }
})

# ---- Pending yes/no intents (global) ----
st.session_state.setdefault("pending", {"intent": None, "data": {}})

def set_pending(intent: str, data: dict | None = None):
    st.session_state.pending = {"intent": intent, "data": (data or {})}

def consume_pending():
    p = st.session_state.pending
    st.session_state.pending = {"intent": None, "data": {}}
    return p

def ask_yesno(intent: str, message: str, data: dict | None = None) -> str:
    """Emit a yes/no question and set a pending intent that will consume the next yes/no reply."""
    set_pending(intent, data or {})
    return message


# =========================
# UI: Header & Identity
# =========================
st.title("🤖 Chatbot User Engagement Experiment")
st.markdown("Please choose your chatbot settings below:")

# --- Identity & greeting (drop-in replacement) ---
identity_option = st.radio(
    "Choose the chatbot identity:",
    options=["No name or image", "With name only", "With image only", "With name and image"],
    index=0  # default: No name
)
show_name = identity_option in ["With name only", "With name and image"]
show_picture = identity_option in ["With image only", "With name and image"]
CHATBOT_NAME = "Riley" if show_name else ""
CHATBOT_PICTURE = "https://i.imgur.com/4uLz4FZ.png" if show_picture else None

# Speaker label: With name → Riley, Without name → Chatbot
def _chatbot_speaker():
    return "Riley" if show_name else "Chatbot"

# Normalize past messages when identity setting changes (relabel + adjust greeting text)
st.session_state.setdefault("last_identity_option", identity_option)
if st.session_state.last_identity_option != identity_option:
    st.session_state.last_identity_option = identity_option

    def inject_name(text: str) -> str:
        # Turn nameless greeting into named one when switching TO name-present
        if "Riley" in text:
            return text
        replacements = [
            ("Hi, I'm Style Loom’s virtual assistant", "Hi, I'm Riley, Style Loom’s virtual assistant"),
            ("Hi, I am Style Loom’s virtual assistant", "Hi, I am Riley, Style Loom’s virtual assistant"),
            ("Hi, I’m Style Loom’s virtual assistant", "Hi, I’m Riley, Style Loom’s virtual assistant"),
        ]
        for old, new in replacements:
            if old in text:
                return text.replace(old, new)
        return text

    def remove_name(text: str) -> str:
        # Remove only the personal name from greeting when switching TO no-name
        patterns = [
            ("Hi, I'm Riley, ", "Hi, I'm "),
            ("Hi, I am Riley, ", "Hi, I am "),
            ("Hi, I’m Riley, ", "Hi, I’m "),
        ]
        for old, new in patterns:
            text = text.replace(old, new)
        return text

    new_history = []
    for spk, msg in st.session_state.chat_history:
        if spk in ("Chatbot", "Riley"):
            new_spk = "Riley" if show_name else "Chatbot"
            new_msg = inject_name(msg) if show_name else remove_name(msg)
        else:
            new_spk, new_msg = spk, msg
        new_history.append((new_spk, new_msg))
    st.session_state.chat_history = new_history

# Optional profile image
if show_picture:
    try:
        st.image(CHATBOT_PICTURE, width=100)
    except Exception:
        st.warning("Profile image could not be loaded.")

# Initial greeting (runs once)
if not st.session_state.greeted_once:
    greet_text = (
        "Hi, I'm Riley, Style Loom’s virtual assistant. I’m here to help with your shopping."
        if show_name
        else "Hi, I'm Style Loom’s virtual assistant. I’m here to help with your shopping."
    )
    st.session_state.chat_history.append((_chatbot_speaker(), greet_text))
    st.session_state.greeted_once = True


# =========================
# Scenarios (with placeholder; announce only when user selects)
# =========================
st.markdown("### How can I help?")

SCENARIOS = [
    "— Select a scenario —",          # placeholder
    "Check product availability",
    "Shipping & returns",
    "Size & fit guidance",
    "New arrivals & collections",
    "Rewards & membership",
    "Discounts & promotions",
    "About the brand",
    "Other"
]

scenario = st.selectbox("Choose one:", SCENARIOS, index=0)

other_goal = ""
if scenario == "Other":
    other_goal = st.text_input("If 'Other', briefly describe your goal (optional)")

# When the user selects a real scenario (not placeholder), announce once and reset flow for that scenario
if (
    scenario != "— Select a scenario —"
    and st.session_state.last_user_selected_scenario != scenario
):
    st.session_state.scenario_selected_once = True
    st.session_state.last_user_selected_scenario = scenario

    # Reset flow for the newly selected scenario
    st.session_state.flow = {
        "scenario": scenario, "stage": "start",
        "slots": {
            "product": None, "color": None, "size": None,
            "contact_pref": None, "tier_known": None, "selected_collection": None,
            "return_item": None, "received_date": None, "return_reason": None
        }
    }

    st.session_state.chat_history.append(
        (_chatbot_speaker(), f"Sure, I will help you with **{scenario}**. Please ask me a question.")
    )


# =========================
# Tone / Categories
# =========================
TONE = "informal"
TONE_STYLE = {
    "informal": "Use a friendly, casual tone. Use emojis.",
    "formal": "Use a formal, respectful tone. No emojis."
}
PRODUCT_CATEGORIES = [
    "blouse", "skirt", "pants", "cardigans / sweaters", "dresses",
    "jumpsuits", "jackets", "t-shirts", "sweatshirt / sweatpants",
    "outer", "coat / trenches", "tops / bodysuits", "activewear",
    "shirts", "shorts", "lingerie", "etc."
]


# =========================
# Regex & Slot Extractors
# =========================
YES_PAT = re.compile(r"\b(yes|yeah|yep|sure|ok|okay|please)\b", re.I)
NO_PAT  = re.compile(r"\b(no|nope|nah|not now|later)\b", re.I)

def extract_color(t: str):
    m = re.search(
        r"\b(black|white|ivory|navy|blue|mist\s?blue|greige|beige|red|green|rose\s?beige|pink|cream|sand|olive|charcoal|oatmeal|forest|berry|ink|brown|purple|orange|yellow|khaki|teal|burgundy|maroon|grey|gray)\b",
        t, re.I
    )
    return m.group(1).lower() if m else None

def extract_size(t: str):
    text = t.lower()
    word_map = {
        r"\b(extra\s*small|x[\- ]?small|xs|xxs)\b": "XS",
        r"\b(small|s)\b": "S",
        r"\b(medium|med|m)\b": "M",
        r"\b(large|l)\b": "L",
        r"\b(extra\s*large|x[\- ]?large|xl)\b": "XL",
        r"\b(xx[\- ]?large|2xl|xxl)\b": "XXL",
    }
    for pat, label in word_map.items():
        if re.search(pat, text, re.I):
            return label
    m = re.search(r"\b(XXS|XS|S|M|L|XL|XXL|0|2|4|6|8|10|12|14|16|18)\b", t, re.I)
    return m.group(1).upper() if m else None

def extract_product(t: str):
    cats = ["blouse", "skirt", "pants", "cardigan", "cardigans", "sweater", "sweaters",
            "dress", "dresses", "jumpsuit", "jumpsuits", "jacket", "jackets",
            "t-shirt", "t-shirts", "sweatshirt", "sweatpants", "outer", "coat",
            "trench", "trenches", "top", "tops", "bodysuit", "bodysuits",
            "activewear", "shirt", "shirts", "shorts", "lingerie"]
    for c in cats:
        if re.search(rf"\b{re.escape(c)}\b", t, re.I):
            if c in ["cardigans", "sweaters", "jackets", "dresses", "tops", "shirts", "jumpsuits"]:
                return c.rstrip("s")
            return c
    w = re.search(r"\b(\w+\s+(jacket|skirt|blouse|t-?shirt|dress|pants))\b", t, re.I)
    return w.group(1) if w else None

def _update_slots_from_text(user_text: str):
    slots = st.session_state.flow["slots"]
    p, c, s = extract_product(user_text), extract_color(user_text), extract_size(user_text)
    if p: slots["product"] = p
    if c: slots["color"] = c
    if s: slots["size"] = s


# =========================
# Close only on end stage
# =========================
def maybe_add_one_time_closing(reply: str) -> str:
    stage = (st.session_state.flow or {}).get("stage")
    if stage == "end_or_more" and (not st.session_state.closing_asked) and (st.session_state.user_turns >= MIN_USER_TURNS - 1):
        st.session_state.closing_asked = True
        return reply + "\n\nIs there anything else I can help you with?"
    return reply


# =========================
# RAG: Build/Load Vectorstore (autodetect_encoding=True; chardet used)
# =========================
RAG_DIR = str(Path.cwd() / "rag_docs")

@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore(rag_dir: str):
    rag_path = Path(rag_dir)
    if not rag_path.exists():
        return None

    persist_dir = str(rag_path / ".chroma")
    embeddings = OpenAIEmbeddings(api_key=API_KEY, model="text-embedding-3-small")

    # Load existing index
    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        try:
            return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        except Exception as e:
            st.warning(f"Vectorstore load warning: {e}")

    # Build new index (autodetect requires chardet)
    try:
        loader = DirectoryLoader(
            rag_dir,
            glob="**/*.*",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=True,
            use_multithreading=True,
        )
        docs = loader.load()
    except Exception as e:
        st.warning(f"RAG documents could not be loaded: {e}")
        return None

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    try:
        return Chroma.from_documents(chunks, embedding_function=embeddings, persist_directory=persist_dir)
    except Exception as e:
        st.warning(f"Vectorstore build failed: {e}")
        return None

vectorstore = build_or_load_vectorstore(RAG_DIR)

def retrieve_context(query: str, k: int = 5) -> str:
    if not vectorstore:
        return ""
    try:
        hits = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        st.warning(f"Similarity search failed: {e}")
        return ""
    blocks = []
    for i, d in enumerate(hits, 1):
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[Doc{i} from {os.path.basename(src)}]\n{d.page_content.strip()}")
    return "\n\n".join(blocks)

def make_query(user_message: str) -> str:
    slots = st.session_state.flow["slots"]
    parts = [f"scenario:{st.session_state.flow['scenario']}"]
    if slots.get("product"): parts.append(f"product:{slots['product']}")
    if slots.get("color"):   parts.append(f"color:{slots['color']}")
    if slots.get("size"):    parts.append(f"size:{slots['size']}")
    parts.append(f"user:{user_message}")
    return " | ".join(parts)


# =========================
# LLM (RAG) fallback
# =========================
def answer_with_rag(user_message: str) -> str:
    _update_slots_from_text(user_message)
    query = make_query(user_message)
    context = retrieve_context(query, k=6)

    style_instruction = TONE_STYLE[TONE]
    bot_identity = f"named {CHATBOT_NAME}" if show_name else "with no name"
    prompt = f"""
You are a helpful customer service chatbot {bot_identity} for Style Loom.
Ground every answer in the BUSINESS CONTEXT. If critical info is missing, ask **one concise follow-up question** only.
Do not invent policy, numbers, or SKUs. Keep answers short and helpful.

=== BUSINESS CONTEXT (retrieved) ===
{context if context else "[no docs retrieved]"}
=== END CONTEXT ===

Meta:
- Current scenario: {scenario}
- Product categories: {", ".join(PRODUCT_CATEGORIES)}
- Known slots: {st.session_state.flow["slots"]}

Style:
{style_instruction}

User: {user_message}
Chatbot:
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.warning(f"LLM call failed: {e}")
        return "Sorry, I had trouble generating a response. Could you rephrase your question?"

def llm_fallback(user_message: str) -> str:
    return answer_with_rag(user_message)


# =========================
# Pending yes/no router (global)
# =========================
def handle_pending_yes(user_text: str) -> str | None:
    pend = st.session_state.pending
    intent = pend.get("intent")
    if not intent:
        return None

    # YES
    if YES_PAT.search(user_text):
        if intent == "rewards_more":
            consume_pending()
            return (
                "**Rewards at a glance**\n"
                "- **Earning:** 1 pt per $1 spent (Bronze). Silver 1.5× (≥ $300/yr), "
                "Gold 2× + Free Express (≥ $800/yr), VIP 2× + Free Express + Gifts (≥ $1,500/yr)\n"
                "- **Redemption:** 100 pts = $1 off. Applies to **merchandise subtotal only** (no tax/shipping).\n"
                "- **Tier window:** Rolling 12 months; downgrades if you fall below thresholds at review.\n"
                "- **Expiration:** Points expire after **12 months** of no earn/redeem activity.\n"
                "Would you like tips on **earning faster** or **redeeming** now?"
            )

        if intent == "colors_sizes_more":
            slots = st.session_state.flow["slots"]
            product = slots.get("product") or "item"
            color = slots.get("color")
            size  = slots.get("size")
            consume_pending()
            size_line = (
                f"For the {product}, typical sizes in stock run XXS–XXL."
                if not size else
                f"For the {product}, size **{size}** is commonly stocked; adjacent sizes are often available."
            )
            color_line = (
                f"Common colors include black, white, beige, navy, brown, and seasonal drops."
                if not color else
                f"Alongside **{color}**, we usually carry black, white, beige, navy, and seasonal colors."
            )
            return (
                f"**Availability guide for {product}**\n"
                f"- {size_line}\n"
                f"- {color_line}\n"
                "Would you like me to check a **specific color/size** now?"
            )

        if intent == "confirm_switch":
            data = consume_pending()
            target = data["data"].get("target")
            if target:
                st.session_state.flow = {
                    "scenario": target, "stage": "start",
                    "slots": { **st.session_state.flow["slots"] }
                }
                return f"Great—switching to **{target}**. How can I help within this topic?"
            consume_pending()
            return "Okay—switching contexts."

        # extend intents as needed
        consume_pending()
        return "Got it."

    # NO
    if NO_PAT.search(user_text):
        if intent == "confirm_switch":
            consume_pending()
            return "No problem—let’s continue with the current topic."
        consume_pending()
        return "All set! If you want the details later, just ask."

    return None


# =========================
# A) Auto-pending inference (scenario-aware)
# =========================
def infer_pending_from_bot_reply(reply_text: str) -> None:
    """
    Look for yes/no-style follow-ups in the bot reply and set a pending intent,
    but only if that intent makes sense for the CURRENT SCENARIO.
    """
    sc = (st.session_state.flow.get("scenario") or "").strip()
    text = (reply_text or "").strip().lower()
    if not text:
        return

    def _match_any(patterns):
        return any(re.search(p, text, re.I) for p in patterns)

    # Rewards follow-ups (earn/redeem)
    if sc == "Rewards & membership":
        if _match_any([
            r"\bwant to know\b.*\b(earn|earning|redeem|redemption|points|rewards)\b",
            r"\bwould you like\b.*\b(earn|earning|redeem|redemption|points|rewards)\b",
        ]):
            set_pending("rewards_more")
            return

    # Product availability / Size & fit: colors & sizes follow-ups
    if sc in ("Check product availability", "Size & fit guidance"):
        if _match_any([
            r"\bwant to know\b.*\b(color|colors|size|sizes)\b",
            r"\binterested in\b.*\b(color|colors|size|sizes)\b",
            r"\bwould you like\b.*\b(color|colors|size|sizes)\b",
        ]):
            set_pending("colors_sizes_more")
            return

    # Shipping & returns 등 기타 시나리오에서는 pending을 설정하지 않음
    return


# =========================
# Rule-based scenario router (deterministic)
# =========================
def route_by_scenario(current_scenario: str, user_text: str) -> str | None:
    flow = st.session_state.flow
    slots = flow["slots"]
    stage = flow.get("stage") or "start"

    # Keep extracting slots from free text
    _update_slots_from_text(user_text)

    # ---- Rewards & membership ----
    if current_scenario == "Rewards & membership":
        if stage in (None, "start"):
            flow["stage"] = "rewards_intro"
            return ask_yesno(
                intent="rewards_more",
                message=(
                    "Every 100 points = $1 off your next order. Points apply to merchandise subtotal only (not taxes or shipping). "
                    "Tiers are based on a rolling 12-month spend, and points expire if not used or earned within 12 months. "
                    "Want to know more about earning or redeeming points?"
                )
            )

        # Keyword shortcuts (optional)
        if re.search(r"\b(earn|earning|accumulate|faster)\b", user_text, re.I):
            return (
                "**Earning faster**\n"
                "- Silver: 1.5× points (≥ $300/yr)\n"
                "- Gold: 2× + Free Express (≥ $800/yr)\n"
                "- VIP: 2× + Free Express + Gifts (≥ $1,500/yr)\n"
                "Promotions may stack, but points are calculated on the **discounted** subtotal."
            )
        if re.search(r"\b(redeem|redemption|use points|apply points)\b", user_text, re.I):
            return (
                "**Redeeming**\n"
                "- 100 pts = $1 off\n"
                "- Apply at checkout on the payment step\n"
                "- Points do not apply to taxes or shipping"
            )
        return None  # let LLM handle other rewards questions

    # ---- Check product availability ----
    if current_scenario == "Check product availability":
        if stage == "start":
            flow["stage"] = "collect"
            stage = "collect"

        if stage == "collect":
            if not slots.get("product"):
                return "Sure—what product are you looking for (e.g., jacket, dress, t-shirt)?"
            if not slots.get("color"):
                return f"Great—what color of {slots['product']}?"
            if not slots.get("size"):
                return f"What size for the {slots['product']} in {slots['color']}?"
            # all slots present → deterministic message (with follow-up yes/no)
            flow["stage"] = "offer_low_stock_alt"
            return ask_yesno(
                intent="colors_sizes_more",  # or define a 'low_stock_alt' later
                message=(
                    f"We have 5+ in stock for the {slots['product']} in {slots['color']} / {slots['size']}. "
                    "However, a different color is running low in stock. Would you also like me to suggest a similar low-stock option?"
                ),
                data={"slots_snapshot": slots.copy()}
            )

        if stage == "offer_low_stock_alt":
            if YES_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return (
                    f"A similar last-season {slots['product']} in the same {slots['color']} is down to the final 2 pieces. "
                    f"Would you like a restock alert for the current color/size, or to see similar items?"
                )
            if NO_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return "All set! Anything else you’d like to check?"
            return "Would you like me to suggest a similar low-stock option?"

        if stage == "end_or_more":
            return "Happy to help. If you need anything else, just say the word!"

        return None

    # ---- Shipping & returns ----  (robust return flow; no color/size branches here)
    if current_scenario == "Shipping & returns":
        if stage in (None, "start"):
            flow["stage"] = "returns_collect_item"
            return ("Of course! I can help with a return. "
                    "What item would you like to return? (e.g., white tennis shoes, size 9)")

        if stage == "returns_collect_item":
            if user_text.strip():
                flow["slots"]["return_item"] = user_text.strip()
            flow["stage"] = "returns_collect_date"
            return ("Got it. When did you receive the item? "
                    "(Please provide a date like 2025-09-10)")

        if stage == "returns_collect_date":
            m = re.search(r"\b(20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})\b", user_text)
            flow["slots"]["received_date"] = m.group(1) if m else "unknown"
            flow["stage"] = "returns_condition_check"
            return ("Thanks. For returns, items must be unworn and in original condition. "
                    "Can you confirm the item is unworn and in its original condition? (yes/no)")

        if stage == "returns_condition_check":
            if YES_PAT.search(user_text):
                flow["stage"] = "returns_reason"
                return ("Understood. Could you tell me the reason for the return? "
                        "(e.g., too small, defective, changed mind)")
            if NO_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return ("Unfortunately, we can only accept returns that are unworn and in original condition. "
                        "If you have questions, I can share our exchange/repair options.")
            return "Please reply yes or no: is the item unworn and in original condition?"

        if stage == "returns_reason":
            flow["slots"]["return_reason"] = user_text.strip()
            flow["stage"] = "returns_instructions"
            item = flow["slots"].get("return_item", "the item")
            return (f"Thanks. To initiate your return for **{item}**, please follow these steps:\n"
                    "1) I’ll create a prepaid return label for you via email.\n"
                    "2) Pack the item securely with all tags/accessories.\n"
                    "3) Drop it off within **14 days** of delivery.\n"
                    "Once we receive and inspect it, we’ll process your refund to the original payment method.\n"
                    "Would you like me to send the return label to your email on file? (yes/no)")

        if stage == "returns_instructions":
            if YES_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return ("Great—return label request submitted. You’ll receive it shortly. "
                        "Is there anything else I can help you with?")
            if NO_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return ("No problem. If you need the label later, just ask. "
                        "Anything else I can help you with?")
            return "Would you like me to email you the return label now? (yes/no)"

        if stage == "end_or_more":
            return "Happy to help. If you need anything else, just say the word!"

        return None

    # ---- Size & fit guidance ----  (numeric-first sizing; avoid M/L if numbers-only)
    if current_scenario == "Size & fit guidance":
        if stage in (None, "start"):
            flow["stage"] = "fit_collect"
            return ("Sure—tell me your current size and how it fits (e.g., pants 28 too small). "
                    "I’ll suggest the next size.")

        if stage == "fit_collect":
            num = re.search(r"\b(\d{2}(\.\d)?|\d{1,2}(\.\d)?)\b", user_text)
            letter = re.search(r"\b(XXS|XS|S|M|L|XL|XXL)\b", user_text, re.I)
            too_small = re.search(r"\b(too\s*small|tight|snug)\b", user_text, re.I)
            too_big   = re.search(r"\b(too\s*big|loose)\b", user_text, re.I)
            numbers_only = re.search(r"\bnumbers?\s+only\b", user_text, re.I)

            if num:
                base = num.group(1)
                try:
                    val = float(base)
                except:
                    val = None

                if val is not None:
                    if too_small:
                        rec = val + 1 if val >= 20 else val + 0.5
                    elif too_big:
                        rec = val - 1 if val >= 20 else val - 0.5
                    else:
                        rec = val + 1
                    flow["stage"] = "end_or_more"
                    return (f"Since **{base}** feels small, try **{rec:.1f}** (or the next up). "
                            "If the brand runs small, you may need one more size up. "
                            "Need help finding stock for that size?")

            if letter and not numbers_only:
                L = letter.group(1).upper()
                nxt = {"XXS":"XS","XS":"S","S":"M","M":"L","L":"XL","XL":"XXL"}.get(L,"next size up")
                flow["stage"] = "end_or_more"
                return (f"Since **{L}** feels small, try **{nxt}**. "
                        "Want me to check availability in that size?")

            return ("Got it. If your **current size** (e.g., 28, 30, 9) feels small or big, "
                    "tell me which way, and I’ll recommend the next size up/down.")

        if stage == "end_or_more":
            return "Happy to help. If you need anything else, just say the word!"

        return None

    # Other scenarios → let LLM handle
    return None


# =========================
# Global intent detection (cross-scenario)
# =========================
GLOBAL_INTENTS = [
    # pattern, intent_key, target_scenario, priority, can_inline
    (r"\b(return|refund|send back|exchange)\b", "returns_intent", "Shipping & returns", 10, False),
    (r"\b(exchange|swap size|different size|too (small|big))\b", "fit_intent", "Size & fit guidance", 8, True),
    (r"\b(availability|in stock|stock|have .* size|colors?|sizes?)\b", "availability_intent", "Check product availability", 7, True),
    (r"\b(reward|point|redeem|earn|membership|tier)\b", "rewards_intent", "Rewards & membership", 6, True),
]

def detect_global_intent(user_text: str):
    text = (user_text or "").lower()
    best = None
    for pat, key, target, prio, can_inline in GLOBAL_INTENTS:
        if re.search(pat, text, re.I):
            if (best is None) or (prio > best["priority"]):
                best = {"key": key, "target": target, "priority": prio, "can_inline": can_inline}
    return best

# Inline short answers (keep current scenario; propose switch)
def inline_answer_availability(user_text: str) -> str:
    _update_slots_from_text(user_text)
    slots = st.session_state.flow["slots"]
    p = slots.get("product") or "item"
    c = slots.get("color")
    s = slots.get("size")
    base = f"For the {p}"
    if c: base += f" in {c}"
    if s: base += f" / {s}"
    return (
        f"{base}, I can check availability in detail if you like. "
        f"Would you like to **switch to Check product availability**?"
    )

def inline_answer_fit(user_text: str) -> str:
    mnum = re.search(r"\b(\d{2}(\.\d)?|\d{1,2}(\.\d)?)\b", user_text)
    if mnum:
        base = float(mnum.group(1))
        rec = base + 1 if base >= 20 else base + 0.5
        return (f"If **{base:.1f}** feels small, try **{rec:.1f}**. "
                "Want to **switch to Size & fit guidance** for a precise recommendation?")
    return "I can help with sizing. Want to **switch to Size & fit guidance**?"

def inline_answer_rewards(user_text: str) -> str:
    return ("Every 100 pts = $1 off (merchandise subtotal only). Tiers are rolling 12 months; "
            "points expire after 12 months of no activity. "
            "Want to **switch to Rewards & membership** to see earning/redeem tips?")

INLINE_HANDLERS = {
    "availability_intent": inline_answer_availability,
    "fit_intent": inline_answer_fit,
    "rewards_intent": inline_answer_rewards,
}


# =========================
# Orchestrator
# =========================
def handle_message(user_text: str) -> str:
    # 0) pending 우선
    pending_reply = handle_pending_yes(user_text)
    if pending_reply:
        return maybe_add_one_time_closing(pending_reply)

    # 0.5) Global intent pre-check (cross-scenario)
    detected = detect_global_intent(user_text)
    if detected:
        current = st.session_state.flow.get("scenario")
        target = detected["target"]
        if current != target:
            if detected["can_inline"]:
                inline_fun = INLINE_HANDLERS.get(detected["key"])
                if inline_fun:
                    reply = inline_fun(user_text)
                    set_pending("confirm_switch", {"target": target})
                    return maybe_add_one_time_closing(reply)
            # not inline-capable (e.g., returns): ask to switch
            msg = f"It sounds like **{target}** might be more helpful. Switch to that topic?"
            set_pending("confirm_switch", {"target": target})
            return maybe_add_one_time_closing(msg)
        # same scenario → just continue to rules below

    # 1) Scenario rules
    rule_reply = route_by_scenario(st.session_state.flow.get("scenario") or scenario, user_text)
    if rule_reply is not None:
        infer_pending_from_bot_reply(rule_reply)
        return maybe_add_one_time_closing(rule_reply)

    # 2) RAG + LLM fallback
    bot_reply = llm_fallback(user_text)
    infer_pending_from_bot_reply(bot_reply)
    return maybe_add_one_time_closing(bot_reply)


# =========================
# Chat Loop
# =========================
if not st.session_state.awaiting_feedback and not st.session_state.ended:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your message:")
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        st.session_state.user_turns += 1
        st.session_state.chat_history.append(("User", user_input.strip()))

        bot_reply = handle_message(user_input.strip())

        speaker = _chatbot_speaker()
        st.session_state.chat_history.append((speaker, bot_reply))
        st.session_state.bot_turns += 1

    if st.session_state.user_turns < MIN_USER_TURNS:
        remaining = MIN_USER_TURNS - st.session_state.user_turns
        st.info(f"You’ve sent {st.session_state.user_turns}/{MIN_USER_TURNS} messages (minimum). {remaining} more to go.")
        st.progress(min(st.session_state.user_turns / MIN_USER_TURNS, 1.0))
else:
    if st.session_state.awaiting_feedback and not st.session_state.ended:
        st.info("This chat is paused. Please complete the quick survey below to finish.")
    else:
        st.info("This session has ended. Thank you for your feedback!")


# =========================
# Transcript & Survey
# =========================
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")

if st.button("Download Chat Log"):
    filename = f"chatlog_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for spk, msg in st.session_state.chat_history:
                f.write(f"{spk}: {msg}\n")
        st.success("Chat log saved locally (if running on your machine).")
    except Exception as e:
        st.error(f"Failed to write chat log: {e}")

st.markdown("---")
can_end = (st.session_state.user_turns >= MIN_USER_TURNS)
help_text = None if can_end else f"Please send at least {MIN_USER_TURNS - st.session_state.user_turns} more message(s) before ending."
if (not st.session_state.awaiting_feedback) and (not st.session_state.ended):
    if st.button("End Session", disabled=not can_end, help=help_text):
        st.session_state.awaiting_feedback = True
        st.success("Session paused. Please answer the quick satisfaction question below to finish.")

if st.session_state.awaiting_feedback and not st.session_state.ended:
    st.subheader("Before you go…")
    st.write("**Overall, how satisfied are you with this chatbot service today?**")
    st.caption("1 = Very dissatisfied, 7 = Very satisfied")
    rating = st.slider("Your overall satisfaction", min_value=1, max_value=7, value=5, step=1)

    if st.button("Submit Rating"):
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        fname = f"transcript_{st.session_state.session_id}_{ts}.txt"
        fpath = os.path.join(st.session_state.log_dir, fname)
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("===== Session Transcript =====\n")
                f.write(f"timestamp       : {ts}\n")
                f.write(f"session_id      : {st.session_state.session_id}\n")
                f.write(f"identity_option : {identity_option}\n")
                f.write(f"name_present    : {'present' if show_name else 'absent'}\n")
                f.write(f"picture_present : {'present' if show_picture else 'absent'}\n")
                scenostr = scenario if scenario != "Other" else f"Other: {other_goal.strip() if other_goal else ''}"
                f.write(f"scenario        : {scenostr}\n")
                f.write(f"user_turns      : {st.session_state.user_turns}\n")
                f.write(f"bot_turns       : {st.session_state.bot_turns}\n")
                f.write("--------------------------------\n")
                for spk, msg in st.session_state.chat_history:
                    f.write(f"{spk}: {msg}\n")
                f.write("--------------------------------\n")
                f.write(f"Satisfaction (1-7): {rating}\n")
        except Exception as e:
            st.error(f"Failed to save transcript: {e}")
        else:
            st.session_state.saved_fpath = fpath

        csv_path = os.path.join(st.session_state.log_dir, "satisfaction.csv")
        header = ["timestamp", "session_id", "identity_option", "name_present", "picture_present",
                  "scenario", "user_turns", "bot_turns", "satisfaction_1to7"]
        row = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               st.session_state.session_id, identity_option,
               "present" if show_name else "absent",
               "present" if show_picture else "absent",
               scenostr, st.session_state.user_turns,
               st.session_state.bot_turns, rating]

        try:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(header)
                w.writerow(row)
        except Exception as e:
            st.error(f"Failed to save rating CSV: {e}")
        else:
            st.session_state.rating_saved = True
            st.session_state.ended = True
            st.session_state.awaiting_feedback = False
            st.success("Thanks! Your feedback has been recorded. The session is now closed.")

if st.session_state.ended and st.session_state.rating_saved:
    st.info("Your session has ended and your feedback was recorded. You may close this window.")
