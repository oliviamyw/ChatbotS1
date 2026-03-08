# =========================
# Style Loom Chatbot Experiment (STUDY 2 - VISUAL PRESENT × RELEVANT)
# Visual cue fixed (name + image present) + KB-grounded answers (LangChain) + GPT fallback
# Study 2 factor: response relevance (THIS FILE = RELEVANT). Brand factor removed.
#
# Folder requirement:
#   ./data/  (md/json knowledge files)
#
# Streamlit Secrets required:
#   OPENAI_API_KEY
#   SUPABASE_URL
#   SUPABASE_ANON_KEY
#
# Supabase tables (must exist):
#   public.sessions(
#       session_id text primary key,
#       ts_start timestamptz,
#       ts_end timestamptz,
#       identity_option text,
#       relevance_condition text,
#       name_present text,
#       picture_present text,
#       scenario text,
#       user_turns int,
#       bot_turns int
#   )
# =========================

import os
import re
import uuid
import json
import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import streamlit as st
from openai import OpenAI
from supabase import create_client  # Supabase is REQUIRED

# LangChain / Vector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Style Loom Chatbot Experiment", layout="centered")


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"  # LangChain loads .md/.json knowledge files from this folder


# -------------------------
# Experiment constants
# -------------------------
MODEL_CHAT = "gpt-4o-mini"
MODEL_EMBED = "text-embedding-3-small"
MIN_USER_TURNS = 5

TBL_SESSIONS = "sessions"
TBL_TRANSCRIPTS = "transcripts"
TBL_SATISFACTION = "satisfaction"


# -------------------------
# OpenAI client
# -------------------------
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not API_KEY:
    st.error("OPENAI_API_KEY is not set. Please configure it in environment variables or st.secrets.")
    st.stop()
client = OpenAI(api_key=API_KEY)


# -------------------------
# Supabase client (REQUIRED)
# -------------------------
SUPA_URL = st.secrets.get("SUPABASE_URL", None)
SUPA_KEY = st.secrets.get("SUPABASE_ANON_KEY", None)
if not SUPA_URL or not SUPA_KEY:
    st.error("Supabase credentials are missing. Please set SUPABASE_URL and SUPABASE_ANON_KEY in st.secrets.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_supabase():
    return create_client(SUPA_URL, SUPA_KEY)

supabase = get_supabase()


# -------------------------
# -------------------------
# Study condition (THIS CELL: Mass-Market / 20 years)
# -------------------------
identity_option = "With name and image"   # adjust per cell if needed (e.g., name only, image only, none)
show_name = True
show_picture = True
CHATBOT_NAME = "Skyler"
CHATBOT_PICTURE = "https://i.imgur.com/4uLz4FZ.png"

# IMPORTANT: these labels must match your manipulation and what you want stored in Supabase
brand_type = "Mass-market Brand"             # <-- Mass-market condition label (Supabase 저장값)
BRAND_AGE_YEARS_TEXT = "twenty years ago"  # <-- Greeting 문장에 그대로 들어갈 표현


def chatbot_speaker() -> str:
    return CHATBOT_NAME if show_name else "Assistant"


# -------------------------
# Header UI (photo only here; chat transcript is text-only)
# -------------------------
st.markdown(
    "<div style='display:flex;align-items:center;gap:8px;margin:8px 0 4px 0;'>"
    "<div style='font-weight:700;font-size:20px;letter-spacing:0.3px;'>Style Loom</div>"
    "</div>",
    unsafe_allow_html=True,
)
if show_picture:
    try:
        st.image(CHATBOT_PICTURE, width=84)
    except Exception:
        pass


# -------------------------
# Scenarios (dropdown)
# -------------------------
SCENARIOS = [
    "— Select a scenario —",
    "Check product availability",
    "Shipping & returns",
    "Size & fit guidance",
    "New arrivals & collections",
    "Rewards & membership",
    "Discounts & promotions",
    "About the brand",
    "Other",
]

SCENARIO_TO_INTENT = {
    "Check product availability": "availability",
    "Shipping & returns": "shipping_returns",
    "Size & fit guidance": "size_fit",
    "New arrivals & collections": "new_arrivals",
    "Rewards & membership": "rewards",
    "Discounts & promotions": "promotions",
    "About the brand": "about",
    "Other": "other",
    "— Select a scenario —": "none",
}

INTENT_TO_FILES = {
    "availability": [
        "availability_playbook.md",
        "availability_rules.md",
        "inventory_schema.json",
        "mens_and_womens_catalog.md",
    ],
    "shipping_returns": [
        "shipping_returns.md",
        "free_returns_policy.md",
    ],
    "size_fit": [
        "size_chart.md",
        "vocab.md",
    ],
    "new_arrivals": [
        "new_drop.md",
        "current.md",
    ],
    "rewards": [
        "rewards.md",
    ],
    "promotions": [
        "current.md",
        "promotions_rules.md",
        "price_policy_and_ranges.md",
    ],
    "about": [
        "about.md",
    ],
}

FILE_TO_INTENT: Dict[str, str] = {}
for ik, files in INTENT_TO_FILES.items():
    for fn in files:
        FILE_TO_INTENT[fn] = ik


def scenario_to_intent(scenario: Optional[str]) -> str:
    if not scenario:
        return "none"
    return SCENARIO_TO_INTENT.get(scenario, "other")


# -------------------------
# Intent detection (ENGLISH ONLY) for auto-switch (Option C)
# -------------------------
INTENT_KEYWORDS: Dict[str, List[str]] = {
    # New arrivals / product drop (include known item names to avoid "about" misrouting)
    "new_arrivals": [
        "new drop", "new arrivals", "new arrival", "new collection", "latest", "this season",
        "spring collection", "summer collection", "fall collection", "winter collection",
        "soft blouse", "city knit", "everyday jacket", "tailored pants", "weekend dress",
        "collection", "new products", "new product"
    ],
    # Size and fit
    "size_fit": [
        "size", "sizing", "fit", "measurement", "measurements", "bust", "waist", "hip",
        "xs", "xl", "x-small", "x large", "cm", "inch", "inches", "runs small", "runs large"
    ],
    # Shipping and returns
    "shipping_returns": [
        "shipping", "ship", "delivery", "deliver", "carrier", "ups", "fedex", "usps", "ground",
        "standard shipping", "express shipping", "how long", "shipping time", "delivery time", "tracking",
        "return", "returns", "exchange", "refund", "return window", "return policy"
    ],
    # Promotions
    "promotions": [
        "discount", "promo", "promotion", "coupon", "code", "sale", "deal", "welcome10",
        "excluded", "exclusions", "exclude", "final sale", "gift card", "mto_excluded",
        "apply", "apply code"
    ],
    # Rewards / membership
    "rewards": [
        "reward", "rewards", "points", "membership", "member", "vip", "tier",
        "benefits", "join", "sign up", "enroll", "account"
    ],
    # Availability / inventory
    "availability": [
        "available", "availability", "in stock", "out of stock", "restock", "sold out", "inventory",
        "do you have", "do you carry"
    ],
    # About brand
    "about": [
        "about", "brand", "story", "who are you", "who is", "ceo", "quality", "sustainability"
    ],
}

INTENT_TO_SCENARIO = {
    "availability": "Check product availability",
    "shipping_returns": "Shipping & returns",
    "size_fit": "Size & fit guidance",
    "new_arrivals": "New arrivals & collections",
    "rewards": "Rewards & membership",
    "promotions": "Discounts & promotions",
    "about": "About the brand",
}


def detect_intent(user_text: str) -> Tuple[Optional[str], int]:
    """
    Lightweight intent detection (ENGLISH ONLY).
    Returns (best_intent, score) where score is the number of keyword hits for that intent.
    """
    t = (user_text or "").strip().lower()
    if not t:
        return None, 0
    t = re.sub(r"\s+", " ", t)

    best_intent: Optional[str] = None
    best_score = 0

    for intent_key, kws in INTENT_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best_score = score
            best_intent = intent_key

    return (best_intent, best_score) if best_score >= 1 else (None, 0)




# -------------------------
# Availability: product-type locking to prevent category jumps (pants -> jacket)
# -------------------------
PRODUCT_TYPE_KEYWORDS = {
    "pants": ["pants", "training pants", "joggers", "leggings", "trousers", "sweatpants"],
    "shirts": ["shirt", "t-shirt", "tee", "top", "tank", "sports bra"],
    "jackets": ["jacket", "outerwear", "coat", "windbreaker"],
    "knitwear": ["knit", "sweater", "hoodie", "cardigan"],
}

def detect_product_type(text: str) -> Optional[str]:
    t = (text or "").lower()
    for ptype, kws in PRODUCT_TYPE_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return ptype
    return None


# -------------------------
# Knowledge base loader (LangChain)
# -------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore(data_dir: Path) -> Optional[Chroma]:
    if not data_dir.exists():
        return None

    docs = []

    md_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=False,
    )
    docs.extend(md_loader.load())

    json_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.json",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=False,
    )
    docs.extend(json_loader.load())

    if not docs:
        return None

    for d in docs:
        src = d.metadata.get("source", "")
        name = os.path.basename(src)
        d.metadata["intent"] = FILE_TO_INTENT.get(name, "general")
        d.metadata["filename"] = name

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=MODEL_EMBED, openai_api_key=API_KEY)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="styleloom_kb",
    )

vectorstore = build_vectorstore(DATA_DIR)


def retrieve_context(
    query: str,
    intent_key: Optional[str],
    k: int = 8,
    min_score: float = 0.25,
) -> str:
    if not vectorstore:
        return ""

    filt = None
    if intent_key and intent_key not in ("none", "other"):
        filt = {"intent": intent_key}

    try:
        hits = vectorstore.similarity_search_with_relevance_scores(query, k=k, filter=filt)
        filtered = [(d, s) for (d, s) in hits if s is not None and s >= min_score]
        if not filtered:
            return ""
        blocks = []
        for i, (d, s) in enumerate(filtered, 1):
            fn = d.metadata.get("filename", "unknown")
            blocks.append(f"[Doc{i} score={s:.2f} file={fn}]\n{d.page_content.strip()}")
        return "\n\n".join(blocks)
    except Exception:
        try:
            hits = vectorstore.similarity_search(query, k=k, filter=filt)
        except Exception:
            hits = vectorstore.similarity_search(query, k=k)

        if not hits:
            return ""
        blocks = []
        for i, d in enumerate(hits, 1):
            fn = d.metadata.get("filename", "unknown")
            blocks.append(f"[Doc{i} file={fn}]\n{d.page_content.strip()}")
        return "\n\n".join(blocks)


# -------------------------
# Deterministic scenario fallback + follow-up continuity
# -------------------------
FOLLOWUP_ACK_PAT = re.compile(
    r"^(sure|yes|yeah|yep|ok|okay|go ahead|please do|do it|sounds good|tell me|show me)\b",
    re.IGNORECASE,
)

TOPIC_SWITCH_PAT = re.compile(
    r"\b(switch|change)\s+(topic|topics|subject|category)\b",
    re.IGNORECASE,
)

def is_topic_switch_request(text: str) -> bool:
    return bool(TOPIC_SWITCH_PAT.search((text or "").strip()))

def is_generic_followup(text: str) -> bool:
    t = (text or "").strip()
    return (len(t) <= 18) and bool(FOLLOWUP_ACK_PAT.search(t))

def load_intent_files_as_context(intent_key: str) -> str:
    files = INTENT_TO_FILES.get(intent_key, [])
    if not files:
        return ""
    blocks = []
    for fn in files:
        fp = DATA_DIR / fn
        if fp.exists():
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                content = ""
            if content:
                blocks.append(f"[FILE: {fn}]\n{content}")
    return "\n\n".join(blocks)


# -------------------------
# LLM helpers
# -------------------------
def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
    resp = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def format_recent_history(chat_history: List[Tuple[str, str]], limit: int = 6) -> str:
    """
    Format the most recent turns for lightweight conversational continuity.
    Keeps the LLM aware of local context without turning this into a free-form chat model.
    """
    if not chat_history:
        return ""
    turns = chat_history[-limit:]
    lines = []
    for spk, msg in turns:
        role = "User" if spk == "User" else chatbot_speaker()
        lines.append(f"{role}: {msg}")
    return "\n".join(lines)


# -------------------------
# Study 2 (Relevant) understanding cue + sub-intent detection
# -------------------------
ACK_ROTATION = ["Got it.", "Understood.", "Sure.", "Okay."]

NEW_ARRIVALS_ITEMS = {
    "Soft Blouse": ["soft blouse"],
    "City Knit": ["city knit"],
    "Everyday Jacket": ["everyday jacket"],
    "Tailored Pants": ["tailored pants"],
    "Weekend Dress": ["weekend dress"],
}

def detect_active_item(text: str) -> Optional[str]:
    t = (text or "").lower()
    for canonical, kws in NEW_ARRIVALS_ITEMS.items():
        if any(kw in t for kw in kws):
            return canonical
    return None


def detect_subintent(user_text: str, intent_key: Optional[str], active_item: Optional[str] = None) -> Optional[str]:
    """
    Narrower intent hints used to:
      (1) improve retrieval queries, and
      (2) reduce mismatched answers in follow-up turns.
    """
    t = (user_text or "").lower()

    if intent_key == "shipping_returns":
        if re.search(r"\b(how much|cost|price|fee)\b", t):
            return "shipping_cost"
        if re.search(r"\b(how fast|how long|delivery time|shipping time|arrive|days)\b", t):
            return "shipping_time"
        if re.search(r"\b(ups|fedex|usps|ground|carrier)\b", t):
            return "shipping_carrier"
        if re.search(r"\b(return window|within \d+|within\b|\b\d+\s*days?\b)", t) and "return" in t:
            return "return_window"
        if re.search(r"\b(steps|process|how do i return|how to return)\b", t):
            return "return_steps"
        return None

    if intent_key == "promotions":
        if re.search(r"\b(exclude|excluded|exclusions)\b", t):
            return "promo_exclusions"
        if re.search(r"\b(apply|use code|promo field|checkout)\b", t):
            return "promo_apply"
        return None

    if intent_key == "rewards":
        if re.search(r"\b(cost|fee|price)\b", t):
            return "membership_cost"
        if re.search(r"\b(join|sign up|enroll|become a member|require)\b", t):
            return "membership_join"
        if re.search(r"\b(benefit|perks|discount)\b", t):
            return "membership_benefits"
        return None

    if intent_key == "new_arrivals":
        if active_item:
            return f"item_{active_item.replace(' ', '_').lower()}"
        if re.search(r"\b(picture|photo|image)\b", t):
            return "product_images"
        return None

    if intent_key == "availability":
        if re.search(r"\b(black|white|navy|blue|red|green|gray|grey|beige|brown|pink|purple|yellow|orange|cream|ivory)\b", t):
            return "availability_color"
        if re.search(r"\b(xs|s|small|m|medium|l|large|xl|x-?large|\d{1,2})\b", t):
            return "availability_size"
        return None

    return None


def pick_ack(turn_index: int) -> str:
    return ACK_ROTATION[turn_index % len(ACK_ROTATION)]


def extract_last_question(text_block: str) -> Optional[str]:
    """
    Naive extraction of the last question sentence, used to handle short follow-ups like 'Yes'.
    """
    if not text_block:
        return None
    # Split on line breaks then sentences.
    txt = re.sub(r"\s+", " ", text_block).strip()
    if "?" not in txt:
        return None
    parts = re.split(r"(?<=[\?])\s+", txt)
    qs = [p.strip() for p in parts if p.strip().endswith("?")]
    return qs[-1] if qs else None


def answer_grounded(
    user_text: str,
    context: str,
    intent_key: Optional[str] = None,
    subintent: Optional[str] = None,
    recent_history: str = "",
    pending_question: Optional[str] = None,
    include_ack: bool = True,
) -> str:
    """
    Relevant answer: KB-grounded where possible, without Study 2's short acknowledgment cue.
    Avoids mechanical parroting ("You're asking about ...") and category-log disclosures.
    """

    # Study 1 should not use Study 2's short acknowledgment cue (e.g., "Got it.").
    prefix = ""

    # Deterministic micro-overrides for common "missing detail" questions.
    # These reduce awkward clarification loops when the KB does not specify a requested field.
    low_ctx = (context or "").lower()
    t = (user_text or "").lower()

    if intent_key == "shipping_returns" and subintent == "shipping_cost":
        if ("$" not in context) and ("shipping cost" not in low_ctx) and ("shipping fee" not in low_ctx):
            core = (
                "Shipping fees are calculated at checkout based on your location and the shipping speed you choose. "
                "The policy materials list delivery timeframes but do not specify a flat shipping rate."
            )
            return f"{prefix} {core}".strip()

    if intent_key == "rewards" and subintent == "membership_cost":
        if ("$" not in context) and ("fee" not in low_ctx) and ("cost" not in low_ctx):
            core = (
                "The membership materials describe benefits and access features, but they do not list a membership fee. "
                "If you share what you are trying to access, I can point to the relevant membership benefit."
            )
            return f"{prefix} {core}".strip()

    system = f"""You are {CHATBOT_NAME}, Style Loom's virtual assistant in a controlled shopping Q&A study.

Use BUSINESS CONTEXT as the source of truth for brand-specific facts, policies, and item details.
If the requested brand-specific detail is not provided in the BUSINESS CONTEXT, state that plainly (no apology),
then provide the closest related information that IS in the context.

Output rules (Study 2: RELEVANT responses):
- Do NOT repeat the user's question or describe internal routing (no "You're asking about...", no "It looks like...").
- Be concise: 1–3 sentences (a brief acknowledgment may be added automatically).
- Ask at most ONE follow-up question, only if it is necessary to proceed.
- Keep tone neutral, professional, and natural. No emojis.

Intent: {intent_key or "unknown"}.
Sub-intent: {subintent or "none"}.
"""

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]

    if recent_history.strip():
        msgs.append({"role": "system", "content": f"RECENT CHAT (for continuity):\n{recent_history}"})

    if pending_question and is_generic_followup(user_text):
        msgs.append({"role": "system", "content": f"PREVIOUS ASSISTANT QUESTION: {pending_question}"})

    if context.strip():
        msgs.append({"role": "system", "content": f"BUSINESS CONTEXT:\n{context}"})

    msgs.append({"role": "user", "content": user_text})

    core = llm_chat(msgs, temperature=0.2).strip()

    # Final response: optional prefix + grounded core
    return f"{prefix} {core}".strip()


def answer_fallback(user_text: str, intent_key: Optional[str] = None) -> str:
    """
    Minimal, relevant fallback when retrieval yields no usable context.
    """
    # Keep short and non-mechanical.
    if intent_key in ("shipping_returns", "promotions", "rewards"):
        return "Could you share one more detail (for example, the item name or what part of the policy you want to confirm)?"
    if intent_key in ("size_fit", "availability", "new_arrivals"):
        return "Could you share the item name and, if relevant, your preferred size or color?"
    return "Could you share one more detail so I can help?"


# -------------------------
# Availability query specificity (to avoid triggering the fixed 5+ script on broad assortment questions)
# -------------------------
COLOR_WORDS = [
    "black","white","navy","blue","red","green","gray","grey","beige","brown","pink","purple","yellow","orange","cream","ivory"
]

# Size extraction: avoid false matches like the possessive "'s" in "women's"
_SIZE_ALPHA_CTX_RE = re.compile(
    r"(?:\bsize\b|\bin\s+(?:a\s+)?size\b|\bsize:\b)\s*(xs|s|m|l|xl|x-?large|small|medium|large)\b",
    re.IGNORECASE
)
_SIZE_NUM_CTX_RE = re.compile(
    r"(?:\bsize\b|\bin\s+(?:a\s+)?size\b|\bwaist\b|\binseam\b|\blength\b)\s*(\d{1,2}(?:/\d{1,2})?)\b",
    re.IGNORECASE
)
_SIZE_SLASH_RE = re.compile(r"\b\d{1,2}/\d{1,2}\b")

def extract_size(text: str) -> Optional[str]:
    """Extract a size only when the user explicitly indicates a size."""
    t = (text or "").strip().lower()
    if not t:
        return None

    # Common W/L style (e.g., 34/34) often appears without the word "size"
    m = _SIZE_SLASH_RE.search(t)
    if m:
        return m.group(0)

    # Numeric sizes when explicitly indicated
    m = _SIZE_NUM_CTX_RE.search(t)
    if m:
        return m.group(1)

    # Alpha sizes when explicitly indicated
    m = _SIZE_ALPHA_CTX_RE.search(t)
    if m:
        return m.group(1)

    return None

def is_specific_availability_query(text: str) -> bool:
    t = (text or "").lower()

    has_size = bool(extract_size(t))
    has_color = any(c in t for c in COLOR_WORDS)

    # Product / item mention (broad)
    has_item = any(w in t for w in [
        "dress","party dress","shirt","t-shirt","tee","top","jacket","coat","pants","trousers","joggers","leggings",
        "hoodie","sweater","cardigan","skirt","blazer","outerwear"
    ])

    # Broad assortment phrasing (should NOT trigger fixed 5+)
    broad = any(p in t for p in [
        "what different", "what kinds", "what type", "what types", "what do you have", "different product",
        "product availability", "available products", "different color", "different colors", "what colors", "colors do you have"
    ])

    if broad and not (has_size or has_color):
        return False

    return has_item and (has_size or has_color)



# -------------------------
# Availability 상담형 응답: RAG/DB가 없어도 자연스럽게 "필터링/좁혀가기"를 제공
# -------------------------
_AVAIL_COLOR_ALIASES = {
    "navy": "navy",
    "khaki": "khaki",
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "beige": "beige",
    "ivory": "ivory",
    "blue": "blue",
    "brown": "brown",
    "green": "green",
}
_AVAIL_ATTR_KEYWORDS = {
    "pleated": ["pleat", "pleated", "pleats"],
    "european": ["european", "euro", "italian", "french"],
    "slim": ["slim", "trim"],
    "tapered": ["taper", "tapered"],
    "regular": ["regular", "classic"],
    "high_rise": ["high rise", "high-rise"],
    "mid_rise": ["mid rise", "mid-rise"],
    "low_rise": ["low rise", "low-rise"],
}

def _update_availability_state(user_text: str) -> dict:
    state = st.session_state.get("availability_state") or {
        "product": None,       # e.g., pants, shirt, outerwear, dress, suiting
        "colors": set(),
        "size": None,          # raw size string
        "attrs": set(),        # e.g., pleated, european
        "turns": 0,
        "disclaimer_shown": False,
    }
    t = (user_text or "").lower()

    # broad product inference (run first; can be overridden by more specific matches below)
    if any(w in t for w in [
        "coat", "coats", "overcoat", "trench", "parka", "puffer", "outerwear", "raincoat", "topcoat"
    ]):
        state["product"] = "outerwear"
    if any(w in t for w in ["dress", "dresses", "sundress", "gown", "maxi", "midi"]):
        state["product"] = "dress"

    # product inference
    if any(w in t for w in ["pants", "trousers", "slacks"]):
        state["product"] = "pants"
    elif any(w in t for w in ["shirt", "shirts", "button-down", "button down", "dress shirt"]):
        state["product"] = "shirt"
    elif any(w in t for w in ["suit", "blazer", "sport coat", "suiting"]):
        state["product"] = "suiting"

    # colors
    for k,v in _AVAIL_COLOR_ALIASES.items():
        if k in t:
            state["colors"].add(v)

    # sizes
    sz = extract_size(t)
    if sz:
        state["size"] = sz

    # attributes
    for attr, kws in _AVAIL_ATTR_KEYWORDS.items():
        if any(kw in t for kw in kws):
            state["attrs"].add(attr)

    state["turns"] = int(state.get("turns", 0)) + 1
    st.session_state["availability_state"] = state
    return state

def _availability_next_question(state: dict) -> Optional[str]:
    # Ask at most one next question, prioritized to reduce loops.
    attrs = state.get("attrs", set())
    has_fit_pref = ("slim" in attrs) or ("regular" in attrs)

    if state.get("product") == "pants":
        if "pleated" in attrs:
            return "Do you prefer a **single pleat** (subtle) or **double pleat** (more structure)?"
        if "european" in attrs and (not has_fit_pref):
            return "For a European look, would you prefer a **slim-tapered** fit or a more **classic/regular** fit?"
        if not has_fit_pref:
            return "Do you want a **slim-tapered** fit or a more **classic/regular** fit?"
        # Next-best narrowing once fit is known
        if not state.get("colors"):
            return "Any color preference, such as **navy**, **black**, or **khaki**?"
        return "Do you prefer a **mid-rise** or **high-rise**, or should the focus stay on a clean, classic look?"
    if state.get("product") == "shirt":
        if not has_fit_pref:
            return "Do you prefer a **slim** fit or a more **regular** fit?"
        # Next-best narrowing once fit is known
        if not state.get("colors"):
            return "Any color preference, such as **white**, **light blue**, or **navy**?"
        return "Do you prefer a **crisper** feel or something **softer and more comfortable**?"
    if state.get("product") == "outerwear":
        return "What type of coat are you looking for: **wool overcoat**, **trench**, or **puffer/parka**?"
    if state.get("product") == "dress":
        if not state.get("colors"):
            return "Any color preference, and do you prefer a **midi** or **maxi** length?"
        return "Is this for a casual daytime look or a more **dressy** occasion?"
    if state.get("product") == "suiting":
        return "Are you looking for a **blazer/sport coat** or a full **suit**?"
    return "Are you shopping for **tops**, **bottoms**, **outerwear**, or **dresses**?"

def build_availability_consult_reply(user_text: str) -> str:
    prefix = pick_ack(int(st.session_state.get("bot_turns", 0) + 1))
    state = _update_availability_state(user_text)

    t = (user_text or "").lower()
    is_count_q = bool(re.search(r"\bhow many\b|\bnumber of\b|\bstock count\b|\bhow many\b.*\bin stock\b", t))

    # short summary (no fake inventory counts)
    product = state.get("product")
    colors = sorted(list(state.get("colors", set())))
    size = state.get("size")
    attrs = state.get("attrs", set())

    bits = []
    if product == "pants":
        bits.append("tailored trousers")
    elif product == "shirt":
        bits.append("dress shirts")
    elif product == "suiting":
        bits.append("suiting pieces")
    elif product == "outerwear":
        bits.append("men's coats")
    elif product == "dress":
        bits.append("dresses")
    else:
        bits.append("items")

    if colors:
        bits.append("in " + " and ".join(colors))
    if size:
        bits.append(f"(size {size})")
    if "european" in attrs:
        bits.append("with a European-style silhouette")
    if "pleated" in attrs:
        bits.append("with pleats")

    summary = " ".join(bits).strip()

    # Guidance: show full disclaimer once per availability flow to prevent repetitive loops.
    if not state.get("disclaimer_shown"):
        guidance = (
            "I can help you narrow down what to look for. "
            "This chat cannot display a live product list or exact stock counts, but you can use the filters on the product page to match your preferences."
        )
        state["disclaimer_shown"] = True
        st.session_state["availability_state"] = state
    else:
        guidance = "I can help you narrow it down using one quick filter."

    q = _availability_next_question(state)

    if is_count_q:
        return f"{prefix} I can’t see a live stock count inside this chat. For {summary}, the fastest approach is filtering by category and size on the product page. {guidance} {q}".strip()

    return f"{prefix} For {summary}, {guidance} {q}".strip()

def generate_answer(user_text: str, scenario: Optional[str]) -> Tuple[str, str, bool]:
    intent_key = scenario_to_intent(scenario)

    # Track active item for new-arrivals continuity (e.g., "Weekend Dress" follow-ups).
    item = detect_active_item(user_text)
    if item:
        st.session_state["active_item"] = item
    active_item = st.session_state.get("active_item")

    # Sub-intent hint (improves retrieval and reduces mismatched policy answers)
    subintent = detect_subintent(user_text, intent_key, active_item=active_item)

    # Lightweight continuity context
    recent_history = format_recent_history(st.session_state.get("chat_history", []), limit=6)
    pending_q = st.session_state.get("pending_question")

    # -------------------------
    # Follow-up continuity for short replies (e.g., "Yes", "Sure")
    # -------------------------
    if is_generic_followup(user_text):
        used_intent = st.session_state.get("last_intent_used") or intent_key
        used_subintent = st.session_state.get("last_subintent_used") or subintent
        ctx = st.session_state.get("last_kb_context", "")

        if (not ctx.strip()) and used_intent not in ("none", "other"):
            ctx = load_intent_files_as_context(used_intent)

        if ctx.strip():
            ans = answer_grounded(
                user_text,
                ctx,
                intent_key=used_intent,
                subintent=used_subintent,
                recent_history=recent_history,
                pending_question=pending_q,
                include_ack=False,
            )
            st.session_state["last_kb_context"] = ctx
            st.session_state["last_intent_used"] = used_intent
            st.session_state["last_subintent_used"] = used_subintent
            return ans, used_intent, True

        # No usable context to continue with
        st.session_state["last_kb_context"] = ""
        st.session_state["last_intent_used"] = used_intent
        st.session_state["last_subintent_used"] = used_subintent
        return answer_fallback(user_text, intent_key=used_intent), used_intent, False

    # -------------------------
    # Availability: 상담형(컨설팅) 응답으로 전환 (DB/RAG 없을 때도 반복 루프 방지)
    # -------------------------
    if intent_key == "availability":
        reply = build_availability_consult_reply(user_text)
        st.session_state["last_kb_context"] = ""
        st.session_state["last_intent_used"] = intent_key
        st.session_state["last_subintent_used"] = subintent
        return reply.strip(), intent_key, False

    # -------------------------
    # Retrieval query shaping (improves relevance without exposing internal routing)
    # -------------------------
    query_for_search = user_text

    # Availability bias by locked product type
    if intent_key == "availability":
        ptype = st.session_state.get("active_product_type")
        if ptype:
            query_for_search = f"{ptype} {query_for_search}"

    # New arrivals: pin retrieval to the last mentioned item if present
    if intent_key == "new_arrivals" and active_item and (active_item.lower() not in query_for_search.lower()):
        query_for_search = f"{active_item} {query_for_search}"

    # Sub-intent cue for retrieval
    if subintent:
        query_for_search = f"{subintent.replace('_', ' ')} {query_for_search}"

    context = ""
    used_kb = False

    # 1) Vector retrieval (filtered by intent)
    if vectorstore:
        context = retrieve_context(query_for_search, intent_key=intent_key, k=8, min_score=0.25)
        if context.strip():
            used_kb = True

    # 2) Deterministic fallback (load all files for that intent)
    if not context.strip() and intent_key not in ("none", "other"):
        context = load_intent_files_as_context(intent_key)
        if context.strip():
            used_kb = True

    # 3) Fallback when nothing is available
    if not context.strip():
        if intent_key == "availability":
            prefix = pick_ack(int(st.session_state.get("bot_turns", 0) + 1))
            reply = (
                f"{prefix} "
                "We offer a range of items across key categories. "
                "Is there a specific item, color, or size you are looking for?"
            )
            st.session_state["last_kb_context"] = ""
            st.session_state["last_intent_used"] = intent_key
            st.session_state["last_subintent_used"] = subintent
            return reply.strip(), intent_key, False

        st.session_state["last_kb_context"] = ""
        st.session_state["last_intent_used"] = intent_key
        st.session_state["last_subintent_used"] = subintent
        return answer_fallback(user_text, intent_key=intent_key), intent_key, False

    # Grounded answer
    ans = answer_grounded(
        user_text,
        context,
        intent_key=intent_key,
        subintent=subintent,
        recent_history=recent_history,
        pending_question=pending_q,
        include_ack=False,
    )

    # Persist
    st.session_state["last_kb_context"] = context
    st.session_state["last_intent_used"] = intent_key
    st.session_state["last_subintent_used"] = subintent

    return ans, intent_key, used_kb



# -------------------------
# Session state initialization
# -------------------------
defaults = {
    "chat_history": [],
    "session_id": uuid.uuid4().hex[:10],
    "greeted_once": False,
    "ended": False,
    "rating_saved": False,
    "user_turns": 0,
    "bot_turns": 0,
    "last_user_selected_scenario": "— Select a scenario —",
    "active_scenario": None,
    "switch_log": [],
    "session_started_logged": False,
    "last_kb_context": "",
    "last_intent_used": None,
    "last_subintent_used": None,
    "active_product_type": None,
    "active_item": None,
    "pending_question": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v



# -------------------------
# Supabase: log session start ONCE
# -------------------------
def log_session_start_once():
    if st.session_state.session_started_logged:
        return

    ts_now = datetime.datetime.utcnow().isoformat() + "Z"
    supabase.table(TBL_SESSIONS).upsert({
        "session_id": st.session_state.session_id,
        "ts_start": ts_now,
        "identity_option": identity_option,
        "brand_type": brand_type, 
        "name_present": "present" if show_name else "absent",
        "picture_present": "present" if show_picture else "absent",
    }).execute()

    st.session_state.session_started_logged = True


# -------------------------
# Greeting (first assistant message) - EXACT TEXT YOU PROVIDED
# -------------------------
if not st.session_state.greeted_once:
    log_session_start_once()

    greet_text = (
        "Hi, I'm Skyler, Style Loom’s virtual assistant. "
        "Style Loom is a mass-market fashion brand founded twenty years ago, "
        "known for its accessibility and broad consumer reach. "
        "I’m here to help with your shopping."
    )
    st.session_state.chat_history.append((chatbot_speaker(), greet_text))
    st.session_state.greeted_once = True


# -------------------------
# UI: scenario dropdown
# -------------------------
st.markdown("**How can I help you today?**")

selected = st.selectbox(
    "Choose a topic:",
    options=SCENARIOS,
    index=SCENARIOS.index(st.session_state.last_user_selected_scenario)
    if st.session_state.last_user_selected_scenario in SCENARIOS else 0,
)

prev_selected = st.session_state.last_user_selected_scenario
st.session_state.last_user_selected_scenario = selected

# Confirmation message when user changes category
if selected != "— Select a scenario —" and selected != prev_selected:
    st.session_state.active_scenario = selected

    if selected != "Check product availability":
        st.session_state.active_product_type = None

    confirm_text = f"Sure, I will help you with **{selected}**. Please ask me a question."
    st.session_state.chat_history.append((chatbot_speaker(), confirm_text))

st.divider()


# -------------------------
# Render chat history (TEXT ONLY; no chat bubbles/icons)
# -------------------------
for spk, msg in st.session_state.chat_history:
    if spk == chatbot_speaker():
        st.markdown(f"**{spk}:** {msg}")
    else:
        st.markdown("**User:** " + msg)


# -------------------------
# Chat input
# -------------------------
user_text = None
if not st.session_state.ended:
    user_text = st.chat_input("Type your message here...")


# -------------------------
# End button and rating UI
# -------------------------
end_col1, end_col2 = st.columns([1, 2])

with end_col1:
    can_end = (st.session_state.user_turns >= MIN_USER_TURNS) and (not st.session_state.ended)
    if st.button("End chat", disabled=not can_end):
        st.session_state.ended = True

with end_col2:
    if not st.session_state.ended:
        completed = st.session_state.user_turns
        remaining = max(0, MIN_USER_TURNS - completed)

        if remaining > 0:
            st.caption(
                f"Please complete at least {MIN_USER_TURNS} user turns before ending the chat. "
                f"Progress: {completed}/{MIN_USER_TURNS} (need {remaining} more)."
            )
        else:
            st.caption(f"Progress: {completed}/{MIN_USER_TURNS}. You can end the chat now.")

# -------------------------
# Save ONLY at the end
# -------------------------
# -------------------------
# Save ONLY at the end (transcripts + satisfaction + sessions end)
# -------------------------
if st.session_state.ended and not st.session_state.rating_saved:
    rating = st.slider("Overall satisfaction with the chatbot (1 = very low, 7 = very high)", 1, 7, 4)
    prolific_id = st.text_input("Prolific ID", value="")

    if st.button("Submit rating and save"):
        ts_now = datetime.datetime.utcnow().isoformat() + "Z"

        final_scenario = st.session_state.active_scenario or (
            selected if selected != "— Select a scenario —" else "Other"
        )

        # ===== Transcript text (human-readable; same style as your older version) =====
        transcript_lines = []
        transcript_lines.append("===== Session Transcript =====")
        transcript_lines.append(f"timestamp       : {ts_now}")
        transcript_lines.append(f"session_id      : {st.session_state.session_id}")
        transcript_lines.append(f"identity_option : {identity_option}")
        transcript_lines.append(f"brand_type      : {brand_type}")
        transcript_lines.append(f"name_present    : {'present' if show_name else 'absent'}")
        transcript_lines.append(f"picture_present : {'present' if show_picture else 'absent'}")
        transcript_lines.append(f"scenario        : {final_scenario}")
        transcript_lines.append(f"user_turns      : {st.session_state.user_turns}")
        transcript_lines.append(f"bot_turns       : {st.session_state.bot_turns}")
        transcript_lines.append(f"prolific_id     : {(prolific_id.strip() or 'N/A')}")
        transcript_lines.append("")
        transcript_lines.append("---- Switch log ----")
        transcript_lines.append(json.dumps(st.session_state.switch_log, ensure_ascii=False))
        transcript_lines.append("")
        transcript_lines.append("---- Chat transcript ----")
        for spk, msg in st.session_state.chat_history:
            transcript_lines.append(f"{spk}: {msg}")
        transcript_lines.append("")
        transcript_lines.append(f"Satisfaction (1-7): {int(rating)}")

        transcript_text = "\n".join(transcript_lines)

        # 1) Save transcript (NOW includes satisfaction column)
        supabase.table(TBL_TRANSCRIPTS).insert({
            "session_id": st.session_state.session_id,
            "ts": ts_now,
            "transcript_text": transcript_text,
            "satisfaction": int(rating),   # <-- NEW COLUMN in transcripts
        }).execute()

        # 2) Save rating in separate table (keep this)
        supabase.table(TBL_SATISFACTION).insert({
            "session_id": st.session_state.session_id,
            "ts": ts_now,
            "rating": int(rating),
        }).execute()

        # 3) Update session end + turns + scenario + prolific_id
        supabase.table(TBL_SESSIONS).upsert({
            "session_id": st.session_state.session_id,
            "ts_end": ts_now,
            "scenario": final_scenario,
            "user_turns": st.session_state.user_turns,
            "bot_turns": st.session_state.bot_turns,
            "prolific_id": prolific_id.strip() or None,
        }).execute()

        st.session_state.rating_saved = True
        st.success("Saved. Thank you.")

# -------------------------
# Main interaction
# -------------------------
if user_text and not st.session_state.ended:
    st.session_state.chat_history.append(("User", user_text))
    st.session_state.user_turns += 1

    user_selected = selected if selected != "— Select a scenario —" else None
    active = st.session_state.active_scenario or user_selected or "Other"

    # Availability: lock product type when explicitly mentioned
    if active == "Check product availability":
        ptype = detect_product_type(user_text)
        if ptype:
            st.session_state.active_product_type = ptype

    # Optional auto-switch (internal only; no disclosure text).
    # We switch only when confidence is reasonably high OR the user explicitly requests a topic switch.
    detected_intent, detected_score = detect_intent(user_text)
    detected_scenario = INTENT_TO_SCENARIO.get(detected_intent) if detected_intent else None
    switch_req = is_topic_switch_request(user_text)

    if detected_scenario and (detected_scenario != active):
        should_switch = False

        # If the user did not choose a scenario (or is on "Other"), switching helps.
        if (user_selected is None) or (active == "Other"):
            should_switch = True

        # If the user explicitly requests switching topics, allow switching on weaker evidence.
        if switch_req and detected_score >= 1:
            should_switch = True

        # If the user did choose a topic, require stronger evidence to override it.
        if (user_selected is not None) and (detected_score >= 2):
            should_switch = True

        if should_switch:
            st.session_state.switch_log.append({
                "ts": datetime.datetime.utcnow().isoformat() + "Z",
                "user_selected_scenario": user_selected,
                "from_scenario": active,
                "to_scenario": detected_scenario,
                "detected_intent": detected_intent,
                "detected_score": detected_score,
                "user_text": user_text,
            })
            active = detected_scenario
            st.session_state.active_scenario = active

            if active != "Check product availability":
                st.session_state.active_product_type = None

    answer, used_intent, used_kb = generate_answer(user_text, scenario=active)

    # Track the last assistant question to support short follow-ups like "Yes".
    st.session_state["pending_question"] = extract_last_question(answer)

    st.session_state.chat_history.append((chatbot_speaker(), answer))
    st.session_state.bot_turns += 1

    st.rerun()
























