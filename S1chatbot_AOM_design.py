# =========================
# Chatbot App with Name and Picture + Mass-market — RAG + Rules + Pending Intents
# Natural UI flow (no explicit step sections)
# =========================

# --- Imports ---
import os
import re
import uuid
import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI
from supabase import create_client

# LangChain / Vector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# =========================
# Streamlit Page Config (place early)
# =========================
st.set_page_config(page_title="Style Loom — Chatbot Experiment", layout="centered")


# =========================
# Session-state initialization (must be above any session_state usage)
# =========================
defaults = {
    "chat_history": [],
    "session_id": uuid.uuid4().hex[:10],     # 새 세션마다 생성
    "awaiting_feedback": False,
    "ended": False,
    "saved_fpath": None,
    "rating_saved": False,
    "greeted_once": False,
    "scenario_selected_once": False,
    "last_user_selected_scenario": "— Select a scenario —",
    "user_turns": 0,
    "bot_turns": 0,
    "closing_asked": False,
    "flow": {
        "scenario": None, "stage": "start",
        "slots": {
            "product": None, "color": None, "size": None,
            "contact_pref": None, "tier_known": None, "selected_collection": None,
            "return_item": None, "received_date": None, "return_reason": None
        }
    },
    "pending": {"intent": None, "data": {}},
    "session_meta_logged": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 최소 대화 턴 수
MIN_USER_TURNS = 5


# =========================
# OpenAI Client
# =========================
API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not API_KEY:
    st.error("OPENAI_API_KEY is not set. Please configure it in environment variables or st.secrets.")
    st.stop()
client = OpenAI(api_key=API_KEY)


# =========================
# Supabase Client (single, cached)
# =========================
SUPA_URL = st.secrets.get("SUPABASE_URL")
SUPA_KEY = st.secrets.get("SUPABASE_ANON_KEY")

if not SUPA_URL or not SUPA_KEY:
    st.error("Supabase credentials are missing. Please set SUPABASE_URL and SUPABASE_ANON_KEY in st.secrets.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_supabase():
    return create_client(SUPA_URL, SUPA_KEY)

supabase = get_supabase()


# =========================
# Branding (small, logo-like)
# =========================
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:8px;margin:8px 0 4px 0;">
        <div style="font-weight:700;font-size:20px;letter-spacing:0.3px;">Style Loom</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Identity
# =========================
identity_option = "With name and image"
show_name = True
show_picture = True
CHATBOT_NAME = "Skyler"
CHATBOT_PICTURE = "https://i.imgur.com/4uLz4FZ.png"
brand_type = "Mass-market Brand"

def _chatbot_speaker():
    return CHATBOT_NAME

if show_picture:
    try:
        st.image(CHATBOT_PICTURE, width=84)
    except Exception:
        pass


# =========================
# Initial greeting (appears first in chat)
# =========================
if not st.session_state.greeted_once:
    greet_text = (
        "Hi, I'm Skyler, Style Loom’s virtual assistant. "
        "Style Loom is a mass-market fashion brand founded twenty years ago, "
        "known for its accessibility and broad consumer reach. "
        "I’m here to help with your shopping."
    )
    st.session_state.chat_history.append((_chatbot_speaker(), greet_text))
    st.session_state.greeted_once = True


# --- Record session meta to Supabase (run once at start) ---
if not st.session_state.session_meta_logged:
    _payload = {
        "session_id": st.session_state.session_id,
        "ts_start": datetime.datetime.utcnow().isoformat() + "Z",
        "identity_option": identity_option,
        "brand_type": brand_type,
        "name_present": "present",
        "picture_present": "present",
        "scenario": st.session_state.flow.get("scenario") or None,
        "user_turns": st.session_state.user_turns,
        "bot_turns": st.session_state.bot_turns,
    }
    try:
        supabase.table("sessions").insert(_payload).execute()
        st.session_state.session_meta_logged = True
    except Exception as e:
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            st.session_state.session_meta_logged = True
        else:
            st.warning(f"(non-blocking) Failed to insert session meta: {e}")


# =========================
# Tone / Categories
# =========================
TONE = "informal"
TONE_STYLE = {
    # informal: 친근하지만 군더더기 없는 톤, 이모지는 '최대 1개', 문장 맨 끝에만 사용
    "informal": "Use a friendly, concise tone. Use at most one emoji per reply and place it only at the very end when it truly adds warmth. Do not start with 'Hey there'.",
    # formal: 이모지 금지
    "formal": "Use a formal, respectful tone. No emojis."
}

PRODUCT_CATEGORIES = [
    "blouse", "skirt", "pants", "cardigans / sweaters", "dresses",
    "jumpsuits", "jackets", "t-shirts", "sweatshirt / sweatpants",
    "outer", "coat / trenches", "tops / bodysuits", "activewear",
    "shirts", "shorts", "lingerie", "etc."
]


# =========================
# Regex & Extractors
# =========================
YES_PAT = re.compile(r"\b(yes|yeah|yep|sure|ok|okay|please)\b", re.I)
NO_PAT  = re.compile(r"\b(no|nope|nah|not now|later)\b", re.I)

def _is_size_chart_query(t: str) -> bool:
    """Detects 'size chart/guide' style questions anywhere in the text."""
    return bool(re.search(
        r"\b(size\s*(chart|guide)|sizing\s*(chart|guide)?|size\s*info|measurement(s)?)\b",
        t or "", re.I
    ))

def _preprocess_user_text(t: str) -> str:
    """Light normalization: common typos, synonyms, season words, and separators."""
    s = (t or "").strip()

    # 공백/슬래시 정리
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*/\s*", " / ", s)  # "City Knit / s" → "City Knit / s"

    # 계절/수식어(제품 추론에 불필요) 제거
    seasonals = [
        r"\bfall\b", r"\bautumn\b", r"\bwinter\b", r"\bspring\b", r"\bsummer\b",
        r"\bnew\b", r"\blatest\b", r"\brecent\b", r"\bcollection\b", r"\bnew\s+arrivals?\b"
    ]
    for pat in seasonals:
        s = re.sub(pat, " ", s, flags=re.I)

    # 자주 나오는 오타 보정
    fixes = {
        r"\boatmilk\b": "oatmeal",
        r"\boat meal\b": "oatmeal",
        r"\bgre(y|ie)ge\b": "greige",
    }
    for pat, repl in fixes.items():
        s = re.sub(pat, repl, s, flags=re.I)

    # 대표 제품명은 대소문자 섞여도 표준 표기로 정규화
    s = re.sub(r"\bcity\s+knit\b", "City Knit", s, flags=re.I)
    s = re.sub(r"\bsoft\s+blouse\b", "Soft Blouse", s, flags=re.I)
    s = re.sub(r"\beveryday\s+jacket\b", "Everyday Jacket", s, flags=re.I)
    s = re.sub(r"\btailored\s+pants?\b", "Tailored Pants", s, flags=re.I)
    s = re.sub(r"\bweekend\s+dress\b", "Weekend Dress", s, flags=re.I)

    return s.strip()

def extract_color(t: str):
    m = re.search(
        r"\b(black|white|ivory|navy|blue|mist\s?blue|greige|beige|red|green|rose\s?beige|pink|cream|sand|olive|charcoal|oatmeal|forest|berry|ink|brown|purple|orange|yellow|khaki|teal|burgundy|maroon|grey|gray)\b",
        t or "", re.I
    )
    return m.group(1).lower() if m else None

def extract_size(t: str):
    text = (t or "").lower()
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
    m = re.search(r"\b(XXS|XS|S|M|L|XL|XXL|0|2|4|6|8|10|12|14|16|18)\b", t or "", re.I)
    return m.group(1).upper() if m else None

def extract_product(t: str):
    text = _preprocess_user_text(t)
    low = text.lower()

    # 1) 명명된 라인업(정확 표기 우선)
    named = [
        "City Knit",
        "Soft Blouse",
        "Everyday Jacket",
        "Tailored Pants",
        "Weekend Dress",
    ]
    for name in named:
        if re.search(rf"\b{re.escape(name)}\b", text, re.I):
            return name

    # 2) 느슨한 키워드/동의어 → 대표 카테고리로 매핑
    #    예: knit → sweater (City Knit가 문맥에 있으면 City Knit)
    if "knit" in low:
        if "city" in low:   # "city knit", "city blue knit" 등
            return "City Knit"
        return "sweater"     # 일반 니트는 스웨터로 표준화

    if "tee" in low or "t-shirt" in low or "tshirt" in low:
        return "t-shirt"

    # ✅ shoes / sneakers 인식
    if re.search(r"\b(running\s+shoes?|sneakers?|shoes?)\b", low, re.I):
        return "shoes"

    # 3) 일반 카테고리(복수형은 단수로 정규화)
    cats = [
        "blouse", "skirt", "pants", "cardigan", "cardigans", "sweater", "sweaters",
        "dress", "dresses", "jumpsuit", "jumpsuits", "jacket", "jackets",
        "t-shirt", "t-shirts", "sweatshirt", "sweatpants", "outer", "coat",
        "trench", "trenches", "top", "tops", "bodysuit", "bodysuits",
        "activewear", "shirt", "shirts", "shorts", "lingerie", "shoes"
    ]
    for c in cats:
        if re.search(rf"\b{re.escape(c)}\b", low, re.I):
            if c in ["cardigans", "sweaters", "jackets", "dresses", "tops", "shirts", "jumpsuits"]:
                return c.rstrip("s")
            return c

    # 4) 두 단어 조합(느슨): "chunky knit sweater", "leather jacket" 등
    w = re.search(r"\b([\w\-]+(?:\s+[\w\-]+)?)\s+(jacket|skirt|blouse|t-?shirt|dress|pants|sweater|shoes)\b", text, re.I)
    if w:
        noun = w.group(2).lower()
        if noun == "t-shirt":
            return "t-shirt"
        if noun == "sweater":
            return "sweater"
        return noun

    return None

def _update_slots_from_text(user_text: str):
    cleaned = _preprocess_user_text(user_text)

    slots = st.session_state.flow["slots"]
    p = extract_product(cleaned)
    c = extract_color(cleaned)
    s = extract_size(cleaned)

    if p:
        slots["product"] = p
    if c:
        slots["color"] = c
    if s:
        slots["size"] = s


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
# RAG: Build/Load Vectorstore
# =========================
RAG_DIR = str(Path.cwd() / "rag_docs")

@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore(rag_dir: str):
    rag_path = Path(rag_dir)
    if not rag_path.exists():
        return None

    persist_dir = str(rag_path / ".chroma")
    embeddings = OpenAIEmbeddings(api_key=API_KEY, model="text-embedding-3-small")

    # Try loading existing index first
    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        try:
            return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        except Exception as e:
            st.warning(f"Vectorstore load warning: {e}")

    # === changed: load only human-authored text/markdown docs ===
    try:
        # Load .txt first
        loader = DirectoryLoader(
            rag_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=True,
            use_multithreading=True,
        )
        # Optionally add .md files as well
        try:
            md_loader = DirectoryLoader(
                rag_dir,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"autodetect_encoding": True},
                show_progress=True,
                use_multithreading=True,
            )
            docs = loader.load() + md_loader.load()
        except Exception:
            docs = loader.load()
    except Exception as e:
        st.warning(f"RAG documents could not be loaded: {e}")
        return None
    # === end changed ===

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    try:
        return Chroma.from_documents(
            chunks,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
    except Exception as e:
        st.warning(f"Vectorstore build failed: {e}")
        return None

vectorstore = build_or_load_vectorstore(RAG_DIR)

def retrieve_context(query: str, k: int = 6) -> str:
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

# === changed: enrich query with scenario/intent & hints ===
def make_query(user_message: str) -> str:
    slots = st.session_state.flow["slots"]
    scenario = st.session_state.flow.get("scenario")
    intent = detect_global_intent(user_message)
    intent_key = intent["key"] if intent else None

    parts = []
    if scenario:
        parts.append(f"scenario:{scenario}")
    if intent_key:
        parts.append(f"intent:{intent_key}")

    if slots.get("product"): parts.append(f"product:{slots['product']}")
    if slots.get("color"):   parts.append(f"color:{slots['color']}")
    if slots.get("size"):    parts.append(f"size:{slots['size']}")

    # light hints to steer retrieval to the right doc
    text = _preprocess_user_text(user_message)
    hint_terms = []
    if intent_key == "new_arrivals_intent":
        hint_terms += ["new arrivals", "new collection", "latest drop", "fall collection", "winter collection"]
    if intent_key == "promotions_intent":
        hint_terms += ["promotions", "discounts", "WELCOME10", "stacking", "first-time 5%"]
    if intent_key == "price_intent":
        hint_terms += ["price range", "typical ranges", "tees", "knitwear", "jackets", "dresses"]
    if intent_key == "free_returns_intent":
        hint_terms += ["free returns", "return shipping covered"]
    if intent_key == "mens_catalog_intent":
        hint_terms += ["men and women", "men's categories", "shirts", "pants", "jackets", "knitwear"]

    if hint_terms:
        parts.append("hints:" + ",".join(hint_terms))

    parts.append(f"user:{text}")
    return " | ".join(parts)
# === end changed ===


# =========================
# LLM (RAG) fallback
# =========================

def answer_with_rag(user_message: str) -> str:
    # 슬롯 업데이트
    _update_slots_from_text(user_message)

    # RAG 조회
    query = make_query(user_message)
    context = retrieve_context(query, k=6)

    # --- Style instruction 안전 획득 ---
    # tone_instruction() 헬퍼가 있으면 사용, 없거나 오타면 TONE_STYLE로 폴백
    try:
        style_instruction = tone_instruction()  # 헬퍼가 정상 정의되어 있으면 이 줄이 실행됨
    except NameError:
        # 헬퍼 없음/오타 ↔ 안전 폴백
        try:
            style_instruction = TONE_STYLE.get(TONE, TONE_STYLE["informal"])
        except Exception:
            # 최후 폴백(사전 자체가 없다면)
            style_instruction = "Use a friendly, casual tone. Use emojis."

    # 세션 상태 방어적 참조
    flow = getattr(st.session_state, "flow", {}) or {}
    known_slots = flow.get("slots", {})
    current_scenario = flow.get("scenario")

    bot_identity = f"named {CHATBOT_NAME}" if show_name else "with no name"

    prompt = f"""
You are a helpful customer service chatbot {bot_identity} for Style Loom.
Ground every answer in the BUSINESS CONTEXT. If critical info is missing, ask **one concise follow-up question** only.
Do not invent policy, numbers, or SKUs. Keep answers short and helpful.

=== BUSINESS CONTEXT (retrieved) ===
{context if context else "[no docs retrieved]"}
=== END CONTEXT ===

Meta:
- Current scenario: {current_scenario}
- Product categories: {", ".join(PRODUCT_CATEGORIES)}
- Known slots: {known_slots}

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
# Pending yes/no logic (global)
# =========================
def set_pending(intent: str, data: dict | None = None):
    # 불필요한 중첩 제거: {"intent": ..., "data": {...}} 한 단계만 유지
    st.session_state.pending = {"intent": intent, "data": (data or {})}

def consume_pending():
    p = st.session_state.pending
    st.session_state.pending = {"intent": None, "data": {}}
    return p

def ask_yesno(intent: str, message: str, data: dict | None = None) -> str:
    set_pending(intent, data or {})
    return message

def handle_pending_yes(user_text: str) -> str | None:
    pend = st.session_state.pending
    intent = pend.get("intent")
    if not intent:
        return None

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
            pend_consumed = consume_pending()
            target = pend_consumed["data"].get("target")  # ✅ 여기! data["data"] 아님
            if target:
                st.session_state.flow = {
                    "scenario": target, "stage": "start",
                    "slots": { **st.session_state.flow["slots"] }
                }
                return f"Great—switching to **{target}**. How can I help within this topic?"
            return "Okay—switching contexts."

        # 기타 펜딩 의도에 대해 기본 응답
        consume_pending()
        return "Got it."

    if NO_PAT.search(user_text):
        if intent == "confirm_switch":
            consume_pending()
            return "No problem—let’s continue with the current topic."
        consume_pending()
        return "All set! If you want the details later, just ask."

    return None


# =========================
# Auto-pending inference (scenario-aware)
# =========================
def infer_pending_from_bot_reply(reply_text: str) -> None:
    sc = (st.session_state.flow.get("scenario") or "").strip()
    text = (reply_text or "").strip().lower()
    if not text:
        return

    def _match_any(patterns):
        return any(re.search(p, text, re.I) for p in patterns)

    if sc == "Rewards & membership":
        if _match_any([
            r"\bwant to know\b.*\b(earn|earning|redeem|redemption|points|rewards)\b",
            r"\bwould you like\b.*\b(earn|earning|redeem|redemption|points|rewards)\b",
        ]):
            set_pending("rewards_more")
            return

    if sc in ("Check product availability", "Size & fit guidance"):
        if _match_any([
            r"\bwant to know\b.*\b(color|colors|size|sizes)\b",
            r"\binterested in\b.*\b(color|colors|size|sizes)\b",
            r"\bwould you like\b.*\b(color|colors|size|sizes)\b",
        ]):
            set_pending("colors_sizes_more")
            return
    return


# =========================
# Shipping intent detector
# =========================
try:
    _is_shipping_query
except NameError:
    def _is_shipping_query(t: str) -> bool:
        """Detect shipping/delivery questions (exclude 'free return(s)' and 'return shipping')."""
        text = (t or "")
        # exclude explicit free-returns phrasing so it doesn't fall into generic shipping
        if re.search(r"\bfree\s+return(s)?(\s+shipping)?\b", text, re.I):
            return False
        if re.search(r"\breturn\s+shipping\b", text, re.I):
            return False
        return bool(re.search(
            r"\b(ship|shipping|deliver(y|ed|ing)?|eta|track(ing)?|when\s+will\s+it\s+(arrive|be\s+delivered)|how\s+long.*(deliver|shipping|arrive))\b",
            text,
            re.I
        ))

# =========================
# Rule-based scenario router
# =========================
def route_by_scenario(current_scenario: str, user_text: str) -> str | None:
    flow = st.session_state.flow
    slots = flow["slots"]
    stage = flow.get("stage") or "start"

    _update_slots_from_text(user_text)

    # ---- Rewards & membership ----
    if current_scenario == "Rewards & membership":
        if stage in (None, "start"):
            flow["stage"] = "rewards_intro"
            return ask_yesno(
                intent="rewards_more",
                message=(
                    "Every 100 points = $1 off your next order (merchandise subtotal only). "
                    "Tiers are based on a rolling 12-month spend, and points expire after 12 months of inactivity. "
                    "Would you like to know more about earning or redeeming points?"
                )
            )

        # ✅ NEW: "free shipping?"
        if re.search(r"\bfree\s+(express\s+)?shipping\b", user_text, re.I):
            return (
                "Gold includes **Free Express Shipping** (≥ $800/year). "
                "VIP also includes Free Express plus gifts (≥ $1,500/year)."
            )

        # ✅ NEW: "how do I get to VIP faster?" 
        if (
            re.search(r"\b(get|reach|make|hit|unlock)\b.*\bvip\b.*\b(fast|faster|quick|quickly|sooner)\b", user_text, re.I)
            or re.search(r"\bvip\b.*\b(fast|faster|quick|quickly|sooner)\b", user_text, re.I)
        ):
            return (
                "**Getting to VIP faster:**\n"
                "- Consolidate purchases on one account (points apply to the discounted subtotal).\n"
                "- Shop during promos; points still apply to the **discounted** subtotal.\n"
                "- Gold starts at **$800/yr** (2× points); VIP at **$1,500/yr** (2× + Free Express + gifts)."
            )

        # prior logic
        if re.search(r"\b(earn|earning|accumulate|faster)\b", user_text, re.I):
            return (
                "**Earning faster:**\n"
                "- Silver: 1.5× points (≥ $300/year)\n"
                "- Gold: 2× points + Free Express (≥ $800/year)\n"
                "- VIP: 2× points + Free Express + Gifts (≥ $1,500/year)\n"
                "Promotions may stack, but points apply to the **discounted subtotal**."
            )

        if re.search(r"\b(redeem|redemption|use points|apply points)\b", user_text, re.I):
            return (
                "**Redeeming points:**\n"
                "- 100 pts = $1 off\n"
                "- Apply at checkout (payment step)\n"
                "- Points do not apply to taxes or shipping"
            )

        return None

    # ---- Check product availability ----
    if current_scenario == "Check product availability":
        if stage == "start":
            flow["stage"] = "collect"
            stage = "collect"

        if stage == "collect":
            if not slots.get("product"):
                _update_slots_from_text(user_text)
            if not slots.get("product"):
                return "Sure—what product are you looking for (e.g., jacket, dress, t-shirt)?"
            if not slots.get("color"):
                return f"Great—what color of {slots['product']}?"
            if not slots.get("size"):
                return f"What size for the {slots['product']} in {slots['color']}?"
            flow["stage"] = "offer_low_stock_alt"
            return ask_yesno(
                intent="colors_sizes_more",
                message=(
                    f"We have 5+ in stock for the {slots['product']} in {slots['color']} / {slots['size']}. "
                    "However, another color is running low in stock. Would you like me to suggest a low-stock option?"
                ),
                data={"slots_snapshot": slots.copy()}
            )

        if stage == "offer_low_stock_alt":
            if YES_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return (
                    f"A similar last-season {slots['product']} in {slots['color']} is down to the final 2 pieces. "
                    "Would you like a restock alert or see similar styles?"
                )
            if NO_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return "All set! Anything else you’d like to check?"
            return "Would you like me to suggest a similar low-stock option?"

        if stage == "end_or_more":
            return "Happy to help. Anything else you’d like to check?"
        return None

    # ---- New arrivals & collections ----
    if current_scenario == "New arrivals & collections":
        flow["stage"] = "end_or_more"
        return (
            "Soft Blouse is a cotton-modal with a softly draped silhouette in Ivory, Mist Blue, Rose Beige, and Black. "
            "City Knit is a cozy wool-blend in Oatmeal, Charcoal, Forest, and Pink. "
            "Everyday Jacket has clean tailoring in twill/nylon—Sand, Navy, Olive. "
            "Tailored Pants are cut in stretch twill—Black, Greige, Navy. "
            "Weekend Dress is fluid modal jersey—Berry, Ink, Cream. "
            "Want details on a specific item or color?"
        )

    # ---- Shipping & returns ----
    if current_scenario == "Shipping & returns":
        # Shipping question detection — prioritized over return flow
        if _is_shipping_query(user_text):
            flow["stage"] = "end_or_more"
            return (
                "**Shipping overview:**\n"
                "- Processing: usually 1 business day\n"
                "- Standard (domestic): about 3–5 business days\n"
                "- Express (domestic): about 1–2 business days\n"
                "- International: typically 7–14 business days"
            )

        # Return flow
        if stage in (None, "start"):
            flow["stage"] = "returns_collect_item"
            return "Of course! I can help with a return. What item would you like to return?"

        if stage == "returns_collect_item":
            if user_text.strip():
                flow["slots"]["return_item"] = user_text.strip()
            flow["stage"] = "returns_collect_date"
            return "Got it. When did you receive the item? (Please provide a date like 2025-09-10)"

        if stage == "returns_collect_date":
            m = re.search(r"\b(20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})\b", user_text)
            flow["slots"]["received_date"] = m.group(1) if m else "unknown"
            flow["stage"] = "returns_condition_check"
            return (
                "Thanks. For returns, items must be unworn and in original condition. "
                "Can you confirm the item is unworn and in its original condition? (yes/no)"
            )

        if stage == "returns_condition_check":
            if YES_PAT.search(user_text):
                flow["stage"] = "returns_reason"
                return "Understood. Could you tell me the reason for the return? (e.g., too small, defective, changed mind)"
            if NO_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return (
                    "Unfortunately, we can only accept returns that are unworn and in original condition. "
                    "If you’d like, I can share our exchange or repair options."
                )
            return "Please reply yes or no: is the item unworn and in original condition?"

        if stage == "returns_reason":
            flow["slots"]["return_reason"] = user_text.strip()
            flow["stage"] = "returns_instructions"
            item = flow["slots"].get("return_item", "the item")
            return (
                f"Thanks. To start your return for **{item}**, please follow these steps:\n"
                "1) I’ll send a prepaid return label via email.\n"
                "2) Pack the item securely with all tags/accessories.\n"
                "3) Drop it off within **14 days** of delivery.\n"
                "Once received and inspected, your refund will be processed to the original payment method.\n"
                "Would you like me to email the return label to you now? (yes/no)"
            )

        if stage == "returns_instructions":
            if YES_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return "Great—return label request submitted! You’ll receive it shortly. Anything else I can help with?"
            if NO_PAT.search(user_text):
                flow["stage"] = "end_or_more"
                return "No problem. If you need the label later, just ask. Anything else I can help with?"
            return "Would you like me to email the return label now? (yes/no)"

        if stage == "end_or_more":
            return "Happy to help. Anything else I can assist you with?"
        return None

    # ---- Discounts & promotions ----
    if current_scenario == "Discounts & promotions":
        flow["stage"] = "end_or_more"
        return (
            "Current promotions: sitewide **10%** (code **WELCOME10**), plus new-member **+5%** on the first order. "
            "Order of operations: subtotal → % coupons → points → tax & shipping. "
            "Some items may be excluded (FINAL_SALE, GIFT_CARD, MTO_EXCLUDED)."
        )

    # ---- Size & fit guidance ----
    if current_scenario == "Size & fit guidance":
        if _is_size_chart_query(user_text):
            if stage in (None, "start"):
                flow["stage"] = "fit_collect"
            return "You can view our size guide on each product page under **Size & Fit**."

        if stage in (None, "start"):
            flow["stage"] = "fit_collect"
            return "Sure—tell me your current size and how it fits."

        if stage == "fit_collect":
            too_small = re.search(r"\b(too\s*small|tight|snug)\b", user_text, re.I)
            too_big = re.search(r"\b(too\s*big|loose)\b", user_text, re.I)
            num = re.search(r"\b(\d{2}(\.\d)?|\d{1,2}(\.\d)?)\b", user_text)
            letter = re.search(r"\b(XXS|XS|S|M|L|XL|XXL)\b", user_text, re.I)

            if num:
                base = num.group(1)
                try:
                    val = float(base)
                except Exception:
                    val = None

                if val is not None and (too_small or too_big):
                    rec = max(0, val - 1 if too_big else val + 1)
                    flow["stage"] = "end_or_more"
                    return (
                        f"Since **{base}** feels {'small' if too_small else 'big'}, try **{rec:.1f}**. "
                        "Want me to check availability in that size?"
                    )
                if val is not None and not (too_small or too_big):
                    return "Thanks. How does that size fit—**too small, too big, or just right**?"

            if letter:
                L = letter.group(1).upper()
                if too_small or too_big:
                    nxt_up = {"XXS": "XS", "XS": "S", "S": "M", "M": "L", "L": "XL", "XL": "XXL"}
                    nxt_down = {"XXL": "XL", "XL": "L", "L": "M", "M": "S", "S": "XS", "XS": "XXS"}
                    rec = nxt_up.get(L) if too_small else nxt_down.get(L)
                    flow["stage"] = "end_or_more"
                    if rec:
                        return (
                            f"Since **{L}** feels {'small' if too_small else 'big'}, try **{rec}**. "
                            "Want me to check availability in that size?"
                        )
                    return "You may need to adjust one size. Want me to check availability?"
                return f"Got it. How does **{L}** fit—**too small, too big, or just right**?"

            return "To recommend a size, tell me what you usually wear and whether it feels **too small, too big, or just right**."

        if stage == "end_or_more":
            return "Happy to help. Anything else I can assist you with?"
        return None

    # No matching rule — fallback to LLM/RAG
    return None


# =========================
# Global intent detection (cross-scenario)
# =========================

GLOBAL_INTENTS = [
    # New arrivals / collections (cover fall/autumn phrasing too)
    (r"\b(new\s+arrivals?|latest\s+(drop|collection|release)s?|new\s+collection|this\s+(winter|fall|autumn|spring|summer)|"
     r"(winter|fall|autumn)\s+(arrivals?|collection))\b",
     "new_arrivals_intent", "New arrivals & collections", 10, True),

    # Size chart / guide
    (r"\b(size\s*(chart|guide)|sizing\s*(chart|guide)?|size\s*info|size\s*measurement(s)?)\b",
     "size_chart_intent", "Size & fit guidance", 9, True),

    # Free returns (generic, doesn’t hijack shipping)
    (r"\b(free\s+return(s)?(\s+shipping)?|return\s+shipping\s+covered)\b",
     "free_returns_intent", "Shipping & returns", 10, True),

    # Shipping (keep generic; free-returns has equal/higher prio so it wins)
    (r"\b(ship|shipping|deliver(y|ed|ing)?|eta|track(ing)?|when\s+will\s+it\s+(arrive|be\s+delivered)|how\s+long.*(deliver|shipping|arrive))\b",
     "shipping_intent", "Shipping & returns", 9, True),

    # Returns/exchange generic 
    (r"\b(return\s+policy|refund\s+policy|return\s+window|return|refund|send back|exchange)\b",
     "returns_intent", "Shipping & returns", 8, True),

    # Availability
    (r"\b(availability|in stock|stock|have .* size|colors?|sizes?(?!\s*(chart|guide)))\b",
     "availability_intent", "Check product availability", 7, True),

    # Rewards
    (r"\b(reward|point|redeem|earn|membership|tier)\b",
     "rewards_intent", "Rewards & membership", 6, True),

    # Promotions / deals / discounts (new)
    (r"\b(deals?|discounts?|promotions?|sale|promo code|coupon)\b",
     "promotions_intent", "Discounts & promotions", 8, True),

    # Price / range (new)
    (r"\b(price|price\s*range|how much|cost)\b",
     "price_intent", "Check product availability", 7, True),

    # Men’s catalog (new)
    (r"\b(mens|men’s|for\s+a\s+man|male)\b",
     "mens_catalog_intent", "Other", 7, True),
]


def detect_global_intent(user_text: str):
    text = (user_text or "").lower()
    best = None
    for pat, key, target, prio, can_inline in GLOBAL_INTENTS:
        if re.search(pat, text, re.I):
            if (best is None) or (prio > best["priority"]):
                best = {"key": key, "target": target, "priority": prio, "can_inline": can_inline}
    return best


# =========================
# Inline answer functions
# =========================

def inline_answer_shipping(user_text: str) -> str:
    # 간단한 배송 개요만 제공 (추가 질문 없음)
    return (
        "**Shipping overview**\n"
        "- Processing: usually 1 business day\n"
        "- Standard (domestic): about 3–5 business days\n"
        "- Express (domestic): about 1–2 business days\n"
        "- International: typically 7–14 business days"
    )

def inline_answer_return_policy(user_text: str) -> str:
    # free_returns_policy + shipping_returns 요지를 2–3문장으로 요약
    return (
        "Returns are accepted within **14 days** of delivery, and items must be **unworn** and in original condition. "
        "Return **shipping is covered** for defective or wrong-item cases; for fit/changed-mind, coverage can vary by policy or promotion. "
        "Refunds are typically processed **3–5 business days** after the carrier scans the return."
    )

def inline_answer_availability(user_text: str) -> str:
    _update_slots_from_text(user_text)
    slots = st.session_state.flow["slots"]
    p = slots.get("product") or "item"
    c = slots.get("color")
    s = slots.get("size")
    base = f"For the {p}"
    if c:
        base += f" in {c}"
    if s:
        base += f" / {s}"
    return (
        f"{base}, I can check availability in detail if you like. "
        f"Would you like to **switch to Check product availability**?"
    )


def inline_answer_fit(user_text: str) -> str:
    mnum = re.search(r"\b(\d{2}(\.\d)?|\d{1,2}(\.\d)?)\b", user_text)
    if mnum:
        base = float(mnum.group(1))
        rec = base + 1 if base >= 20 else base + 0.5
        return (
            f"If **{base:.1f}** feels small, try **{rec:.1f}**. "
            "Want to **switch to Size & fit guidance** for a precise recommendation?"
        )
    return "I can help with sizing. Want to **switch to Size & fit guidance**?"


def inline_answer_rewards(user_text: str) -> str:
    return (
        "Every 100 pts = $1 off (merchandise subtotal only). "
        "Tiers are rolling 12 months; points expire after 12 months of no activity. "
        "Want to **switch to Rewards & membership** to see earning/redeem tips?"
    )


def inline_answer_size_chart(user_text: str) -> str:
    # 간단한 사이즈 표 제공 (Top / Bottom 공용)
    return (
        "Here’s our general **Size Guide** (inches):\n\n"
        "| Size | Bust | Waist | Hip |\n"
        "|:------:|:------:|:------:|:------:|\n"
        "| XXS | 30–31 | 23–24 | 33–34 |\n"
        "| XS  | 32–33 | 25–26 | 35–36 |\n"
        "| S   | 34–35 | 27–28 | 37–38 |\n"
        "| M   | 36–37 | 29–30 | 39–40 |\n"
        "| L   | 38.5–40 | 31.5–33 | 41.5–43 |\n"
        "| XL  | 41.5–43 | 34.5–36 | 44.5–46 |\n"
        "| XXL | 44.5–46 | 37.5–39 | 47.5–49 |\n\n"
    )

def inline_answer_new_arrivals(user_text: str) -> str:
    # short, prose-friendly handoff to the collection
    return ("Our new collection is live. Tell me a category or color and I’ll tailor picks. "
            "Reply **yes** to switch to **New arrivals & collections**.")

def inline_answer_promotions(user_text: str) -> str:
    # mirrors your promotions_rules Short Answer Template
    return ("You can apply % coupons to the merchandise subtotal (e.g., WELCOME10). "
            "New members get an extra +5% on their first order and it stacks. "
            "Some items may be excluded (FINAL_SALE, GIFT_CARD, MTO_EXCLUDED).")

def inline_answer_price(user_text: str) -> str:
    _update_slots_from_text(user_text)
    return ("Prices vary by fabric and construction. If you tell me a product/color/size, "
            "I’ll pull the current price. Typical ranges: tees $18–$32, shirts $35–$79, "
            "knitwear $55–$129, pants $59–$119, jackets $89–$189, dresses $69–$149.")

def inline_answer_mens_catalog(user_text: str) -> str:
    # stops “we only sell women’s” errors and guides to category
    return ("We offer apparel for men and women. For men, common categories include shirts, pants, "
            "jackets, knitwear, and sleepwear. Which category would you like?")

def inline_answer_free_returns(user_text: str) -> str:
    return (
        "Return shipping is covered for defective or wrong-item cases. "
        "For other reasons, coverage may vary by policy or promotion. "
        "Would you like to switch to **Shipping & returns** for the steps?"
    )


# =========================
# Emoji / tone post-processor
# =========================
# Matches most common Unicode emoji ranges
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]+")

def apply_tone_policies(text: str) -> str:
    """
    Tone policy:
    - formal: remove all emojis
    - informal: keep at most one emoji and move it to the very end
    """
    if not isinstance(text, str) or not text:
        return text

    # formal: no emojis at all
    if TONE == "formal":
        return _EMOJI_RE.sub("", text)

    # informal: allow at most one emoji
    emojis = list(_EMOJI_RE.finditer(text))
    if not emojis:
        return text

    # keep only the first emoji
    first_emoji = emojis[0].group(0)
    # remove all emojis from the text
    cleaned = _EMOJI_RE.sub("", text).rstrip()

    # append one emoji at the very end
    if cleaned and cleaned[-1] in ".!?":
        return f"{cleaned} {first_emoji}"
    return f"{cleaned} {first_emoji}"


# =========================
# Inline handler mapping
# =========================

INLINE_HANDLERS = {
    "availability_intent": inline_answer_availability,
    "fit_intent":          inline_answer_fit,
    "rewards_intent":      inline_answer_rewards,
    "size_chart_intent":   inline_answer_size_chart,
    "shipping_intent":     inline_answer_shipping,
    "new_arrivals_intent": inline_answer_new_arrivals,
    "promotions_intent":   inline_answer_promotions,
    "price_intent":        inline_answer_price,
    "mens_catalog_intent": inline_answer_mens_catalog,
    "returns_intent":      inline_answer_return_policy, 
}



# =========================
# Orchestrator
# =========================
def handle_message(user_text: str) -> str:
    # 1) Handle pending yes/no first
    pending_reply = handle_pending_yes(user_text)
    if pending_reply:
        return maybe_add_one_time_closing(apply_tone_policies(pending_reply))

    # 2) Global intent detection and (if needed) inline suggestion / switch prompt
    detected = detect_global_intent(user_text)
    if detected:
        current = st.session_state.flow.get("scenario")
        target  = detected["target"]

        # ⭐ 동일 시나리오여도 인라인 응답이 가능하면 '바로' 답변 (switch 확인 없이)
        if detected["can_inline"]:
            inline_fun = INLINE_HANDLERS.get(detected["key"])
            if inline_fun:
                # 같은 시나리오면 바로 인라인 답변
                if current == target:
                    reply = inline_fun(user_text)
                    return maybe_add_one_time_closing(apply_tone_policies(reply))

        # 시나리오가 다르면, 인라인 미사용 시 스위치 제안
        if current != target:
            if detected["can_inline"]:
                inline_fun = INLINE_HANDLERS.get(detected["key"])
                if inline_fun:
                    reply = inline_fun(user_text)
                    set_pending("confirm_switch", {"target": target})
                    return maybe_add_one_time_closing(apply_tone_policies(reply))
            msg = f"It sounds like **{target}** might be more helpful. Switch to that topic?"
            set_pending("confirm_switch", {"target": target})
            return maybe_add_one_time_closing(apply_tone_policies(msg))

    # 3) Scenario-specific rule routing
    current_scenario = st.session_state.flow.get("scenario")
    rule_reply = route_by_scenario(current_scenario, user_text)
    if rule_reply is not None:
        infer_pending_from_bot_reply(rule_reply)
        return maybe_add_one_time_closing(apply_tone_policies(rule_reply))

    # 4) LLM/RAG fallback
    bot_reply = llm_fallback(user_text)
    infer_pending_from_bot_reply(bot_reply)
    return maybe_add_one_time_closing(apply_tone_policies(bot_reply))


# =========================
# UI — 전체(인사/채팅 → 시나리오 → 입력/진행/종료/만족도) + 즉시 표시
# =========================

# 화면 상단에 렌더링될 영역(고정 순서용 컨테이너) 먼저 배치
chat_area = st.container()       # 최상단: 채팅(인사 포함)
st.markdown("---")
scenario_area = st.container()   # 중간: 시나리오 드롭다운
st.markdown("---")
control_area = st.container()    # 하단: 입력/진행/종료(또는 만족도)

# -------------------------
# (중간) 시나리오 드롭다운 — 선택 처리
# -------------------------
with scenario_area:
    st.markdown("**How can I help you with?**")
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

    # ✅ 접근성 경고 해결: 빈 레이블("") 대신 의미 있는 레이블을 주고 화면에서는 숨김
    scenario = st.selectbox(
        "Select a scenario",
        SCENARIOS,
        index=0,
        key="scenario_select",
        label_visibility="collapsed",
    )

    other_goal_input = ""
    if scenario == "Other":
        other_goal_input = st.text_input(
            "If 'Other', briefly describe your goal (optional)"
        )

    # 선택 변경 감지 → 플로우 초기화 + Skyler 안내를 즉시 채팅에 추가
    if (
        scenario != "— Select a scenario —"
        and st.session_state.last_user_selected_scenario != scenario
    ):
        st.session_state.scenario_selected_once = True
        st.session_state.last_user_selected_scenario = scenario
        st.session_state.flow = {
            "scenario": scenario,
            "stage": "start",
            "slots": {
                "product": None, "color": None, "size": None,
                "contact_pref": None, "tier_known": None, "selected_collection": None,
                "return_item": None, "received_date": None, "return_reason": None
            }
        }
        st.session_state.chat_history.append(
            (_chatbot_speaker(), f"Sure, I will help you with **{scenario}**. Please ask me a question.")
        )

# -------------------------
# (하단) 입력/진행/종료 또는 만족도 단계
# -------------------------
with control_area:
    scenario_selected = (st.session_state.flow.get("scenario") is not None)

    # 1) 대화 단계
    if not st.session_state.awaiting_feedback and not st.session_state.ended:
        # 시나리오가 선택되어야 입력 허용
        if scenario_selected:
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input("Your message:")
                submitted = st.form_submit_button("Send")

            if submitted and user_input.strip():
                # 즉시 메모리 반영 → 같은 사이클에서 바로 보이도록
                st.session_state.user_turns += 1
                st.session_state.chat_history.append(("User", user_input.strip()))
                bot_reply = handle_message(user_input.strip())
                st.session_state.chat_history.append((_chatbot_speaker(), bot_reply))
                st.session_state.bot_turns += 1
        else:
            st.info("Please choose a topic above to start chatting.")

        # 진행 안내(문구 유지)
        if st.session_state.user_turns < MIN_USER_TURNS:
            remaining = MIN_USER_TURNS - st.session_state.user_turns
            st.info(
                f"You’ve sent {st.session_state.user_turns}/{MIN_USER_TURNS} messages (minimum). "
                f"{remaining} more to go."
            )
        st.progress(min(st.session_state.user_turns / MIN_USER_TURNS, 1.0))

        # End Session (5턴 전 회색 비활성 유지)
        st.markdown("---")
        can_end = (st.session_state.user_turns >= MIN_USER_TURNS)
        help_text = None if can_end else f"Please send at least {MIN_USER_TURNS - st.session_state.user_turns} more message(s) before ending."
        if st.button("End Session", disabled=not can_end, help=help_text):
            st.session_state.awaiting_feedback = True
            st.rerun()

    # 2) 만족도 수집 단계
    else:
        if st.session_state.awaiting_feedback and not st.session_state.ended:
            st.subheader("Before you go…")
            st.write("**Overall, how satisfied are you with this chatbot service today?**")
            st.caption("1 = Very dissatisfied, 7 = Very satisfied")
            rating = st.slider("Your overall satisfaction", min_value=1, max_value=7, value=5, step=1)

            # 🔹 Prolific ID 입력 (선택 사항)
            prolific_id = st.text_input(
                "Please provide your Prolific ID (write N/A if none) — only to check the submission completion.",
                value=""
            )

            if st.button("Submit Rating"):
                ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                # 시나리오 문자열
                scenostr = st.session_state.flow.get("scenario") or "— Select a scenario —"

                # 전사 구성 (Prolific ID 포함)
                transcript_lines = []
                transcript_lines.append("===== Session Transcript =====")
                transcript_lines.append(f"timestamp       : {ts}")
                transcript_lines.append(f"session_id      : {st.session_state.session_id}")
                transcript_lines.append(f"identity_option : {identity_option}")
                transcript_lines.append(f"name_present    : {'present' if show_name else 'absent'}")
                transcript_lines.append(f"picture_present : {'present' if show_picture else 'absent'}")
                transcript_lines.append(f"scenario        : {scenostr}")
                transcript_lines.append(f"user_turns      : {st.session_state.user_turns}")
                transcript_lines.append(f"bot_turns       : {st.session_state.bot_turns}")
                transcript_lines.append(f"prolific_id     : {prolific_id if prolific_id.strip() else 'N/A'}")
                transcript_lines.append("--------------------------------")
                for spk, msg in st.session_state.chat_history:
                    transcript_lines.append(f"{spk}: {msg}")
                transcript_lines.append("--------------------------------")
                transcript_lines.append(f"Satisfaction (1-7): {rating}")
                transcript_text = "\n".join(transcript_lines)

                # 저장
                try:
                    # 1) transcript 저장
                    supabase.table("transcripts").insert({
                        "session_id": st.session_state.session_id,
                        "ts": datetime.datetime.utcnow().isoformat() + "Z",
                        "transcript_text": transcript_text,
                    }).execute()
                
                    # 2) session 정보 upsert (중복 insert 방지)
                    supabase.table("sessions").upsert(
                        {
                            "session_id": st.session_state.session_id,
                            "ts_start": datetime.datetime.utcnow().isoformat() + "Z",
                            "ts_end": datetime.datetime.utcnow().isoformat() + "Z",
                            "identity_option": identity_option,
                            "brand_type": brand_type,
                            "name_present": "present",
                            "picture_present": "present",
                            "scenario": scenostr,
                            "user_turns": st.session_state.user_turns,
                            "bot_turns": st.session_state.bot_turns,
                        },
                        on_conflict="session_id"
                    ).execute()
                
                except Exception as e:
                    st.error(f"Failed to save to Supabase: {e}")
                else:
                    st.session_state.rating_saved = True
                    st.session_state.ended = True
                    st.session_state.awaiting_feedback = False
                    st.success("Thanks! Your feedback has been recorded. The session is now closed.")

# -------------------------
# (상단) 채팅 렌더링 — 인사 → 선택 반영 → 방금 입력/응답 순서로 즉시 표시
# -------------------------
with chat_area:
    for speaker, message in st.session_state.chat_history:
        if speaker == "User":
            st.markdown(
                f"""
                <div style='text-align:right; margin:6px 0;'>
                    <span style='background-color:#DCF8C6; padding:8px 12px; border-radius:12px; display:inline-block;'>
                        <b>You:</b> {message}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='text-align:left; margin:6px 0;'>
                    <span style='background-color:#F1F0F0; padding:8px 12px; border-radius:12px; display:inline-block;'>
                        <b>{speaker}:</b> {message}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )



























