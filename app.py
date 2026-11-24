"""
English with Asha — a kind, patient voice English teacher for a Nepali beginner.

Runs on Streamlit. 100% free:
Teacher: Google Gemini API free tier (free key from https://aistudio.google.com/apikey)
Voice:   speech-to-text = Google's free speech recognition (via SpeechRecognition)
         text-to-speech = gTTS (Google Translate voice, supports slow English + Nepali)

Setup
-----
1. pip install -r requirements.txt
2. Create .streamlit/secrets.toml next to this file (never commit it to GitHub):

       GEMINI_API_KEY = "AIza..."
       APP_PASSWORD   = "choose-a-password-for-mom"
       # optional, if Google renames models:
       # GEMINI_MODEL = "gemini-flash-latest"

3. streamlit run app.py
   (or deploy free on share.streamlit.io and paste the same secrets in the app's Settings → Secrets)
"""

import hmac
import io
import json
import re
from datetime import datetime
from pathlib import Path

import speech_recognition as sr
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS

st.set_page_config(page_title="English with Asha", page_icon="🌸", layout="centered")

# Tried in order; the first one that works on your free key is remembered.
MODELS = [m for m in [st.secrets.get("GEMINI_MODEL"), "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
                      "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"] if m]


def discover_models(client) -> list:
    """Ask Google which Flash models this key can use (in case the names above change again)."""
    try:
        found = []
        for m in client.models.list():
            name = (m.name or "").removeprefix("models/")
            actions = getattr(m, "supported_actions", None) or []
            if "flash" in name and "generateContent" in actions and not any(
                    x in name for x in ("tts", "image", "live", "audio", "transcribe", "embedding")):
                found.append(name)
        return sorted(found, reverse=True)
    except Exception:
        return []
PROGRESS_FILE = Path(__file__).with_name("progress.json")

TOPICS = {
    "Shopping · किनमेल": "buying vegetables and fruit at a shop; asking prices, quantities, paying",
    "Travel · यात्रा": "taking a bus or plane; asking the way, buying a ticket, at the airport",
    "Family · परिवार": "talking about her family, children and grandchildren, home and daily routine",
    "Doctor · डाक्टर": "visiting a doctor or pharmacy; saying how she feels, simple body words",
    "Neighbours · छिमेकी": "friendly small talk with a neighbour: weather, greetings, food, the day",
    "Free talk · जे पनि कुरा": "an easy chat about anything she likes; follow her lead",
}

st.markdown(
    """
<style>
html, body, [class*="css"] { font-size: 20px; }
.bubble { padding: 14px 18px; border-radius: 18px; margin: 10px 0; font-size: 22px; line-height: 1.45; }
.teacher { background: #eef5f0; border: 1px solid #cfe0d5; color: #14231c; }
.me { background: #fde8ec; margin-left: 15%; color: #14231c; }
.ne { color: #4d5f55; font-size: 20px; }
.fix { border-top: 1px dashed #b9cbbf; margin-top: 8px; padding-top: 8px; font-size: 18px; color: #4d5f55; }
.fix b { color: #b3122e; }
div.stButton > button { font-size: 20px; min-height: 56px; width: 100%; border-radius: 14px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- login
def login() -> bool:
    if st.session_state.get("authed"):
        return True
    st.title("🌸 English with Asha")
    st.write("नमस्ते! Please enter your password. · कृपया पासवर्ड लेख्नुहोस्।")
    pw = st.text_input("Password · पासवर्ड", type="password")
    if st.button("Log in · लग इन"):
        if hmac.compare_digest(pw, st.secrets["APP_PASSWORD"]):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Wrong password. · पासवर्ड मिलेन।")
    return False


if not login():
    st.stop()

client = genai.Client(api_key=str(st.secrets["GEMINI_API_KEY"]).strip())


# ---------------------------------------------------------------- helpers
def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def rules(topic: str) -> str:
    return f"""You are Asha, a kind and patient English teacher. Your student is a Nepali mother who is a beginner in English. Speak like a warm friend.
Today's topic: {topic}. Role-play this everyday situation with her: {TOPICS[topic]}. You play the other person (shopkeeper, ticket seller, doctor, neighbour, etc.) when it fits.
Rules:
- Very short, simple English sentences: at most 8 words each, at most 3 sentences per turn. Only easy everyday words.
- Ask ONE question per turn, and end your turn with it.
- If she makes mistakes, gently correct only ONE (the most important), e.g. "Very good! We say: 'I want two kilos.'" Then carry on. Praise her efforts often.
- Her words come from speech recognition, so ignore spelling, punctuation and capital letters. Only correct real word or grammar mistakes.
- If she says she doesn't understand, speaks Nepali, or says "मैले बुझिनँ": first explain simply in Nepali (Devanagari script), then repeat the English slowly and simply.
- If she answers in Nepali, praise her, give the English words, and ask her to try saying it in English.
Reply ONLY with JSON, no other text:
{{"say":[{{"lang":"en","text":"..."}},{{"lang":"ne","text":"..."}}],"mistake":null}}
"say" is your turn in order; use "ne" items only for Nepali. When you corrected her, set "mistake" to {{"said":"her words","better":"correct English","why_ne":"very short Nepali reason"}}."""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON")
    return json.loads(m.group(0))


def ask_ai(system: str, messages: list) -> str:
    """Send the conversation to Gemini and return its text. Tries each model until one works."""
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages]
    config = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                         temperature=0.7)
    if st.session_state.get("model_ok"):
        models = [st.session_state["model_ok"]]
    else:
        models = MODELS + [m for m in discover_models(client) if m not in MODELS]
    errors = []
    for model in models:
        try:
            r = client.models.generate_content(model=model, contents=contents, config=config)
            st.session_state["model_ok"] = model
            st.session_state["last_error"] = None
            return r.text or ""
        except Exception as e:  # model missing / not free / busy → try the next one
            errors.append(f"{model}: {str(e)[:300]}")
    st.session_state.pop("model_ok", None)
    raise RuntimeError("\n\n".join(errors))


def tts(say: list, slow: bool) -> bytes:
    """Turn the teacher's lines into one MP3 (English slow, Nepali normal speed)."""
    out = io.BytesIO()
    for part in say:
        text = part.get("text", "").strip()
        if not text:
            continue
        lang = "ne" if part.get("lang") == "ne" else "en"
        try:
            gTTS(text=text, lang=lang, slow=(slow and lang == "en"), tld="com").write_to_fp(out)
        except Exception:
            pass  # if Nepali voice fails, the text is still on screen
    return out.getvalue()


def transcribe(wav_bytes: bytes, lang: str) -> str:
    r = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
        audio = r.record(src)
    try:
        return r.recognize_google(audio, language=lang)
    except sr.UnknownValueError:
        return ""


def load_progress() -> list:
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_progress(entry: dict) -> None:
    data = load_progress()
    data.insert(0, entry)
    PROGRESS_FILE.write_text(json.dumps(data[:100], ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("topic", None)
ss.setdefault("turns", [])        # [{"r":"t","say":[...],"mistake":...} | {"r":"m","text":"..."}]
ss.setdefault("audio", None)      # last teacher audio (bytes)
ss.setdefault("autoplay", False)
ss.setdefault("mic_key", 0)
ss.setdefault("slow", True)
ss.setdefault("feedback", None)


def api_messages() -> list:
    msgs = [{"role": "user", "content": "Please begin the lesson: greet her warmly and ask your first easy question."}]
    for t in ss.turns:
        if t["r"] == "t":
            msgs.append({"role": "assistant", "content": json.dumps({"say": t["say"], "mistake": t.get("mistake")}, ensure_ascii=False)})
        else:
            msgs.append({"role": "user", "content": t["text"]})
    return msgs


def teacher_turn() -> None:
    with st.spinner("Asha is thinking… · आशाले सोच्दैछिन्…"):
        try:
            raw = ask_ai(rules(ss.topic), api_messages())
            try:
                m = parse_json(raw)
            except Exception:
                m = {"say": [{"lang": "en", "text": raw.strip()}], "mistake": None}
            say = [{"lang": "ne" if x.get("lang") == "ne" else "en", "text": str(x.get("text", ""))}
                   for x in m.get("say", []) if x.get("text")]
            mistake = m.get("mistake") if isinstance(m.get("mistake"), dict) and m["mistake"].get("better") else None
        except Exception as e:
            say = [{"lang": "en", "text": "Sorry, something went wrong. Please try again in a minute."},
                   {"lang": "ne", "text": "माफ गर्नुहोस्, केही गडबड भयो। एक मिनेटपछि फेरि प्रयास गर्नुहोस्।"}]
            mistake = None
            ss.last_error = str(e)
    ss.turns.append({"r": "t", "say": say, "mistake": mistake})
    ss.audio = tts(say, ss.slow)
    ss.autoplay = True


def student_says(text: str) -> None:
    text = (text or "").strip()
    if text:
        ss.turns.append({"r": "m", "text": text})
        teacher_turn()


def make_feedback() -> dict:
    conv = "\n".join(
        ("Teacher: " + " ".join(x["text"] for x in t["say"])) if t["r"] == "t" else ("Student: " + t["text"])
        for t in ss.turns
    )
    prompt = f"""Here is today's practice conversation (topic: {ss.topic}) between Asha, a kind English teacher, and a Nepali mother who is a beginner:
{conv}

Write gentle, encouraging feedback for her. Reply ONLY with JSON:
{{"well":[{{"en":"short simple English","ne":"same in Nepali"}}],
"improve":[{{"said":"what she said","better":"better English","tip_ne":"short Nepali explanation"}}],
"words":[{{"en":"word or phrase","ne":"Nepali meaning"}}],
"next_en":"one short sentence: what to practise next time","next_ne":"same in Nepali"}}
"well": 2-3 items. "improve": at most 3, the most useful only, from her real sentences (empty list if none). "words": 3-6 useful words from today."""
    raw = ask_ai("You are Asha, a kind English teacher.", [{"role": "user", "content": prompt}])
    return parse_json(raw)


def show_feedback(fb: dict, topic: str, when: str) -> None:
    st.caption(f"{topic} · {when}")
    st.subheader("✅ You did well · तपाईंले राम्रो गर्नुभयो")
    for i in fb.get("well", []):
        st.markdown(f"- {esc(i.get('en',''))}  \n  <span class='ne'>{esc(i.get('ne',''))}</span>", unsafe_allow_html=True)
    if fb.get("improve"):
        st.subheader("🌱 You can improve · यसरी अझ राम्रो")
        for n, i in enumerate(fb["improve"]):
            st.markdown(f"<div class='bubble teacher'><s>{esc(i.get('said',''))}</s><br><b>✓ {esc(i.get('better',''))}</b>"
                        f"<div class='ne'>{esc(i.get('tip_ne',''))}</div></div>", unsafe_allow_html=True)
            st.audio(tts([{"lang": "en", "text": i.get("better", "")}], True), format="audio/mp3")
    if fb.get("words"):
        st.subheader("📖 New words · नयाँ शब्दहरू")
        for i in fb["words"]:
            st.markdown(f"**{esc(i.get('en',''))}** — {esc(i.get('ne',''))}")
    if fb.get("next_en"):
        st.subheader("➡️ Next time · अर्को पटक")
        st.write(fb["next_en"])
        st.markdown(f"<span class='ne'>{esc(fb.get('next_ne',''))}</span>", unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar: progress
with st.sidebar:
    st.header("My progress · मेरो प्रगति")
    hist = load_progress()
    week = [h for h in hist if (datetime.now() - datetime.fromisoformat(h["date"])).days < 7]
    c1, c2 = st.columns(2)
    c1.metric("This week · यो हप्ता", len(week))
    c2.metric("Days · दिन", len({h["date"][:10] for h in hist}))
    for n, h in enumerate(hist[:20]):
        with st.expander(f"{h['date'][:10]} · {h['topic'].split(' · ')[0]}"):
            fb = h.get("feedback", {})
            for i in fb.get("improve", []):
                st.write(f"✓ {i.get('better','')}")
            for i in fb.get("words", []):
                st.write(f"📖 {i.get('en','')} — {i.get('ne','')}")
    st.divider()
    ss.slow = st.toggle("Speak slowly · बिस्तारै बोल्ने", value=ss.slow)
    if st.button("Log out · लग आउट"):
        ss.clear()
        st.rerun()


# ---------------------------------------------------------------- pages
# Feedback page
if ss.feedback:
    st.title("How did I do? · म कस्तो गरें?")
    show_feedback(ss.feedback, ss.topic, datetime.now().strftime("%d %b %Y"))
    if st.button("🏠 Home · घर"):
        ss.topic, ss.turns, ss.feedback, ss.audio = None, [], None, None
        st.rerun()
    st.stop()

# Home page: choose topic
if not ss.topic:
    st.title("🌸 Namaste!")
    st.write("आज थोरै अंग्रेजी अभ्यास गरौँ। · Let's practise a little English today.")
    st.subheader("What shall we practise? · के अभ्यास गर्ने?")
    cols = st.columns(2)
    for i, name in enumerate(TOPICS):
        if cols[i % 2].button(name, key=f"topic{i}"):
            ss.topic, ss.turns = name, []
            teacher_turn()
            st.rerun()
    st.stop()

# Lesson page
st.title(ss.topic)

for t in ss.turns:
    if t["r"] == "t":
        html = "".join(f"<div class='ne'>{esc(x['text'])}</div>" if x["lang"] == "ne" else f"<div>{esc(x['text'])}</div>"
                       for x in t["say"])
        if t.get("mistake"):
            m = t["mistake"]
            html += f"<div class='fix'>Better: <b>{esc(m['better'])}</b> — {esc(m.get('why_ne',''))}</div>"
        st.markdown(f"<div class='bubble teacher'>👩‍🏫 {html}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bubble me'>{esc(t['text'])}</div>", unsafe_allow_html=True)

if ss.get("last_error"):
    with st.expander("⚠️ Technical details", expanded=True):
        st.code(ss.last_error, language=None)

if ss.audio:
    st.audio(ss.audio, format="audio/mp3", autoplay=ss.autoplay)
    ss.autoplay = False

st.divider()
speak_lang = st.radio("I will speak in · म बोल्छु:", ["English", "नेपाली"], horizontal=True)
rec = st.audio_input("🎤 Tap the mic and speak · माइक थिचेर बोल्नुहोस्", key=f"mic{ss.mic_key}")
if rec is not None:
    with st.spinner("Listening… · सुन्दैछु…"):
        text = transcribe(rec.getvalue(), "en-IN" if speak_lang == "English" else "ne-NP")
    ss.mic_key += 1
    if text:
        student_says(text)
    else:
        st.warning("Sorry, I could not hear. Please try again. · सुन्न सकिनँ, फेरि भन्नुहोस्।")
    st.rerun()

with st.form("type", clear_on_submit=True):
    typed = st.text_input("Or type here · वा यहाँ लेख्नुहोस्")
    if st.form_submit_button("Send · पठाउनुहोस्"):
        student_says(typed)
        st.rerun()

c1, c2, c3 = st.columns(3)
if c1.button("🔁 Say again\nफेरि भन्नुहोस्"):
    ss.autoplay = True
    st.rerun()
if c2.button("❓ I don't understand\nमैले बुझिनँ"):
    student_says("मैले बुझिनँ। I don't understand.")
    st.rerun()
if c3.button("✅ Finish\nसकियो"):
    if sum(1 for t in ss.turns if t["r"] == "m") < 2:
        st.info("Talk a little more first! · अलि बढी कुरा गरौँ!")
    else:
        with st.spinner("Asha is writing your feedback… · प्रतिक्रिया लेख्दैछिन्…"):
            try:
                ss.feedback = make_feedback()
                save_progress({"date": datetime.now().isoformat(timespec="seconds"), "topic": ss.topic,
                               "feedback": ss.feedback})
            except Exception as e:
                st.error(f"Could not make feedback, please try again. ({e})")
        st.rerun()
