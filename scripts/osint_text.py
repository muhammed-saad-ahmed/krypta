"""Semantic text analysis for Karachi civic-hazard reports (English, Roman Urdu, Urdu script).

Replaces the earlier exact-match lexicon. A post is now compared with hazard CONCEPTS in a vector space:

    text -> clauses -> candidate spans (1-4 word windows)  --embed-->  unit vectors
    hazard / urgency / negativity / resolved / praise concepts = banks of SEED phrases (also embedded)
    similarity of a span to a concept  =  cosine to its nearest seed, mapped to 0..1 and scaled by the seed's prior

so the spelling chaos of Roman Urdu ('khada', 'khadda', 'khara', 'gaddha', 'gadha', 'gaddhay') no longer needs one
dictionary entry per spelling. A structural parser then reads negation, intensity and sarcasm from the sentence
shape, because similarity alone cannot tell "pani jama hai" from "pani jama nahi hai".

Two interchangeable embedders (same interface, chosen at runtime):

    OnnxSentenceEmbedder   a multilingual MiniLM / MobileBERT sentence encoder exported to ONNX and run with
                           ONNX Runtime on the CPU. TRUE cross-lingual semantics ("pothole" ~ "gaddha" ~ "khadda").
                           The model is NOT bundled (licence, size, and it cannot be downloaded in every environment);
                           see docs/multimodal_fusion.md for the one-line export. Needs `tokenizers` + `onnxruntime`.
    HashedNgramEmbedder    dependency-free fallback: spelling-normalised character n-grams hashed into a fixed
                           vector. It is ORTHOGRAPHIC, not semantic: robust to spelling variants and to seeds
                           written in each language, but it cannot know that two unrelated words mean the same.
                           It is what runs when no ONNX model is installed, and what the test-suite exercises.
Every result says which backend produced it (`backend`).

Outputs of analyze_text() (same keys as before, plus four):
    hazard_type, hazard_strength, urgency, negativity, community_signal, status, keywords, lang
    negated     the hazard was mentioned but denied ("koi gaddha nahi hai")
    sarcasm     praise phrased over a hazard ("wah kya sadak hai, gaddhe hi gaddhe")
    backend     which embedder produced the similarities
    needs_review  the structure was ambiguous (e.g. praise + denied hazard + irony marker)

Known weaknesses (real): thresholds that map cosine similarity to a score are expert-set per backend and not
validated on labelled Karachi text; sarcasm rules cover common structures, not all irony; long-range negation
(> 3 words away) is missed; quoted old incidents look current. The fusion layer limits the damage by counting each
author once and capping how much text can ever contribute.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import unicodedata
from collections import deque

import numpy as np

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- English <-> Urdu-script alignment dictionary
# SUPPORTED LANGUAGES: standard English and Urdu in Urdu script (Unicode Arabic block). Roman Urdu ("pani jama hai") is
# NOT supported: such a post is recognised by detect_language() and reported as unsupported, never scored.
# One row per civic concept; the English list and the Urdu list name the SAME thing, so the two languages stay aligned
# and a reviewer can extend either side. (phrase, prior): prior = how strongly a clean match counts.
ALIGNMENT = {
    "pothole": {
        "en": [("pothole", .80), ("potholes", .80), ("road crater", .70), ("sinkhole", .60), ("caved in road", .60),
               ("road cave in", .55), ("hole in the road", .70)],
        "ur": [("گڑھا", .65), ("گڑھے", .65), ("گڑھوں", .60), ("سڑک میں گڑھا", .80), ("سڑک پر گڑھے", .80),
               ("سڑک میں گڑھے", .80), ("بڑا گڑھا", .75)]},
    "road_damage": {
        "en": [("cracked asphalt", .75), ("broken asphalt", .75), ("crumbling asphalt", .75), ("asphalt cracks", .65),
               ("broken road", .75), ("damaged road", .75), ("road damaged", .75), ("cracked road", .70),
               ("road collapsed", .85), ("bad road", .55)],
        "ur": [("سڑک خراب", .75), ("سڑک کی حالت خراب", .75), ("ٹوٹی سڑک", .75), ("ٹوٹی ہوئی سڑک", .75),
               ("سڑک ٹوٹ گئی", .75), ("سڑک ٹوٹی ہوئی", .75), ("خستہ حال سڑک", .75), ("سڑک بیٹھ گئی", .80)]},
    "open_manhole": {
        "en": [("open manhole", .90), ("missing manhole", .90), ("uncovered manhole", .90), ("manhole cover missing", .90),
               ("manhole", .65), ("manhole cover", .70), ("gutter cover missing", .85), ("open gutter", .80)],
        "ur": [("کھلا مین ہول", .90), ("کھلے مین ہول", .90), ("مین ہول کھلا", .90), ("مین ہول کا ڈھکن غائب", .92),
               ("ڈھکن کے بغیر مین ہول", .90), ("ڈھکن غائب", .80), ("ڈھکن نہیں", .70), ("کھلا گٹر", .85),
               ("مین ہول", .65), ("ڈھکن", .55)]},
    "sewage_overflow": {
        "en": [("sewage overflow", .92), ("overflowing sewer", .90), ("sewer overflow", .90), ("sewage", .70),
               ("sewerage", .70), ("sewer", .60), ("gutter overflow", .90), ("choked drain", .70),
               ("sewerage line leak", .70), ("sewage water", .80)],
        "ur": [("گٹر کا پانی", .88), ("گٹر ابل رہا", .92), ("گٹر ابل پڑا", .92), ("ابلتا گٹر", .90), ("سیوریج", .70),
               ("سیوریج کا پانی", .90), ("سیوریج اوور فلو", .92), ("گندا پانی", .75), ("گٹر", .60)]},
    "waterlogging": {
        "en": [("flooded street", .85), ("flooded road", .85), ("waterlogging", .90), ("water logging", .90),
               ("waterlogged", .85), ("flooded", .70), ("flooding", .65), ("flood", .50), ("standing water", .75),
               ("stagnant water", .70), ("rainwater", .50), ("rain water", .50), ("inundated", .80), ("knee deep", .70),
               ("water accumulation", .65)],
        "ur": [("گلی میں پانی", .80), ("سڑک پر پانی", .80), ("پانی جمع", .85), ("جمع شدہ پانی", .80),
               ("بارش کا پانی", .60), ("سیلاب", .55), ("پانی بھر گیا", .80), ("زیر آب", .80), ("کھڑا پانی", .75),
               ("سڑکیں زیر آب", .85)]},
    "open_drain": {
        "en": [("open drain", .85), ("open nala", .85), ("nullah", .55), ("nala overflow", .85), ("drain overflow", .75),
               ("open sewer line", .75)],
        "ur": [("کھلا نالہ", .85), ("کھلی نالی", .80), ("نالہ ابل", .85), ("نالہ", .50), ("نالے", .50)]},
    "garbage": {
        "en": [("garbage", .60), ("trash", .50), ("rubbish", .50), ("dumped waste", .60), ("garbage pile", .75)],
        "ur": [("کچرا", .60), ("کوڑا", .50), ("کچرے کا ڈھیر", .75), ("کوڑے کا ڈھیر", .75), ("گندگی کا ڈھیر", .70)]},
}
# cross-language equivalences, for documentation and tests: (concept, English term, Urdu term)
ALIGNED_PAIRS = (
    ("pothole", "pothole", "گڑھا"),
    ("road_damage", "cracked asphalt", "سڑک خراب"),
    ("open_manhole", "open manhole", "کھلا مین ہول"),
    ("sewage_overflow", "sewage overflow", "گٹر کا پانی"),
    ("sewage_overflow", "sewage", "سیوریج"),
    ("waterlogging", "flooded street", "گلی میں پانی"),
)


def _flatten(concept):
    return [item for lang in ("en", "ur") for item in ALIGNMENT[concept][lang]]


HAZARD_SEEDS = {concept: _flatten(concept) for concept in ALIGNMENT}

URGENCY_SEEDS = [("child", .60), ("children", .60), ("kid", .50), ("baby", .55), ("accident", .55), ("injured", .70),
                 ("injury", .60), ("died", .90), ("death", .85), ("dead", .85), ("drowned", .95), ("fell into", .75),
                 ("fell", .45), ("electrocuted", .85), ("life threatening", .85), ("emergency", .60),
                 ("urgent", .50), ("dangerous", .45),
                 ("بچہ", .60), ("بچے", .60), ("بچوں", .60), ("حادثہ", .60), ("زخمی", .70), ("ہلاک", .90), ("موت", .85),
                 ("مر گیا", .90), ("ڈوب گیا", .95), ("گر گیا", .60), ("خطرناک", .45), ("فوری", .45), ("ایمرجنسی", .60)]
NEGATIVITY_SEEDS = [("terrible", .50), ("horrible", .55), ("worst", .60), ("disgusting", .55), ("fed up", .60),
                    ("nobody cares", .70), ("ignored", .55), ("helpless", .55), ("complaint", .40), ("complained", .45),
                    ("weeks", .35), ("months", .45), ("unbearable", .60),
                    ("پریشان", .50), ("مسئلہ", .35), ("مسائل", .40), ("کب تک", .50), ("کوئی نہیں سنتا", .70),
                    ("تنگ آ گئے", .60), ("بدترین", .60), ("ناقابل برداشت", .60), ("مہینوں", .45), ("ہفتوں", .35)]
RESOLVED_SEEDS = [("fixed", .9), ("repaired", .9), ("cleared", .9), ("resolved", .9), ("cleaned", .9),
                  ("covered now", .9), ("has been covered", .9),
                  ("ٹھیک ہو گیا", .9), ("صاف ہو گیا", .9), ("حل ہو گیا", .9), ("مرمت ہو گئی", .9), ("کام مکمل", .9),
                  ("پانی نکل گیا", .9), ("صاف کر دیا گیا", .9), ("ڈھکن لگا دیا", .9)]
# surface praise (any strength): over a hazard it is the signature of structural sarcasm
PRAISE_SEEDS = [("great", .8), ("awesome", .8), ("perfect", .8), ("brilliant", .8), ("amazing", .8), ("good job", .8),
                ("great job", .9), ("wonderful", .8), ("lovely", .8), ("nice", .6), ("love it", .8), ("thank you", .6),
                ("thanks", .6), ("enjoy", .6), ("fantastic", .8),
                ("بہت خوب", .8), ("واہ", .8), ("شکریہ", .6), ("شاباش", .8), ("زبردست", .8), ("بہترین", .8), ("کمال", .7)]

# HIGH-INTENSITY praise: the rule-based sarcasm detector (see rule_based_sarcasm) flags a report when one of these
# appears in the same text block as a classified civic hazard term. English regexes run on the lower-cased, punctuation-
# stripped text; Urdu ones on text normalised by normalize_urdu().
HIGH_INTENSITY_PRAISE = {
    "en": [r"excellent(?: work| job| service| performance)?", r"great (?:work|job)", r"good (?:work|job)", r"well done",
           r"(?:amazing|wonderful|fantastic|brilliant|perfect|superb|outstanding) (?:work|job|service)?",
           r"fantastic", r"brilliant", r"amazing", r"outstanding", r"superb", r"bravo", r"kudos", r"hats off",
           r"keep it up", r"much appreciated", r"thank(?:s| you) (?:so much|very much|a lot|a ton)",
           r"love (?:it|this)", r"what a (?:great|wonderful|fantastic|beautiful) (?:road|city|job)"],
    "ur": [r"بہت شکریہ", r"شکریہ", r"شاباش", r"بہت خوب", r"واہ واہ", r"واہ", r"زبردست", r"بہترین(?: کام)?", r"لاجواب",
           r"کمال کر دیا", r"کمال ہے", r"عمدہ کام", r"بہت اچھا کام", r"بہت اچھے", r"مبارک ہو", r"خوب کام", r"ماشاءاللہ"],
}

# closed-class function words (structure, not vocabulary of hazards). English and Urdu script only.
PRE_NEGATORS = {"no", "not", "never", "without", "neither", "nor", "nothing", "isn't", "isnt", "wasn't", "wasnt",
                "aren't", "arent", "hasn't", "hasnt", "doesn't", "doesnt", "didn't", "didnt", "cannot", "can't", "cant",
                "بغیر"}
POST_NEGATORS = {"نہیں", "نہ", "مت", "نہی"}
RESOLVE_BLOCKERS = {"yet", "still", "not", "never", "نہیں", "ابھی", "تک"}
BOOSTERS = {"very", "extremely", "huge", "massive", "completely", "totally", "heavily", "badly", "severe", "terribly",
            "بہت", "انتہائی", "شدید", "بے", "حد", "بڑا", "بڑی", "زیادہ"}
DIMINISHERS = {"little", "slight", "slightly", "minor", "small", "bit", "تھوڑا", "تھوڑی", "ہلکا", "ہلکی", "معمولی",
               "ذرا", "چھوٹا", "چھوٹی"}
IRONY_MARKERS = {"sure", "right", "obviously", "ofcourse", "yeah", "haha", "lol", "ہاں", "بالکل", "ہاہا", "واقعی"}
CONTRAST = r"(?:but|however|though|although|yet|لیکن|مگر|البتہ|تاہم|حالانکہ)"

# Detection ONLY: used to recognise Roman Urdu and refuse it. Nothing below is ever used to score a report.
ROMAN_URDU_MARKERS = {"hai", "hain", "nahi", "nahin", "nhi", "ka", "ki", "ke", "ko", "mein", "bohat", "bahut", "gaya",
                      "gayi", "raha", "rahi", "pani", "paani", "aur", "yeh", "woh", "kya", "abhi", "koi", "bhi", "jama",
                      "khula", "sadak", "gaddha", "kachra"}
ROMAN_URDU_MIN_HITS = 2
ROMAN_URDU_MIN_SHARE = 0.20

# similarity (cosine) -> 0..1 score. EXPERT-SET per backend; see module docstring.
SIM_BAND = {"hashed-char-ngram": (0.72, 0.95), "onnx-sentence-encoder": (0.40, 0.78)}
MIN_HAZARD_STRENGTH = 0.30
MAX_SPAN_WORDS = 4
MAX_SPANS = {"hashed-char-ngram": 400, "onnx-sentence-encoder": 28}
SARCASM_URGENCY_DAMPING = 0.5
SARCASM_NEGATIVITY_FLOOR = 0.60
NEGATED_RESIDUAL = 0.10

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?92|0)?[-\s]?(?:3\d{2}|21)[-\s]?\d{7}(?!\d)")
_MENTION_RE = re.compile(r"(?<![\w])@\w{2,}")
_URDU_RE = re.compile("[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")       # Arabic script block (Urdu)
_LATIN_RE = re.compile(r"[A-Za-z\u00C0-\u024F]")

# ---------------------------------------------------------------- Urdu-script normalisation
_URDU_MARKS = re.compile("[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")        # harakat, dagger alef, Quranic marks, tatweel
_URDU_MAP = str.maketrans({
    "\u064A": "\u06CC", "\u0649": "\u06CC", "\u06D2": "\u06D2",                 # Arabic yeh / alef maksura -> Urdu yeh
    "\u0643": "\u06A9",                                                          # Arabic kaf -> Urdu kaf
    "\u06C3": "\u06C1", "\u06D5": "\u06C1", "\u0629": "\u06C1", "\u0647": "\u06C1",              # teh marbuta / ae -> heh goal
    "\u0623": "\u0627", "\u0625": "\u0627", "\u0671": "\u0627",              # hamza-alef variants -> alef
    "\u200c": " ", "\u200d": "", "\u200e": "", "\u200f": "",                  # ZWNJ splits words ("مین\u200cہول")
    "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3", "\u0664": "4",
    "\u0665": "5", "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
    "\u06F0": "0", "\u06F1": "1", "\u06F2": "2", "\u06F3": "3", "\u06F4": "4",
    "\u06F5": "5", "\u06F6": "6", "\u06F7": "7", "\u06F8": "8", "\u06F9": "9",
})
_URDU_PUNCT = "\u060C\u061B\u061F\u06D4\u066A\u066B\u066C\u066D"                # ، ؛ ؟ ۔ and Arabic percent/decimal marks


def normalize_urdu(text: str) -> str:
    """Canonical form of Urdu-script text so that visually identical spellings compare equal: NFKC, diacritics and
    tatweel removed, Arabic yeh/kaf/heh/alef variants folded to the Urdu letters, zero-width joiners removed (ZWNJ
    becomes a space), Arabic-Indic and Persian digits turned into ASCII. Latin text passes through unchanged."""
    if not text:
        return ""
    return _URDU_MARKS.sub("", unicodedata.normalize("NFKC", text)).translate(_URDU_MAP)


# ---------------------------------------------------------------- normalisation
def _norm(text: str) -> str:
    t = normalize_urdu(_URL_RE.sub(" ", (text or "").lower()))
    t = re.sub(r"[" + _URDU_PUNCT + r"]", " ", t)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", t)).strip()


_VOWEL_DIGRAPHS = (("ee", "i"), ("ii", "i"), ("oo", "u"), ("ph", "f"))


def roman_normalize(token: str) -> str:
    """Spelling/inflection folding for ENGLISH tokens ('potholes' / 'pothole', 'flooded' / 'flooding' / 'flood',
    'colour' typos with doubled letters). Applied identically to seeds and to incoming text. Urdu-script tokens (non-ASCII)
    are returned unchanged. (The name is historical; it no longer does anything for Roman Urdu, which is unsupported.)"""
    if not token or not token.isascii():
        return token
    t = token
    for a, b in _VOWEL_DIGRAPHS:
        t = t.replace(a, b)
    t = re.sub(r"([a-z])\1+", r"\1", t)
    if len(t) > 4:
        for suffix, keep in (("ing", 3), ("ed", 3), ("s", 4)):
            if t.endswith(suffix) and len(t) - len(suffix) >= keep:
                t = t[:-len(suffix)]
                break
    return t


normalize_token = roman_normalize


# ---------------------------------------------------------------- embedders
class Embedder:
    """encode(list[str]) -> float32 array (n, dim) of L2-normalised row vectors."""
    name = "base"
    dim = 0

    def encode(self, texts):
        raise NotImplementedError


class HashedNgramEmbedder(Embedder):
    """Dependency-free orthographic embedding: every word is spelling-normalised, split into character 2/3/4-grams
    (with word-boundary markers), and signed-hashed into `dim` buckets. Cosine similarity is then
    large for spelling variants and for phrases that share word parts. NOT semantic (see module docstring)."""
    name = "hashed-char-ngram"

    def __init__(self, dim=1024):
        self.dim = int(dim)
        self._cache = {}

    def _word_vec(self, word):
        v = self._cache.get(word)
        if v is None:
            v = np.zeros(self.dim, dtype=np.float32)
            padded = f"^{word}$"
            grams = [padded[i:i + n] for n in (2, 3, 4) for i in range(max(0, len(padded) - n + 1))]
            for g in grams:
                h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
                v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            if len(self._cache) < 50_000:
                self._cache[word] = v
        return v

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _norm(t).split():
                out[i] += self._word_vec(roman_normalize(w))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def mean_pool(last_hidden, attention_mask):
    """Mean of token vectors weighted by the attention mask. (B, L, D), (B, L) -> (B, D)."""
    m = np.asarray(attention_mask, dtype=np.float32)[..., None]
    summed = (np.asarray(last_hidden, dtype=np.float32) * m).sum(axis=1)
    return summed / np.maximum(m.sum(axis=1), 1e-9)


class OnnxSentenceEmbedder(Embedder):
    """Multilingual sentence encoder (e.g. paraphrase-multilingual-MiniLM-L12-v2, or a MobileBERT variant) running on
    ONNX Runtime / CPU. Needs a folder with `model.onnx` and `tokenizer.json`.

    Export once, on any machine with internet (free Colab is fine), then copy the folder to models/text/:
        pip install "optimum[exporters]" ; optimum-cli export onnx --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 models/text/
    The model may emit a ready `sentence_embedding` or per-token `last_hidden_state`; both are handled (mean pooling).
    `tokenizer` can be injected (any object whose encode_batch(list[str]) returns items with .ids and
    .attention_mask) - used by the tests; otherwise the `tokenizers` package loads tokenizer.json."""
    name = "onnx-sentence-encoder"

    def __init__(self, model_path, tokenizer_path=None, *, tokenizer=None, max_len=128, threads=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads or max(1, min(4, os.cpu_count() or 1)))
        so.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        outs = [o.name for o in self._sess.get_outputs()]
        self._out = "sentence_embedding" if "sentence_embedding" in outs else outs[0]
        self._pooled = self._out == "sentence_embedding"
        self.max_len = int(max_len)
        if tokenizer is None:
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise ImportError("OnnxSentenceEmbedder needs `pip install tokenizers`") from exc
            tokenizer = Tokenizer.from_file(tokenizer_path or os.path.join(os.path.dirname(model_path), "tokenizer.json"))
            tokenizer.enable_truncation(max_length=self.max_len)
            tokenizer.enable_padding()
        self._tok = tokenizer
        self._lock = threading.Lock()
        self.dim = int(self.encode(["test"]).shape[1])

    def encode(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        enc = self._tok.encode_batch([str(t) for t in texts])
        width = max(len(e.ids) for e in enc)
        ids = np.zeros((len(enc), width), dtype=np.int64)
        mask = np.zeros((len(enc), width), dtype=np.int64)
        for i, e in enumerate(enc):
            n = len(e.ids)
            ids[i, :n], mask[i, :n] = e.ids, e.attention_mask
        feed = {"input_ids": ids}
        if "attention_mask" in self._inputs:
            feed["attention_mask"] = mask
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        with self._lock:
            out = self._sess.run([self._out], feed)[0]
        vec = out if self._pooled else mean_pool(out, mask)
        vec = np.asarray(vec, dtype=np.float32)
        return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)


_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()


def default_model_dir():
    return os.getenv("KP_TEXT_MODEL_DIR", os.path.join("models", "text"))


def get_embedder():
    """The process-wide embedder: the ONNX encoder when models/text/model.onnx (+ tokenizer.json) is installed and
    loads, otherwise the hashed fallback. Never raises."""
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                model = os.path.join(default_model_dir(), "model.onnx")
                emb = None
                if os.path.exists(model):
                    try:
                        emb = OnnxSentenceEmbedder(model)
                    except Exception as exc:                       # missing `tokenizers`, bad model, ...
                        log.warning("ONNX text model found but unusable (%s); using the hashed n-gram fallback", exc)
                _EMBEDDER = emb or HashedNgramEmbedder()
    return _EMBEDDER


def set_embedder(embedder):
    """Install an embedder (None resets to automatic selection). Returns the previous one."""
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        old, _EMBEDDER = _EMBEDDER, embedder
    return old


# ---------------------------------------------------------------- seed bank
class SeedBank:
    """Concept -> seed phrases. Seeds can be added at runtime (`add`) or loaded from JSON, so reviewers can teach the
    system new spellings and slang without touching code. Embedding matrices are cached per embedder."""

    def __init__(self, seeds=None):
        self._seeds = {c: list(v) for c, v in (seeds or {}).items()}
        self._cache = {}
        self._lock = threading.Lock()

    def concepts(self):
        return list(self._seeds)

    def phrases(self, concept):
        return list(self._seeds.get(concept, []))

    def add(self, concept, phrase, prior=0.7):
        phrase = (phrase or "").strip()
        if not phrase or not 0 < prior <= 1:
            raise ValueError("seed needs a non-empty phrase and a prior in (0, 1]")
        with self._lock:
            lst = self._seeds.setdefault(concept, [])
            if all(p.lower() != phrase.lower() for p, _ in lst):
                lst.append((phrase, float(prior)))
                self._cache = {k: v for k, v in self._cache.items() if k[0] != concept}

    def load_json(self, path):
        """JSON: {"pothole": [["khadda", 0.6], ["gadda", 0.5]], "garbage": [...]}. Bad entries are skipped, not fatal."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        n = 0
        for concept, items in data.items():
            for it in items:
                try:
                    self.add(concept, it[0], float(it[1]) if len(it) > 1 else 0.7)
                    n += 1
                except (ValueError, TypeError, IndexError):
                    log.warning("skipped invalid seed %r for %s", it, concept)
        return n

    def matrix(self, concept, embedder):
        key = (concept, embedder.name)
        with self._lock:
            hit = self._cache.get(key)
            if hit is None:
                items = self._seeds.get(concept, [])
                if not items:
                    hit = (np.zeros((0, embedder.dim or 1), np.float32), np.zeros(0), [])
                else:
                    hit = (embedder.encode([_norm(p) for p, _ in items]),
                           np.array([w for _, w in items], dtype=np.float64), [p for p, _ in items])
                self._cache[key] = hit
            return hit


BANK = SeedBank({**HAZARD_SEEDS, "_urgency": URGENCY_SEEDS, "_negativity": NEGATIVITY_SEEDS,
                 "_resolved": RESOLVED_SEEDS, "_praise": PRAISE_SEEDS})
_extra = os.getenv("KP_TEXT_SEEDS", os.path.join("data", "hazard_seeds.json"))
if os.path.exists(_extra):
    try:
        BANK.load_json(_extra)
    except Exception as _exc:                                  # a broken user file must never break the import
        log.warning("could not load extra seeds from %s: %s", _extra, _exc)
HAZARD_CONCEPTS = tuple(HAZARD_SEEDS)


# ---------------------------------------------------------------- deterministic lexicon layer
# Exact (normalised) phrase lookup over the alignment dictionary. It runs next to the embedding similarity, so a known
# civic term in either language is ALWAYS recognised, whichever embedder is installed. Keys are token tuples after
# _norm() + roman_normalize(), so 'Potholes', 'potholes' and Urdu spellings with diacritics or ZWNJ all collapse.
def _lex_key(phrase):
    return tuple(roman_normalize(w) for w in _norm(phrase).split())


LEXICON: dict = {}
for _concept, _items in HAZARD_SEEDS.items():
    _d = LEXICON.setdefault(_concept, {})
    for _phrase, _w in _items:
        _k = _lex_key(_phrase)
        if _k and (_k not in _d or _d[_k][0] < _w):
            _d[_k] = (_w, _phrase)
MAX_LEX_WORDS = max(len(_k) for _d in LEXICON.values() for _k in _d)

_PRAISE_RES = [re.compile(r"(?<!\w)(?:" + "|".join(normalize_urdu(p) for p in pats) + r")(?!\w)")
               for pats in HIGH_INTENSITY_PRAISE.values()]


def detect_high_intensity_praise(text: str) -> list:
    """High-intensity praise phrases ('excellent work', 'great job', 'بہت شکریہ', 'شاباش', ...) found in `text`,
    longest first, with matches that are part of a longer match removed."""
    norm = _norm(text)
    found = []
    for rx in _PRAISE_RES:
        found.extend(m.group(0).strip() for m in rx.finditer(norm) if m.group(0).strip())
    found = sorted(set(found), key=len, reverse=True)
    kept = []
    for f in found:
        if not any(f in k for k in kept):
            kept.append(f)
    return kept


def lexicon_hits(text: str) -> dict:
    """{concept: [phrase, ...]} for every alignment-dictionary term found in `text` (ignores negation; for inspection
    and tests - the scorer uses the same matching through _Spans)."""
    out = {}
    for seg in _segments(text):
        for toks in seg:
            normed = [roman_normalize(w) for w in toks]
            for concept, table in LEXICON.items():
                for size in range(1, MAX_LEX_WORDS + 1):
                    for st in range(0, len(normed) - size + 1):
                        hit = table.get(tuple(normed[st:st + size]))
                        if hit:
                            out.setdefault(concept, [])
                            if hit[1] not in out[concept]:
                                out[concept].append(hit[1])
    return out


# ---------------------------------------------------------------- privacy
def redact(text: str) -> str:
    """Removes emails, phone numbers, @mentions and exact coordinates from stored text (raw handles are never kept)."""
    from geospatial.gazetteer import redact_coordinates     # a precise coordinate in the text would defeat location privacy
    text = _EMAIL_RE.sub("[email]", text or "")
    text = _PHONE_RE.sub("[phone]", text)
    text = redact_coordinates(text)
    return _MENTION_RE.sub("[user]", text)


_salt_warned = False


def author_hash(platform: str, handle) -> str | None:
    """Salted one-way pseudonym so 'same author' can be detected without storing who the author is."""
    global _salt_warned
    if not handle:
        return None
    if config.OSINT_PRIVACY_SALT == "change-me-in-dotenv" and not _salt_warned:
        import logging
        logging.getLogger(__name__).warning("KP_PRIVACY_SALT is not set; author pseudonyms use the default salt "
                                            "and can be brute-forced. Set it in .env.")
        _salt_warned = True
    return hashlib.sha256(f"{config.OSINT_PRIVACY_SALT}|{platform}|{str(handle).strip().lower()}".encode()).hexdigest()[:16]




# words that carry no meaning on their own; a "resolved" claim must rest on a CONTENT word ("theek", "fixed"), never on
# the shared filler of "hal ho gaya" / "pani jama ho gaya" (which would mark a live flood as already fixed)
_FILLER = {"ہو", "گیا", "گئی", "گئے", "ہے", "ہیں", "تھا", "تھی", "تھے", "ہوا", "ہوئی", "ہوئے", "کا", "کی", "کے", "کو", "بھی",
           "اب", "نے", "کر", "دیا", "دی", "لیا", "سے", "پر", "میں",
           "has", "have", "been", "now", "is", "was", "are", "a", "the", "it", "its", "to", "of", "in", "on"}


def _content_tokens(tokens):
    return [roman_normalize(t) for t in tokens if t not in _FILLER]


def _bigrams(w):
    w = f"^{w}$"
    return {w[i:i + 2] for i in range(len(w) - 1)}


def _shares_content(span_tokens, seed_phrase):
    """True if some content word of the span is (spelling-variant) equal to some content word of the seed."""
    seed = _content_tokens(_norm(seed_phrase).split())
    mine = _content_tokens(span_tokens)
    if not seed or not mine:
        return False
    for a in mine:
        for b in seed:
            if a == b:
                return True
            ba, bb = _bigrams(a), _bigrams(b)
            if len(ba | bb) and len(ba & bb) / len(ba | bb) >= 0.6:
                return True
    return False

# ---------------------------------------------------------------- matching helpers
def _noisy_or(ws):
    p = 1.0
    for w in ws:
        p *= 1.0 - min(max(w, 0.0), 1.0)
    return 1.0 - p


SUPPORTED_LANGUAGES = ("en", "ur", "mixed")           # English, Urdu script, or both in one post
UNSUPPORTED_LANGUAGES = ("roman-ur", "other")


def detect_language(text: str) -> str:
    """'en' (Latin script, English), 'ur' (Urdu/Arabic script), 'mixed' (both scripts), 'roman-ur' (Urdu written in
    Latin letters: recognised in order to be REFUSED), or 'other' (letters of some other script, e.g. Cyrillic, Devanagari,
    CJK). Roman Urdu is spotted by function-word markers: at least ROMAN_URDU_MIN_HITS distinct markers AND at least
    ROMAN_URDU_MIN_SHARE of the words, so an English sentence naming 'Gulshan ka road' is not rejected."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return "en"
    n = len(letters)
    urdu = sum(1 for c in letters if _URDU_RE.match(c))
    latin = sum(1 for c in letters if _LATIN_RE.match(c))
    if (n - urdu - latin) / n > 0.5:
        return "other"
    if urdu / n > 0.6:
        return "ur"
    if urdu / n > 0.15:
        return "mixed"
    words = _norm(text).split()
    hits = {w for w in words if w in ROMAN_URDU_MARKERS}
    if len(hits) >= ROMAN_URDU_MIN_HITS and sum(w in ROMAN_URDU_MARKERS for w in words) / max(len(words), 1) >= ROMAN_URDU_MIN_SHARE:
        return "roman-ur"
    return "en"


# ---------------------------------------------------------------- structure: segments, spans, negation, intensity
_CONNECTORS = {"and", "or", "because", "while", "اور", "یا", "کیونکہ", "جبکہ"}


def _segments(text):
    """-> list of segments, each a list of comma-separated clauses, each a list of normalised tokens.
    Segments end at . ! ? ; or a contrast word ('lekin', 'but', ...): praise in one segment must not colour a hazard in
    a contrasting one ("road is nice BUT potholes everywhere"). Clauses end at commas and bound negation windows
    ("pani jama hai, koi nahi sunta": the 'nahi' belongs to the second clause, not to the water)."""
    low = normalize_urdu(_URL_RE.sub(" ", (text or "").lower()))
    out = []
    for part in re.split(rf"[.!?;\n\u06D4\u061F\u061B]+|(?<!\w){CONTRAST}(?!\w)", low):
        clauses = [c for c in (_norm(x).split() for x in re.split(r"[,\u060C]", part)) if c]
        if clauses:
            out.append(clauses)
    return out


def _spans(segs, backend):
    """Candidate spans (seg, clause, start, end). Hashed backend: every 1..4-word window inside a clause. ONNX backend
    (one forward pass per span is expensive): each whole clause plus 3-word windows with stride 2, capped."""
    spans = []
    for si, clauses in enumerate(segs):
        for ci, toks in enumerate(clauses):
            n = len(toks)
            if backend == "hashed-char-ngram":
                for size in range(1, MAX_SPAN_WORDS + 1):
                    for st in range(0, n - size + 1):
                        spans.append((si, ci, st, st + size))
            else:
                spans.append((si, ci, 0, min(n, 24)))
                for st in range(0, max(1, n - 2), 2):
                    spans.append((si, ci, st, min(n, st + 3)))
    return spans[:MAX_SPANS.get(backend, 200)]


def _trim(toks):
    """Cut a context window at the first connector so negation does not leak across 'and' / 'aur'."""
    for i, t in enumerate(toks):
        if t in _CONNECTORS:
            return toks[:i]
    return toks


_ALL_NEGATORS = PRE_NEGATORS | POST_NEGATORS


def is_negated(toks, start, end, *, window=3, resolve=False, allow_inner=False):
    """True if a negator governs toks[start:end]. English negators precede the phrase ("no pothole", "not flooded");
    Roman-Urdu / Urdu verb-final negation follows it ("gaddha nahi hai", "pani jama nahi"). A negator INSIDE the span
    also negates it - unless the matched seed phrase itself contains one (allow_inner, e.g. "dhakkan nahi" = cover
    missing, which is the hazard, not its denial). resolve=True also treats 'yet / still / abhi tak' as blocking a
    'fixed' claim. toks is one comma-delimited clause."""
    before = _trim(toks[max(0, start - window):start][::-1])[::-1] if start else []
    after = _trim(toks[end:end + window])
    pre = PRE_NEGATORS - {"nahi", "nahin", "nhi"}
    if resolve:
        pre = pre | RESOLVE_BLOCKERS
    post = POST_NEGATORS | ({"yet", "still"} if resolve else set())
    inner = [] if allow_inner else toks[start:end]
    return (any(t in pre for t in before) or any(t in post for t in after)
            or any(t in _ALL_NEGATORS for t in inner))


def intensity(toks, start, end):
    """Multiplier from intensifiers ('bohat', 'very', 'huge') and downtoners ('thora', 'slight') next to the phrase;
    a doubled intensifier ('bohat bohat') adds a little more. Bounded to [0.6, 1.3] so words cannot dominate evidence."""
    near = toks[max(0, start - 2):start] + toks[end:end + 1]
    nb, nd = sum(t in BOOSTERS for t in near), sum(t in DIMINISHERS for t in near)
    redup = any(a == b and a in BOOSTERS for a, b in zip(near, near[1:]))
    return float(min(1.3, max(0.6, (1.15 ** nb) * (0.8 ** nd) * (1.1 if redup else 1.0))))


class _Spans:
    """All candidate spans of one post with their embeddings, scored lazily per concept."""

    def __init__(self, text, embedder):
        self.segs = _segments(text)
        self.backend = embedder.name
        self.spans = _spans(self.segs, embedder.name)
        self.embedder = embedder
        self.raw = text or ""
        self.normed = [[[roman_normalize(w) for w in cl] for cl in seg] for seg in self.segs]
        self.E = (embedder.encode([" ".join(self.segs[si][ci][a:b]) for si, ci, a, b in self.spans])
                  if self.spans else np.zeros((0, embedder.dim or 1), np.float32))
        self._cache = {}

    def scored(self, concept, bank):
        """list of dicts {seg, a, b, s, cos, phrase, negated} for spans with s > 0, strongest first."""
        if concept in self._cache:
            return self._cache[concept]
        out = []
        M, pri, phrases = bank.matrix(concept, self.embedder)
        if len(phrases) and len(self.spans):
            lo, hi = SIM_BAND.get(self.backend, (0.5, 0.85))
            sims = self.E @ M.T
            strength = np.clip((sims - lo) / (hi - lo), 0.0, 1.0) * pri[None]
            k = strength.argmax(axis=1)
            idx = np.arange(len(self.spans))
            s, cos = strength[idx, k], sims[idx, k]
            resolve = concept == "_resolved"
            for i in np.nonzero(s > 0.05)[0]:
                si, ci, a, b = self.spans[i]
                phrase = phrases[k[i]]
                if resolve and self.backend == "hashed-char-ngram" and not _shares_content(
                        self.segs[si][ci][a:b], phrase):
                    continue                       # filler-only overlap: not evidence that anything was fixed
                inner_ok = any(t in _ALL_NEGATORS for t in phrase.split())
                out.append({"seg": si, "cl": ci, "a": a, "b": b, "s": float(s[i]), "cos": float(cos[i]), "phrase": phrase,
                            "negated": is_negated(self.segs[si][ci], a, b, resolve=resolve, allow_inner=inner_ok)})
        table = LEXICON.get(concept)
        if table:
            for si, seg in enumerate(self.normed):
                for ci, normed in enumerate(seg):
                    for size in range(1, MAX_LEX_WORDS + 1):
                        for st in range(0, len(normed) - size + 1):
                            hit = table.get(tuple(normed[st:st + size]))
                            if hit is None:
                                continue
                            weight, phrase = hit
                            inner_ok = any(w in _ALL_NEGATORS for w in _norm(phrase).split())
                            out.append({"seg": si, "cl": ci, "a": st, "b": st + size, "s": float(weight), "cos": 1.0,
                                        "phrase": phrase, "source": "lexicon",
                                        "negated": is_negated(self.segs[si][ci], st, st + size, allow_inner=inner_ok)})
        out.sort(key=lambda d: -d["s"])
        self._cache[concept] = out
        return out


def _top_non_overlapping(cands, k=3, floor=0.15):
    chosen = []
    for c in cands:
        if c["s"] < floor:
            break
        if all(not (c["seg"] == d["seg"] and c["cl"] == d["cl"] and c["a"] < d["b"] and d["a"] < c["b"]) for d in chosen):
            chosen.append(c)
            if len(chosen) == k:
                break
    return chosen


def _concept_value(sp, concept, bank):
    """noisy-OR of the strongest non-overlapping, NON-negated spans -> (value, [phrases])."""
    live = [c for c in sp.scored(concept, bank) if not c["negated"]]
    top = _top_non_overlapping(live)
    return _noisy_or([c["s"] for c in top]), [c["phrase"] for c in top]


def _analyze_with(embedder, text, n_images, bank):
    sp = _Spans(text, embedder)
    urgency, urg_phrases = _concept_value(sp, "_urgency", bank)
    negativity, _ = _concept_value(sp, "_negativity", bank)
    lang = detect_language(text or "")

    # ---- hazard concepts: best un-negated evidence per concept; remember denied mentions too
    found, denied = {}, {}
    for concept in HAZARD_CONCEPTS:
        cands = sp.scored(concept, bank)
        live = _top_non_overlapping([c for c in cands if not c["negated"]])
        if live:
            found[concept] = (_noisy_or([c["s"] for c in live]), live)
        gone = [c for c in cands if c["negated"] and c["s"] >= MIN_HAZARD_STRENGTH]
        if gone:
            denied[concept] = gone[0]
    strong = {c: v for c, v in found.items() if v[0] >= MIN_HAZARD_STRENGTH}

    irony = bool(re.search(r"[!?]{2,}", text or ""))
    irony_words = {t for seg in sp.segs for cl in seg for t in cl} & IRONY_MARKERS
    praise_live = [c for c in sp.scored("_praise", bank) if not c["negated"] and c["s"] >= 0.5]
    praise_segs = {c["seg"] for c in praise_live}
    resolved_live = [c for c in sp.scored("_resolved", bank) if not c["negated"] and c["s"] >= 0.5]
    resolved_segs = {c["seg"] for c in resolved_live}

    praise_terms = detect_high_intensity_praise(text or "")
    base = {"urgency": round(urgency, 3), "negativity": round(negativity, 3), "lang": lang, "backend": embedder.name,
            "negated": False, "sarcasm": False, "needs_review": False, "supported": True, "flags": [],
            "praise_terms": praise_terms}
    if not strong:
        ambiguous = [d for d in denied.values() if (d["seg"] in praise_segs or irony or irony_words)]
        if ambiguous:          # "koi gaddha nahi, wah!" - sincere praise OR irony; do not trust either reading
            d = max(ambiguous, key=lambda x: x["s"])
            concept = next(c for c, v in denied.items() if v is d)
            strength = min(1.0, d["s"] * 0.5)
            signal = min(1.0, strength * (0.75 + 0.25 * max(negativity, urgency)))
            return {**base, "hazard_type": concept, "hazard_strength": round(strength, 3), "community_signal": round(signal, 3),
                    "status": "active", "keywords": [d["phrase"], "~ambiguous"], "negated": True, "needs_review": True}
        if denied:
            d = max(denied.values(), key=lambda x: x["s"])
            return {**base, "hazard_type": None, "hazard_strength": 0.0, "community_signal": 0.0, "status": "irrelevant",
                    "keywords": [f"~negated:{d['phrase']}"], "negated": True}
        return {**base, "hazard_type": None, "hazard_strength": 0.0, "community_signal": 0.0, "status": "irrelevant",
                "keywords": []}

    hazard, (strength, live) = max(strong.items(), key=lambda kv: (kv[1][0], len(kv[1][1])))
    best = live[0]
    mult = intensity(sp.segs[best["seg"]][best["cl"]], best["a"], best["b"])
    strength = min(1.0, strength * mult)
    # sarcasm: praise sitting in the same clause as an un-negated hazard (and not thanks-for-fixing)
    # (a) praise in the same sentence, or (b) praise in the next/previous sentence when the writer signals irony with
    # '!!' / '?!' ("Another pothole. Love it!!")
    near_praise = praise_segs | ({s + 1 for s in praise_segs} | {s - 1 for s in praise_segs} if irony else set())
    structural_sarcasm = any(c["seg"] in near_praise and c["seg"] not in resolved_segs for c in live)
    # rule: high-intensity praise anywhere in the same text block as a classified, un-negated hazard term, unless the
    # praise is thanks for a fix ("great job fixing the pothole" - a resolved claim is present)
    rule_sarcasm = bool(praise_terms) and not resolved_live
    sarcasm = structural_sarcasm or rule_sarcasm
    if sarcasm:
        urgency *= SARCASM_URGENCY_DAMPING          # jokes about danger must not inflate the urgency score
        negativity = max(negativity, SARCASM_NEGATIVITY_FLOOR)
    resolved = bool(resolved_live)
    media = 0.10 * strength if n_images > 0 else 0.0
    signal = min(1.0, strength * (0.75 + 0.25 * max(negativity, urgency)) + media)
    keywords = list(dict.fromkeys([c["phrase"] for c in live] + urg_phrases[:3])) + (["~sarcasm"] if sarcasm else [])
    return {**base, "hazard_type": hazard, "hazard_strength": round(strength, 3), "urgency": round(urgency, 3),
            "negativity": round(negativity, 3), "community_signal": round(signal, 3),
            "status": "resolved" if resolved else "active", "keywords": keywords, "sarcasm": sarcasm,
            "sarcasm_rule": ("praise+hazard" if rule_sarcasm else "structural") if sarcasm else None,
            "flags": ["sarcasm"] if sarcasm else [], "needs_review": bool(rule_sarcasm),
            "similarity": round(best["cos"], 3)}


# ---------------------------------------------------------------- main entry
def analyze_text(text: str, n_images: int = 0, *, embedder=None, bank=None) -> dict:
    """Scores one report. Never raises: if the preferred embedder fails at run time the hashed fallback is used."""
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    bank = bank or BANK
    emb = embedder or get_embedder()
    lang = detect_language(text)
    if lang in UNSUPPORTED_LANGUAGES:
        log.info("unsupported language %r: report not analysed (supported: standard English and Urdu script)", lang)
        return {"hazard_type": None, "hazard_strength": 0.0, "urgency": 0.0, "negativity": 0.0, "community_signal": 0.0,
                "status": "irrelevant", "keywords": [f"~unsupported:{lang}"], "lang": lang, "backend": emb.name,
                "negated": False, "sarcasm": False, "sarcasm_rule": None, "needs_review": False, "supported": False,
                "flags": ["unsupported_language"], "praise_terms": []}
    try:
        return _analyze_with(emb, text, n_images, bank)
    except Exception as exc:
        if emb.name == "hashed-char-ngram":
            raise
        log.warning("embedder %s failed (%s); falling back to hashed n-grams for this post", emb.name, exc)
        return _analyze_with(HashedNgramEmbedder(), text, n_images, bank)


# ---------------------------------------------------------------- near-duplicate detection
def shingles(text: str, k: int = 3) -> frozenset:
    toks = _norm(text).split()
    if len(toks) < k:
        return frozenset([" ".join(toks)]) if toks else frozenset()
    return frozenset(" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1))


def jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class DuplicateDetector:
    """Remembers the last N posts and flags a new one that is a copy-paste (Jaccard over 3-word shingles).
    Copies still get stored (provenance), but fusion ignores them so a viral forward cannot fake corroboration."""

    def __init__(self, memory=None, threshold=None):
        self.threshold = config.OSINT_DEDUP_JACCARD if threshold is None else threshold
        self._recent = deque(maxlen=memory or config.OSINT_DEDUP_MEMORY)   # (db_id, shingles)

    def seed(self, rows):
        for db_id, text in rows:
            self._recent.append((db_id, shingles(text)))

    def check(self, text):
        """-> id of the earlier near-identical post, or None."""
        s = shingles(text)
        for db_id, other in self._recent:
            if jaccard(s, other) >= self.threshold:
                return db_id
        return None

    def add(self, db_id, text):
        self._recent.append((db_id, shingles(text)))
