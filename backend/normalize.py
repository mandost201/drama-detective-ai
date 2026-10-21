"""Multilingual text normalization for Drama Detective AI.

Users describe scenes in English, Roman Urdu, Urdu (Arabic script) or
Hindi (Devanagari), while most drama subtitles on YouTube are English
(sometimes Urdu/Hindi). This module only rewrites the *query* side:

  * script detection
  * canonical spelling variants for common Roman Urdu words
    (shadi ~ shaadi, wakeel ~ vakil, nhi ~ nahi ...)
  * a Roman Urdu / Urdu / Hindi -> English concept map used to expand
    keyword queries, so "vakil sahab, dulha dulhan aa gaye" can match
    the English subtitle "Lawyer... the groom and bride are here"
  * fuzzy term matching for name spelling variants (suleman ~ suleiman)

It never invents dialogue or transcript text; matches are always
checked against the real subtitle text of the episodes.
"""
import difflib
import re

# ------------------------------------------------------------------ scripts
_URDU_CHARS = re.compile(r"[؀-ۿ]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def detect_script(text):
    """Return 'hi' (Devanagari), 'ur' (Arabic script) or 'latin'."""
    t = text or ""
    if _DEVANAGARI.search(t):
        return "hi"
    if _URDU_CHARS.search(t):
        return "ur"
    return "latin"


def _normalize_script(text):
    """Unify Arabic-script letter variants so Urdu text compares equal
    regardless of the keyboard it was typed on (Arabic yeh/kaf vs Urdu)."""
    t = text or ""
    t = t.replace("ي", "ی").replace("ك", "ک")
    t = t.replace("ـ", "")  # tatweel
    t = t.replace("ة", "ہ").replace("ؤ", "و")
    return t


def tokens(text):
    """Lowercased word tokens for Latin, Urdu and Devanagari text."""
    return _TOKEN.findall(_normalize_script(text or "").lower())


# ------------------------------------------- Roman Urdu spelling variants
# canonical form -> known variants (every variant maps back to canonical)
_VARIANT_GROUPS = [
    ("hai", ["he", "hay", "ha"]),
    ("nahi", ["nhi", "nahin", "nahiin", "nai", "naheen"]),
    ("kya", ["kia", "kiya", "kyaa"]),
    ("mujhe", ["mujhy", "muje", "mujhay"]),
    ("tujhe", ["tujhy", "tuje", "tujhay"]),
    ("usey", ["use", "usay", "usse"]),
    ("isey", ["ise", "isay", "isse"]),
    ("shaadi", ["shadi", "shadhi", "shaadee", "shadee"]),
    ("dulha", ["dulhaa", "doolha", "dulhe"]),
    ("dulhan", ["dulhun", "doolhan", "dulhanain"]),
    ("vakil", ["wakeel", "vakil", "vakeel"]),
    ("adalat", ["adaalat", "adalt"]),
    ("paise", ["paisay", "paisa", "paisey", "rupay", "rupees"]),
    ("mohabbat", ["muhabbat", "mohabat", "muhabat"]),
    ("larka", ["ladka", "larkaa"]),
    ("larki", ["ladki", "larkee"]),
    ("behan", ["behn", "behen", "bhen"]),
    ("bacha", ["baccha", "bachaa", "bache"]),
    ("goli", ["golee", "goliyan", "golian"]),
    ("ansoo", ["aansoo", "aansoo", "ansoon"]),
    ("wapas", ["waapas", "wapis"]),
    ("sheher", ["shehr", "shaher", "city"]),
    ("gaon", ["gau", "gaao", "ganv"]),
    ("nikah", ["nikaah", "nikkah"]),
    ("talaq", ["talaaq", "talak"]),
    ("rote", ["rota", "roti", "rone", "rona"]),
    ("khamosh", ["khaamosh", "khamoshi"]),
    ("inteqam", ["intiqam", "intqam", "badla"]),
    ("dushman", ["dushmn", "dushman"]),
    ("mehman", ["mehmaan", "mehmano"]),
    ("sultan", ["sultaan", "sultan"]),
    ("suleman", ["suleiman", "sulaiman", "sulemani", "solaiman"]),
]

_CANONICAL = {}
for _canon, _variants in _VARIANT_GROUPS:
    _CANONICAL[_canon] = _canon
    for _v in _variants:
        _CANONICAL[_v] = _canon


def canonical(term):
    """Canonical spelling of a Roman Urdu term (identity if unknown)."""
    t = (term or "").lower()
    return _CANONICAL.get(t, t)


def variants_for(term):
    """All known spellings of a term's variant group (incl. canonical),
    excluding the term itself. Lets keyword search hit subtitles that
    spell a name differently than the user typed it."""
    t = (term or "").lower()
    c = canonical(t)
    for canon, variants in _VARIANT_GROUPS:
        if canon == c:
            return [v for v in [canon] + variants if v != t]
    return []


# --------------------------------------- cross-language concept expansion
# Scene vocabulary: Roman Urdu / Urdu script / Devanagari -> the English
# words official subtitles typically use. Expansion is OR-based: it can
# only add candidate matches, and ranking + coverage still decide.
CONCEPT_MAP = {
    # Roman Urdu
    "shaadi": ["marriage", "wedding", "marry", "married"],
    "nikah": ["nikah", "marriage", "wedding"],
    "dulha": ["groom", "bridegroom"],
    "dulhan": ["bride"],
    "vakil": ["lawyer"],
    "adalat": ["court"],
    "judge": ["judge"],
    "paise": ["money"],
    "fees": ["fee", "fees"],
    "maut": ["death", "died", "dead"],
    "mara": ["died", "killed", "dead"],
    "inteqam": ["revenge"],
    "mohabbat": ["love"],
    "dost": ["friend"],
    "dosti": ["friendship", "friend"],
    "ghar": ["house", "home"],
    "maa": ["mother"],
    "baap": ["father"],
    "beta": ["son"],
    "beti": ["daughter"],
    "bhai": ["brother"],
    "behan": ["sister"],
    "bacha": ["child", "baby"],
    "larka": ["boy"],
    "larki": ["girl"],
    "rote": ["crying", "cry", "tears"],
    "ansoo": ["tears", "crying"],
    "khushi": ["happy", "happiness"],
    "naraz": ["angry", "upset"],
    "talaq": ["divorce"],
    "police": ["police"],
    "hospital": ["hospital"],
    "doctor": ["doctor"],
    "goli": ["bullet", "shot", "gun"],
    "sheher": ["city"],
    "gaon": ["village"],
    "mehman": ["guest"],
    "dushman": ["enemy"],
    "khamosh": ["silent", "silence", "quiet"],
    "chai": ["tea"],
    "kaghaz": ["papers", "paper", "documents"],
    "dastakhat": ["sign", "signed", "signature"],
    # Urdu script
    "شادی": ["marriage", "wedding", "marry", "married"],
    "نکاح": ["nikah", "marriage", "wedding"],
    "دلہا": ["groom", "bridegroom"],
    "دلہن": ["bride"],
    "وکیل": ["lawyer"],
    "عدالت": ["court"],
    "جج": ["judge"],
    "پیسے": ["money"],
    "پیسہ": ["money"],
    "موت": ["death", "died", "dead"],
    "محبت": ["love"],
    "دوست": ["friend"],
    "گھر": ["house", "home"],
    "ماں": ["mother"],
    "باپ": ["father"],
    "بیٹا": ["son"],
    "بیٹی": ["daughter"],
    "بھائی": ["brother"],
    "بہن": ["sister"],
    "بچہ": ["child", "baby"],
    "لڑکا": ["boy"],
    "لڑکی": ["girl"],
    "آنسو": ["tears", "crying"],
    "طلاق": ["divorce"],
    "پولیس": ["police"],
    "ہسپتال": ["hospital"],
    "ڈاکٹر": ["doctor"],
    "گولی": ["bullet", "shot", "gun"],
    "شہر": ["city"],
    "گاؤں": ["village"],
    "مہمان": ["guest"],
    "دشمن": ["enemy"],
    "چائے": ["tea"],
    "کاغذ": ["papers", "paper", "documents"],
    "رانیا": ["rania"],
    # Devanagari
    "शादी": ["marriage", "wedding", "marry", "married"],
    "दूल्हा": ["groom", "bridegroom"],
    "दुल्हन": ["bride"],
    "वकील": ["lawyer"],
    "अदालत": ["court"],
    "जज": ["judge"],
    "पैसा": ["money"],
    "पैसे": ["money"],
    "मौत": ["death", "died", "dead"],
    "मोहब्बत": ["love"],
    "दोस्त": ["friend"],
    "घर": ["house", "home"],
    "माँ": ["mother"],
    "पिता": ["father"],
    "बेटा": ["son"],
    "बेटी": ["daughter"],
    "भाई": ["brother"],
    "बहन": ["sister"],
    "बच्चा": ["child", "baby"],
    "लड़का": ["boy"],
    "लड़की": ["girl"],
    "आँसू": ["tears", "crying"],
    "तलाक": ["divorce"],
    "पुलिस": ["police"],
    "अस्पताल": ["hospital"],
    "डॉक्टर": ["doctor"],
    "गोली": ["bullet", "shot", "gun"],
    "शहर": ["city"],
    "गाँव": ["village"],
    "चाय": ["tea"],
}


def expansions_for(term):
    """English subtitle words a query term may correspond to."""
    t = _normalize_script(term or "").lower()
    out = list(CONCEPT_MAP.get(t, []))
    c = canonical(t)
    if c != t:
        for e in CONCEPT_MAP.get(c, []):
            if e not in out:
                out.append(e)
    return out


def expand_query(query):
    """Query text plus English concept expansions and known spelling
    variants (for keyword/FTS). Original tokens come first so term caps
    keep the user's own words."""
    base = " ".join(tokens(query))
    base_set = set(base.split())
    extra = []
    for tok in tokens(query):
        for e in expansions_for(tok) + variants_for(tok):
            if e not in extra and e not in base_set:
                extra.append(e)
    return (base + " " + " ".join(extra)).strip()


def term_matches(term, text):
    """Does `term` (any language) match `text`? Checks the term itself,
    its canonical Roman Urdu form, its English concept expansions, and -
    for longer Latin terms - a strict fuzzy token match (name spellings).
    Used for distinctive-term coverage; errs on the strict side."""
    if not term or not text:
        return False
    tl = _normalize_script(text).lower()
    t = _normalize_script(term).lower()
    candidates = {t, canonical(t)} | set(expansions_for(t))
    for c in candidates:
        if c and c in tl:
            return True
    # fuzzy fallback: long Latin-script terms only (names), high bar
    ct = canonical(t)
    if len(ct) >= 5 and ct.isascii():
        for w in set(tokens(tl)):
            if len(w) >= 4 and abs(len(w) - len(ct)) <= 2:
                if difflib.SequenceMatcher(None, ct, w).ratio() >= 0.86:
                    return True
    return False
