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


# ------------------------------------------------- script transliteration
# Names have no concept-map entries, so an Urdu/Hindi-script NAME
# query ("رانیہ مرزا") used to match nothing at all against English
# subtitles. A mechanical transliteration to Latin - plus the
# spelling folds Roman Urdu is full of - turns it into candidate
# Latin queries the normal pipeline can run. This only ever
# rewrites the QUERY; transcript text is never altered.
_URDU_MAP = {
    "ا": "a", "آ": "a", "ٲ": "a", "ب": "b", "پ": "p", "ت": "t",
    "ٹ": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "kh",
    "د": "d", "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z",
    "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t",
    "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q", "ک": "k",
    "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "o",
    "ہ": "h", "ھ": "h", "ۃ": "h", "ی": "i", "ے": "e", "ئ": "i",
    "ؤ": "o", "ء": "", "ئ": "i",
}
_URDU_DIACRITICS = set("ًٌٍَُِّْٰٓٔ")

_DEV_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch",
    "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ny", "ट": "t", "ठ": "th",
    "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
    "ध": "dh", "न": "n", "प": "p", "फ": "f", "ब": "b", "भ": "bh",
    "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h",
}
_DEV_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d",
              "ढ": "dh", "फ": "f"}
_DEV_VOWEL = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u",
              "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
              "औ": "au", "ऑ": "o"}
_DEV_MATRA = {"ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "oo",
              "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
              "ॉ": "o", "ं": "n", "ः": "h", "ँ": "n"}


def _transliterate_devanagari(text):
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _DEV_CONS:
            base = _DEV_CONS[ch]
            j = i + 1
            if j < n and text[j] == "़":  # nukta changes the letter
                base = _DEV_NUKTA.get(ch, base)
                j += 1
            out.append(base)
            if j < n and text[j] == "्":  # virama: cluster, no vowel
                i = j + 1
                continue
            if j < n and text[j] in _DEV_MATRA:
                out.append(_DEV_MATRA[text[j]])
                i = j + 1
                continue
            out.append("a")  # inherent vowel
            i = j
            continue
        if ch in _DEV_VOWEL:
            out.append(_DEV_VOWEL[ch])
        elif ch in _DEV_MATRA:
            out.append(_DEV_MATRA[ch])
        elif ch in ("़", "्"):
            pass
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def transliterate(text):
    """Latin transliteration of Urdu (Arabic script) and Hindi
    (Devanagari) text; Latin text passes through unchanged."""
    t = _normalize_script(text or "")
    # Urdu: a word-internal یہ is the digraph "ia" (رانیہ = Rania),
    # not i+h. A standalone یہ keeps its own reading.
    t = re.sub(r"(?<=[؀-ۿ])یہ", "ia", t)
    out = []
    for ch in t:
        if ch in _URDU_DIACRITICS:
            continue
        out.append(_URDU_MAP.get(ch, ch))
    t = "".join(out)
    if _DEVANAGARI.search(t):
        t = _transliterate_devanagari(t)
    return t


def _fold_token(tok):
    """Spelling folds for transliterated tokens only: doubled
    vowels collapse, a trailing silent h drops, -iya- reads -ia-.
    (رانیہ -> raniyah -> rania; शादी -> shaadii -> shadi.)"""
    t = tok
    t = t.replace("iya", "ia")
    t = t.replace("aa", "a").replace("ee", "i").replace("oo", "u")
    t = t.replace("ii", "i")
    if len(t) > 3 and t.endswith("h"):
        t = t[:-1]
    return t


def _vowel_inserted(tok):
    """Urdu script omits short vowels: مرزا is written 'mrza' but
    read 'Mirza'. As a LAST-resort candidate, a token that starts
    with two consonants gets an 'i' after its first letter. This
    only ever adds one more candidate query - ranking and
    coverage still decide what matches."""
    if (len(tok) >= 4 and tok[0] not in "aeiou"
            and tok[1] not in "aeiou"):
        return tok[0] + "i" + tok[1:]
    return tok


def latin_candidates(query):
    """For a query written in Urdu or Devanagari script: candidate
    Latin rewrites, best first. Empty list for Latin queries (the
    normal path already handles those)."""
    if detect_script(query or "") == "latin":
        return []
    tr = transliterate(query)
    folded = " ".join(_fold_token(t) for t in tr.split())
    rescued = " ".join(_vowel_inserted(t) for t in folded.split())
    out = []
    for cand in (folded, tr, rescued):
        cand = " ".join(cand.split()).strip()
        if cand and cand not in out:
            out.append(cand)
    return out


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
