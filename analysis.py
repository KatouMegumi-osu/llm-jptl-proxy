# analysis.py
import re
import json
import spacy
from janome.tokenizer import Tokenizer
import argostranslate.package
import argostranslate.translate
import asyncio
import config
from config import DICTIONARY_FILE, CUSTOM_DICT_FILE, DICT_IGNORE_LIST_FILE

# Globals
tokenizer = None
grammar_analyzer = None
mtl_translator = None
JMDICT_BASE = {}

class GrammarAnalyzer:
    def __init__(self):
        try:
            print("Loading GiNZA/spaCy model...")
            self.nlp = spacy.load("ja_ginza")
            print("GiNZA model loaded successfully.")
        except OSError:
            print("!!! GRAMMAR MODEL ERROR: GiNZA model 'ja_ginza' not found. !!!")
            self.nlp = None

    def parse(self, text: str) -> dict:
        if not self.nlp:
            return {"dependency": "", "ner": [], "linguistics": {}}
        
        analysis_text = re.sub(r'\([MF]\)(?:\(MC\))?', '', text).strip()
        doc = self.nlp(analysis_text)
        
        structured_analyses = []
        for sent in doc.sents:
            root = sent.root
            subject = "".join(tok.text for tok in root.children if "nsubj" in tok.dep_) or "omitted"
            obj = "".join(tok.text for tok in root.children if "obj" in tok.dep_) or "omitted"
            analysis_str = f"\"{sent.text}\" -> {root.lemma_}(Subject: {subject}, Object: {obj}"
            clauses = []
            for token in sent:
                if token.dep_ == "advcl":
                    marker = next((child for child in token.children if child.dep_ == "mark"), None)
                    clause_type = f" ({marker.text})" if marker else ""
                    clauses.append(f"{''.join(t.text for t in token.subtree)}{clause_type}")
            if clauses:
                analysis_str += f", Clauses: [{'; '.join(clauses)}]"
            analysis_str += ")"
            structured_analyses.append(analysis_str)
        
        ner_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        honorifics_found = set()
        formality = "Plain/Informal"
        honorific_pattern = re.compile(r'(?<!て)(\w+)((?:ちゃん|ちゃま|くん|君|さん|様|さま|先輩|せんぱい|先生|せんせい|氏|殿))')
        matches = honorific_pattern.findall(text)
        for name, honorific in matches: honorifics_found.add(f"{name}{honorific}")
        for token in doc:
            if 'Polite' in token.morph.get("Polite", []):
                formality = "Polite (desu/masu form)"; break
        
        linguistics = {"honorifics": list(honorifics_found), "formality": formality}
        return {"dependency": "\n".join(structured_analyses), "ner": ner_entities, "linguistics": linguistics}

def setup_analyzers():
    global tokenizer, grammar_analyzer, mtl_translator
    print("Loading Janome tokenizer..."); tokenizer = Tokenizer(); print("Tokenizer loaded.")
    grammar_analyzer = GrammarAnalyzer()
    try:
        print("Loading Argos Translate model (ja->en)...")
        installed_langs = argostranslate.translate.get_installed_languages()
        ja = next(filter(lambda x: x.code == "ja", installed_langs))
        en = next(filter(lambda x: x.code == "en", installed_langs))
        mtl_translator = ja.get_translation(en)
        print("Argos Translate model loaded.")
    except Exception as e:
        print(f"!!! MTL MODEL ERROR: {e} !!!"); mtl_translator = None

def load_dictionary():
    global JMDICT_BASE
    try:
        print(f"Loading main dictionary file: {DICTIONARY_FILE}...")
        with open(DICTIONARY_FILE, 'r', encoding='utf-8') as f: JMDICT_BASE = json.load(f)
        print(f"Main dictionary loaded with {len(JMDICT_BASE)} entries.")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"!!! MAIN DICTIONARY ERROR: Could not load '{DICTIONARY_FILE}'. !!!")

def load_ignore_list():
    if DICT_IGNORE_LIST_FILE.exists():
        try:
            with open(DICT_IGNORE_LIST_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                config.IGNORE_LIST = set(data) if isinstance(data, list) else set()
                print(f"Loaded {len(config.IGNORE_LIST)} words into ignore list.")
        except Exception as e:
            print(f"!!! IGNORE LIST ERROR: {e} !!!")

def extract_character_genders(text: str) -> dict:
    pattern = re.compile(r"(\w+)\s*\(([MF])\)")
    matches = pattern.findall(text)
    return {name: ("Female" if g == "F" else "Male") for name, g in matches}

def load_custom_dictionary() -> dict:
    try:
        with open(CUSTOM_DICT_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def find_japanese_words(text: str) -> list[str]:
    if not tokenizer: return []
    words = set()
    for token in tokenizer.tokenize(text):
        if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞', '副詞']:
            words.add(token.base_form); words.add(token.surface)
    return list(words)

def get_speaker_from_line(line: str) -> str | None:
    match = re.match(r"(\w+)", line)
    return match.group(1) if match else None

def resolve_subject_and_pov(messages: list, current_text: str) -> dict:
    context = {}
    full_history = " ".join(msg.get("content", "") for msg in messages)
    pov_match = re.search(r"(\w+)\s*\([MF]\)\(MC\)", full_history)
    if pov_match:
        context["pov_character"] = pov_match.group(1)

    passive_match = re.search(r"(\w+)\(?[MF]?\)?さんに?避けられて", current_text)
    if passive_match:
        avoider = passive_match.group(1)
        all_names = re.findall(r"(\w+)\(?[MF]?\)?", current_text)
        current_speaker = get_speaker_from_line(current_text)
        for name in all_names:
            if name != avoider and name != current_speaker:
                context["likely_subject"] = f"{name} (is being avoided by {avoider})"
                print(f"Subject/POV: Detected passive voice. Subject is likely '{name}'.")
                return context

    if "好き" in current_text or "恋" in current_text:
        current_speaker = get_speaker_from_line(current_text)
        if not current_speaker:
            user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
            if len(user_messages) > 1:
                current_speaker = get_speaker_from_line(user_messages[-2])
        
        if current_speaker:
            user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
            for msg_text in reversed(user_messages[:-1]):
                if any(keyword in msg_text for keyword in ["心当たり", "変なんだ", "相談してる", "嘘ついた"]):
                    previous_speaker = get_speaker_from_line(msg_text)
                    if previous_speaker and current_speaker != previous_speaker:
                        context["likely_subject"] = previous_speaker
                        print(f"Subject/POV: Detected teasing context. Subject likely '{previous_speaker}'.")
                        return context
    return context

# <-- START: THIS IS THE FIX. Reverted to the correct return format. -->
async def get_definitions_async(text: str) -> dict:
    custom_dict = load_custom_dictionary()
    defs = {k: v for k, v in custom_dict.items() if k in text}
    tokens = await asyncio.to_thread(find_japanese_words, text)
    
    unknown_word_count = 0
    
    if config.IGNORE_LIST:
        tokens = [w for w in tokens if w not in config.IGNORE_LIST]
    def is_substring(t, f_keys): return any(t in k for k in f_keys)
    filtered = [t for t in tokens if not is_substring(t, defs.keys())]
    
    for word in filtered:
        if word in custom_dict: 
            defs[word] = custom_dict[word]
        elif word in JMDICT_BASE: 
            defs[word] = JMDICT_BASE[word]
        else:
            unknown_word_count += 1

    return {"definitions": defs, "unknown_word_count": unknown_word_count}
# <-- END: THIS IS THE FIX. -->

async def analyze_grammar_async(text: str) -> dict:
    if not grammar_analyzer or not grammar_analyzer.nlp: return {}
    return await asyncio.to_thread(grammar_analyzer.parse, text)

async def analyze_genders_async(text: str) -> dict:
    return await asyncio.to_thread(extract_character_genders, text)

async def translate_with_mtl_async(text: str) -> str:
    if not mtl_translator: return "MTL model not available."
    return await asyncio.to_thread(mtl_translator.translate, text)

async def analyze_character_voice_async(speaker: str, messages: list) -> dict:
    if not speaker or speaker == "Unknown": return {}
    if speaker in config.CHARACTER_VOICES: return config.CHARACTER_VOICES[speaker]

    user_lines = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
    past_lines = [line for line in user_lines if line.startswith(speaker)][-50:]
    if not past_lines: return {}

    enders = [e for e in ["のです", "だぜ", "わよ"] if any(line.endswith(f"{e}」") for line in past_lines)]
    pronouns = [p for p in ["私", "僕", "俺", "わし"] if any(p in line for line in past_lines)]
    
    voice = {}
    if enders: voice["ender"] = max(set(enders), key=enders.count)
    if pronouns: voice["pronoun"] = max(set(pronouns), key=pronouns.count)
    
    if voice:
        config.CHARACTER_VOICES[speaker] = voice
        print(f"Cached new voice profile for '{speaker}': {voice}")
    return voice