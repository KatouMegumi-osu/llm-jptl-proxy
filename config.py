# config.py
import json
import asyncio
from collections import deque
from copy import deepcopy
from pathlib import Path
import csv

# --- File Paths and Constants ---
DICTIONARY_FILE = "jmdict.json"
CUSTOM_DICT_FILE = "custom_dict.json"
DICT_IGNORE_LIST_FILE = Path("dict_ignorelist.json")
PROFILES_FILE = Path("profiles.json")
APP_STATE_FILE = Path("app_state.json")
SUMMARIES_DIR = Path("./summaries")
LANGUAGES_DIR = Path("./languages")
PERSONALITY_TRAIT_PARENT_IDS = { "g445", "g2185", "g448", "g446" }

# --- Global State Variables ---
LOG_BUFFER = deque(maxlen=50)
JAPANESE_SOURCE_BACKLOG = deque(maxlen=200)
ENGLISH_TRANSLATION_BACKLOG = deque(maxlen=200)
IGNORE_LIST = set()
TOTAL_LINE_COUNT = 0
LAST_SUMMARY_LINE_COUNT = 0
SCENE_CONTEXT = {"present_characters": set(), "last_event": ""}
CHARACTER_VOICES = {}
AVAILABLE_LANGUAGES = {}
API_KEY_INDEX = 0

DEFAULT_SETTINGS = {
    "target_api_base_url": "http://localhost:5001",
    "target_api_key": "",
    "target_language": "en",
    "force_app_name": "",
    "timeout_settings": {
        "connect": 10.0,
        "read": 300.0,
        "write": 300.0,
        "pool": 10.0
    },
    "pruning_settings": {"enabled": True, "mode": "prune_english", "keep_n_turns": 8},
    "summarization_settings": {
        "enabled": True, "max_summary_chars": 1000, "lines_per_summary": 50,
        "prompt": (
            "You are a continuity editor for a story. Your task is to update the story summary with new events.\n\n"
            "Read the 'PREVIOUS SUMMARY' for context. Then, read the 'NEW DIALOGUE' and integrate the events from it into the summary.\n\n"
            "The final output must be a single, cohesive narrative paragraph that combines the old summary with the new events. The new summary must be under {max_summary_chars} characters.\n\n"
            "--- PREVIOUS SUMMARY ---\n{previous_summary}\n\n"
            "--- NEW DIALOGUE TO INTEGRATE ---\n{japanese_text}\n\n"
            "--- NEW UPDATED STORY SUMMARY ---"
        )
    },
    "optimization_toggles": {
        "detect_text_type": {"enabled": True},
        "add_mtl_assist": {"enabled": True},
        "add_vocabulary": {"enabled": True},
        "add_grammar": {"enabled": True},
        "add_subject_pov": {"enabled": True, "pov_character": True, "likely_subject": True},
        "add_ner": {"enabled": True, "use_ginza": True, "use_genders_in_ner": True},
        "add_linguistics": {"enabled": True, "honorifics": True, "formality": True},
        "add_char_voice": {"enabled": True},
        "use_corrective_pass": {
            "enabled": False,
            "add_context": True,
            "temperature": 0.4,
            "model_name": "correction/model"
        }
    },
    "proofreader_pruning_settings": {
        "enabled": True,
        "mode": "keep_n_turns",
        "keep_n_turns": 4
    }
}

SETTINGS = {}
PROFILES = {}
ACTIVE_PROFILE_NAME = "default"
SUMMARY_LOCK = asyncio.Lock()
vndb_client = None
llm_client = None

def _merge_settings(default, custom):
    merged = deepcopy(default)
    for key, value in custom.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_settings(merged[key], value)
        elif key in merged and isinstance(merged[key], dict) and isinstance(value, bool):
             merged[key]['enabled'] = value
        else:
            merged[key] = value
    return merged

def load_profiles_from_disk():
    global PROFILES, ACTIVE_PROFILE_NAME
    if not PROFILES_FILE.exists():
        PROFILES = {"default": deepcopy(DEFAULT_SETTINGS)}
        save_profiles_to_disk()
        return
    try:
        with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
            profiles_data = json.load(f)
            for name, profile in profiles_data.items():
                PROFILES[name] = _merge_settings(DEFAULT_SETTINGS, profile)
    except json.JSONDecodeError as e:
        print(f"!!! ERROR: Could not parse '{PROFILES_FILE}'. It may be corrupted. {e} !!!")
        PROFILES = {"default": deepcopy(DEFAULT_SETTINGS)}
        ACTIVE_PROFILE_NAME = "default"

def save_profiles_to_disk():
    with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
        json.dump(PROFILES, f, indent=4, ensure_ascii=False)

def load_app_state():
    global ACTIVE_PROFILE_NAME
    if APP_STATE_FILE.exists():
        with open(APP_STATE_FILE, 'r', encoding='utf-8') as f:
            ACTIVE_PROFILE_NAME = json.load(f).get("active_profile", "default")
    else:
        save_app_state()

def save_app_state():
    with open(APP_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump({"active_profile": ACTIVE_PROFILE_NAME}, f, indent=4, ensure_ascii=False)

def _load_backlog_from_disk(profile_name: str):
    backlog_file = SUMMARIES_DIR / f"{profile_name}.backlog.csv"
    if not backlog_file.exists():
        print(f"No backlog file found for profile '{profile_name}'. Starting fresh.")
        return
    try:
        with open(backlog_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            loaded_pairs = list(reader)
            start_index = max(0, len(loaded_pairs) - JAPANESE_SOURCE_BACKLOG.maxlen)
            for row in loaded_pairs[start_index:]:
                if len(row) >= 3:
                    JAPANESE_SOURCE_BACKLOG.append(row[1])
                    ENGLISH_TRANSLATION_BACKLOG.append(row[2])
        print(f"Loaded {len(JAPANESE_SOURCE_BACKLOG)} entries from backlog file: {backlog_file}")
    except Exception as e:
        print(f"!!! ERROR loading backlog file for profile '{profile_name}': {e} !!!")

def load_counters(profile_name: str):
    global TOTAL_LINE_COUNT, LAST_SUMMARY_LINE_COUNT
    counter_file = SUMMARIES_DIR / f"{profile_name}.counter.json"
    if counter_file.exists():
        try:
            with open(counter_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                TOTAL_LINE_COUNT = data.get("total_lines", 0)
                LAST_SUMMARY_LINE_COUNT = data.get("last_summary_at", 0)
        except (json.JSONDecodeError, OSError) as e:
            print(f"!!! COUNTER FILE ERROR: Could not read {counter_file}. Resetting. Error: {e} !!!")
            TOTAL_LINE_COUNT = 0
            LAST_SUMMARY_LINE_COUNT = 0
    else:
        TOTAL_LINE_COUNT = 0
        LAST_SUMMARY_LINE_COUNT = 0
    print(f"Counters loaded for {profile_name}: Total lines={TOTAL_LINE_COUNT}, Last summary at={LAST_SUMMARY_LINE_COUNT}")

def save_counters(profile_name: str):
    counter_file = SUMMARIES_DIR / f"{profile_name}.counter.json"
    try:
        with open(counter_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_lines": TOTAL_LINE_COUNT,
                "last_summary_at": LAST_SUMMARY_LINE_COUNT
            }, f, indent=2)
    except OSError as e:
        print(f"!!! COUNTER FILE ERROR: Could not write to {counter_file}. Error: {e} !!!")

def load_languages_from_disk():
    global AVAILABLE_LANGUAGES
    LANGUAGES_DIR.mkdir(exist_ok=True)
    for lang_file in LANGUAGES_DIR.glob("*.json"):
        try:
            with open(lang_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "code" in data and "name" in data and "language_name" in data:
                    AVAILABLE_LANGUAGES[data["code"]] = data
        except (json.JSONDecodeError, OSError) as e:
            print(f"!!! LANGUAGE FILE ERROR: Could not load {lang_file}. Error: {e} !!!")
    if not AVAILABLE_LANGUAGES:
        AVAILABLE_LANGUAGES['en'] = {"code": "en", "name": "English", "language_name": "English"}
    print(f"Loaded {len(AVAILABLE_LANGUAGES)} languages.")

def activate_profile(profile_name: str):
    global SETTINGS, ACTIVE_PROFILE_NAME, SCENE_CONTEXT, CHARACTER_VOICES
    if profile_name not in PROFILES:
        print(f"!!! WARNING: Profile '{profile_name}' not found. Activating 'default' profile instead. !!!")
        profile_name = "default"
    
    SETTINGS = deepcopy(PROFILES[profile_name])
    ACTIVE_PROFILE_NAME = profile_name
    print(f"Activated profile: {profile_name}")
    
    JAPANESE_SOURCE_BACKLOG.clear()
    ENGLISH_TRANSLATION_BACKLOG.clear()
    _load_backlog_from_disk(profile_name)
    
    load_counters(profile_name)
    
    SCENE_CONTEXT = {"present_characters": set(), "last_event": ""}
    CHARACTER_VOICES = {}
    
    return True