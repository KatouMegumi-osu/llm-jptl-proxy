# run.py (With Vocabulary, Grammar, AND Named Entity Recognition)
import json
from contextlib import asynccontextmanager
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from janome.tokenizer import Tokenizer
import spacy

# --- 1. Configuration ---
TARGET_API_BASE_URL = "http://localhost:5001"
DICTIONARY_FILE = "jmdict.json"
CUSTOM_DICT_FILE = "custom_dict.json"

# --- 2. Analyzers Setup ---
tokenizer = None
JMDICT_BASE = {}
grammar_analyzer = None

# UPDATED: The GrammarAnalyzer now handles both dependency parsing and NER
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
        """
        Parses text and returns a dictionary containing both the
        dependency parse and any found named entities.
        """
        if not self.nlp:
            return {"dependency": "", "ner": ""}

        doc = self.nlp(text)
        
        # 1. Dependency Parse Logic (unchanged)
        dep_lines = []
        for sent in doc.sents:
            dep_lines.append(f"Sentence: \"{sent.text}\"")
            for token in sent:
                line = (f"  - \"{token.text}\" --({token.dep_})--> \"{token.head.text}\" "
                        f"({token.pos_}, {token.morph.get('Reading')[0] if token.morph.get('Reading') else 'N/A'})")
                dep_lines.append(line)
        
        # 2. NEW: Named Entity Recognition (NER) Logic
        ner_lines = []
        if doc.ents:
            for ent in doc.ents:
                # Format: - "Text" (LABEL)
                ner_lines.append(f"- \"{ent.text}\" ({ent.label_})")

        return {
            "dependency": "\n".join(dep_lines),
            "ner": "\n".join(ner_lines)
        }

def setup_analyzers():
    global tokenizer, grammar_analyzer
    print("Loading Janome tokenizer...")
    tokenizer = Tokenizer()
    print("Tokenizer loaded.")
    grammar_analyzer = GrammarAnalyzer()

def load_dictionary():
    # ... (this function is unchanged)
    global JMDICT_BASE
    try:
        print(f"Loading main dictionary file: {DICTIONARY_FILE}...")
        with open(DICTIONARY_FILE, 'r', encoding='utf-8') as f:
            JMDICT_BASE = json.load(f)
        print(f"Main dictionary loaded successfully with {len(JMDICT_BASE)} entries.")
    except FileNotFoundError:
        print(f"!!! MAIN DICTIONARY ERROR: '{DICTIONARY_FILE}' not found. !!!")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {DICTIONARY_FILE}. The file might be corrupted.")

# ... (load_custom_dictionary and find_japanese_words are unchanged)
def load_custom_dictionary() -> dict:
    try:
        with open(CUSTOM_DICT_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except FileNotFoundError: return {}
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from '{CUSTOM_DICT_FILE}'."); return {}
def find_japanese_words(text: str) -> list[str]:
    if not tokenizer: return []
    words = set()
    tokens = tokenizer.tokenize(text)
    for token in tokens:
        part_of_speech = token.part_of_speech.split(',')[0]
        if part_of_speech in ['名詞', '動詞', '形容詞', '副詞']:
            words.add(token.base_form); words.add(token.surface)
    return list(words)


async def get_definitions_async(text: str) -> dict[str, str]:
    # ... (this function is unchanged)
    japanese_words = await asyncio.to_thread(find_japanese_words, text)
    custom_dict = load_custom_dictionary()
    definitions = {}
    if not japanese_words: return definitions
    for word in japanese_words:
        if word in custom_dict: definitions[word] = custom_dict[word]
        elif word in JMDICT_BASE: definitions[word] = JMDICT_BASE[word]
    return definitions

# UPDATED: This now returns a dictionary of analyses
async def analyze_grammar_async(text: str) -> dict:
    """Runs GiNZA parsing in a thread, returning both dep parse and NER."""
    if not grammar_analyzer or not grammar_analyzer.nlp:
        return {"dependency": "", "ner": ""}
    return await asyncio.to_thread(grammar_analyzer.parse, text)

# --- 3. FastAPI Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... (this function is unchanged)
    setup_analyzers()
    load_dictionary()
    if not load_custom_dictionary(): print(f"Info: '{CUSTOM_DICT_FILE}' not found. You can create it to add custom word definitions.")
    else: print(f"Info: Successfully loaded '{CUSTOM_DICT_FILE}'. It will be re-checked on each request.")
    yield

app = FastAPI(lifespan=lifespan)
client = httpx.AsyncClient()

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_local_llm(request: Request, path: str):
    req_body_bytes = await request.body()
    is_chat_completion = path.rstrip('/').endswith("v1/chat/completions")
    
    if request.method == "POST" and is_chat_completion:
        try:
            req_body_json = json.loads(req_body_bytes)
            messages = req_body_json.get("messages", [])
            if messages and messages[-1].get("role") == "user":
                last_message = messages[-1]
                original_prompt = last_message.get("content", "")
                
                # UPDATED: Run both analyses concurrently and get the results
                definitions, analyses = await asyncio.gather(
                    get_definitions_async(original_prompt),
                    analyze_grammar_async(original_prompt)
                )
                # Unpack the results from the grammar analysis
                grammar_analysis = analyses["dependency"]
                ner_analysis = analyses["ner"]
                
                final_appendix = ""
                final_appendix += "\n\n--- END OF CONVERSATION HISTORY ---\n"
                final_appendix += "Instructions: Using the vocabulary, grammar, and named entity analysis below as a guide, provide a direct English translation of ONLY the last user message. Do not translate anything else.\n"

                if definitions:
                    appendix = "\n--- Vocabulary Appendix ---\n"
                    custom_keys = load_custom_dictionary().keys()
                    for word, definition in definitions.items():
                        marker = " (custom)" if word in custom_keys else ""
                        appendix += f"- {word}:{marker} {definition}\n"
                    final_appendix += appendix
                
                # NEW: Add the Named Entities appendix
                if ner_analysis:
                    final_appendix += "\n--- Named Entities ---\n"
                    final_appendix += ner_analysis

                if grammar_analysis:
                    final_appendix += "\n--- Grammar Analysis (Dependency Parse) ---\n"
                    final_appendix += grammar_analysis
                
                if final_appendix.strip():
                    modified_prompt = original_prompt + final_appendix
                    req_body_json["messages"][-1]["content"] = modified_prompt
                    print("--- Original Final Prompt ---"); print(original_prompt)
                    print("\n--- Modified Final Prompt Sent to LLM ---"); print(modified_prompt)
                    print("------------------------")
                    req_body_bytes = json.dumps(req_body_json).encode('utf-8')

        except (json.JSONDecodeError, IndexError) as e:
            print(f"Could not parse or modify request body: {e}")

    # ... (The rest of the file is unchanged)
    minimal_headers = {
        "Host": "localhost:5001", "User-Agent": "condom-proxy/1.0",
        "Accept": "*/*", "Content-Type": "application/json",
    }
    full_target_url = f"{TARGET_API_BASE_URL.rstrip('/')}{request.url.path}"
    try:
        proxied_req = client.build_request(method=request.method, url=full_target_url, headers=minimal_headers, content=req_body_bytes, timeout=300.0)
        proxied_resp = await client.send(proxied_req, stream=True)
        return StreamingResponse(proxied_resp.aiter_raw(), status_code=proxied_resp.status_code, headers=proxied_resp.headers)
    except httpx.ConnectError as e:
        error_msg = f'{{"error": "Failed to connect to target API at {TARGET_API_BASE_URL}. Is it running?", "details": "{e}"}}'
        return Response(content=error_msg, status_code=502, media_type="application/json")