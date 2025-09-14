# core_logic.py
import json
import re
import asyncio
from copy import deepcopy
import httpx
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import time
from datetime import datetime
import csv
from collections import deque

import config
import analysis

def _save_pair_to_backlog_file(speaker: str, japanese_text: str, english_text: str):
    try:
        profile_name = config.ACTIVE_PROFILE_NAME
        backlog_file = config.SUMMARIES_DIR / f"{profile_name}.backlog.csv"
        config.SUMMARIES_DIR.mkdir(exist_ok=True)
        file_is_empty = not backlog_file.exists() or backlog_file.stat().st_size == 0
        with open(backlog_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if file_is_empty:
                writer.writerow(['speaker', 'field1', 'field2'])
            writer.writerow([speaker, japanese_text, english_text])
        print(f"Saved new pair to backlog: {backlog_file}")
    except Exception as e:
        print(f"!!! ERROR saving pair to backlog file: {e} !!!")

def clean_and_restructure_prompt(text: str) -> str:
    name_tag_contents = [match.strip() for match in re.findall(r"【(.*?)】", text)]
    dialogue_match = re.search(r"([「『].*?[」』])", text, re.DOTALL)
    if name_tag_contents and len(set(name_tag_contents)) == 1 and dialogue_match:
        unique_name = name_tag_contents[0]; dialogue = dialogue_match.group(1); cleaned_text = f"{unique_name} {dialogue}"
        print(f"Restructured prompt: '{text}' -> '{cleaned_text}'"); return cleaned_text
    return text.strip()

async def update_summary():
    if config.SUMMARY_LOCK.locked():
        print("Summarization already in progress, skipping.")
        return
    async with config.SUMMARY_LOCK:
        if not config.JAPANESE_SOURCE_BACKLOG:
            print("Summarization called but backlog is empty.")
            return
        summary_cfg = config.SETTINGS.get("summarization_settings", {})
        lines_to_summarize = summary_cfg.get("lines_per_summary", 50)
        max_chars = summary_cfg.get("max_summary_chars", 1000)
        prompt_template = summary_cfg.get("prompt", config.DEFAULT_SETTINGS["summarization_settings"]["prompt"])
        recent_japanese_text = list(config.JAPANESE_SOURCE_BACKLOG)[-lines_to_summarize:]
        print(f"Summarization triggered for profile: {config.ACTIVE_PROFILE_NAME}, using last {len(recent_japanese_text)} lines.")
        summary_file = config.SUMMARIES_DIR / f"{config.ACTIVE_PROFILE_NAME}.summary.txt"
        previous_summary = summary_file.read_text(encoding='utf-8') if summary_file.exists() else "No previous summary."
        formatted_japanese_text = "\n".join(recent_japanese_text)
        if "{previous_summary}" in prompt_template and not previous_summary:
            lines = prompt_template.split('\n')
            prompt_template = "\n".join([line for line in lines if "{previous_summary}" not in line])
        summary_prompt = prompt_template.format(
            previous_summary=previous_summary,
            japanese_text=formatted_japanese_text,
            max_summary_chars=max_chars
        )
        summary_body = {"model": "summary/model", "messages": [{"role": "user", "content": summary_prompt}]}
        try:
            url = f"{config.SETTINGS['target_api_base_url'].rstrip('/')}/v1/chat/completions"
            headers = {"Host": "localhost:5001", "User-Agent": "condom-proxy/summarizer"}
            req = config.llm_client.build_request("POST", url, headers=headers, content=json.dumps(summary_body, ensure_ascii=False).encode('utf-8'))
            resp = await config.llm_client.send(req)
            resp.raise_for_status()
            new_summary = resp.json()["choices"][0]["message"]["content"].strip()
            summary_file.write_text(new_summary, encoding='utf-8')
            print(f"New summary for '{config.ACTIVE_PROFILE_NAME}' saved: {new_summary[:100]}...")
            config.LAST_SUMMARY_LINE_COUNT = config.TOTAL_LINE_COUNT
            config.save_counters(config.ACTIVE_PROFILE_NAME)
            print(f"Last summary line count updated to: {config.LAST_SUMMARY_LINE_COUNT}")
        except Exception as e:
            print(f"!!! SUMMARIZATION FAILED for profile {config.ACTIVE_PROFILE_NAME}: {e} !!!")

def consolidate_entities(genders: dict, ner_entities: list, cfg: dict) -> str:
    consolidated = {}
    if cfg.get("use_genders_in_ner"):
        for name, gender in genders.items():
            consolidated[name] = f"- {name} (PERSON, Gender: {gender}, Confidence: High, Source: Custom Regex)"
    if cfg.get("use_ginza"):
        for entity in ner_entities:
            name = entity.get("text")
            if name and name not in consolidated:
                consolidated[name] = f"- {name} ({entity.get('label')}, Confidence: Medium, Source: GiNZA)"
    return "\n".join(consolidated.values())

async def build_enhanced_prompt(body: dict, settings: dict, app_name: str) -> tuple:
    enhanced_body = deepcopy(body)
    messages = enhanced_body.get("messages", [])
    is_dialogue = False
    if not (messages and messages[-1].get("role") == "user"):
        return enhanced_body, "", {}, False, False

    last_user_message = messages[-1]
    history_to_prune = messages[:-1]
    
    pruning_cfg = settings.get("pruning_settings", {})
    if pruning_cfg.get("enabled"):
        prune_mode = pruning_cfg.get("mode")
        system_message = [msg for msg in history_to_prune if msg.get("role") == "system"]
        conversation_messages = [msg for msg in history_to_prune if msg.get("role") != "system"]
        
        pruned_conv = conversation_messages
        if conversation_messages:
            if prune_mode == "keep_n_turns":
                n_turns = pruning_cfg.get("keep_n_turns", 8) * 2
                pruned_conv = conversation_messages[-n_turns:]
            else:
                temp_conv = []
                if prune_mode == "prune_japanese":
                    temp_conv = [msg for msg in conversation_messages if msg.get("role") == "assistant"]
                elif prune_mode == "prune_english":
                    temp_conv = [msg for msg in conversation_messages if msg.get("role") == "user"]
                elif prune_mode == "prune_both":
                    temp_conv = []
                pruned_conv = temp_conv
        
        enhanced_body["messages"] = system_message + pruned_conv
        if len(enhanced_body["messages"]) < len(history_to_prune):
            print(f"Pruned history from {len(messages)} to {len(enhanced_body['messages']) + 1} messages (mode: {prune_mode}).")
    else:
         enhanced_body["messages"] = history_to_prune

    cleaned_prompt_text = last_user_message.get("content", "")
    
    speaker_match = re.match(r"(\w+)", cleaned_prompt_text)
    speaker = speaker_match.group(1) if speaker_match else "Unknown"
    
    is_dialogue_flag = '「' in cleaned_prompt_text or '『' in cleaned_prompt_text
    if not is_dialogue_flag and speaker == "Unknown":
        full_history_text = " ".join(msg.get("content", "") for msg in messages)
        pov_match = re.search(r"(\w+)\s*\([MF]\)\(MC\)", full_history_text)
        if pov_match:
            speaker = pov_match.group(1)
            print(f"Identified speaker for narration line as POV character: {speaker}")

    charinfo_file = config.SUMMARIES_DIR / f"{config.ACTIVE_PROFILE_NAME}.charinfo.json"
    character_profiles = {}
    if charinfo_file.exists():
        with open(charinfo_file, 'r', encoding='utf-8') as f:
            character_profiles = json.load(f)

    character_profile_injection = ""
    if character_profiles:
        found_char_ids = set()
        for char_id, profile in character_profiles.items():
            names_to_check = [profile.get('name', ''), profile.get('original', '')] + profile.get('aliases', [])
            names_to_check = [name for name in names_to_check if name]
            for name in names_to_check:
                if name in cleaned_prompt_text:
                    found_char_ids.add(char_id)
        if found_char_ids:
            injection_parts = ["\n--- Character Profiles Mentioned (CRITICAL) ---"]
            for i, char_id in enumerate(list(found_char_ids)[:3]):
                profile = character_profiles[char_id]
                aliases = [a for a in profile.get('aliases', []) if a]
                part = (
                    f"Name: {profile.get('name', '')} (Original: {profile.get('original', '')}, Aliases: {', '.join(aliases)})\n"
                    f"Traits: {', '.join(profile.get('traits', []))}"
                )
                injection_parts.append(part)
            character_profile_injection = "\n".join(injection_parts) + "\nInstruction: Use these profiles to understand the characters involved in this line.\n"

    opt_toggles = settings.get("optimization_toggles", {})
    full_context_text = " ".join(msg.get("content", "") for msg in messages)
    ner_cfg = opt_toggles.get("add_ner", {})
    ling_cfg = opt_toggles.get("add_linguistics", {})
    grammar_cfg = opt_toggles.get("add_grammar", {})
    needs_ginza = any([ner_cfg.get("enabled") and ner_cfg.get("use_ginza"), ling_cfg.get("enabled"), grammar_cfg.get("enabled")])
    needs_genders = ner_cfg.get("enabled") and ner_cfg.get("use_genders_in_ner")

    tasks = [
        analysis.get_definitions_async(cleaned_prompt_text),
        analysis.analyze_grammar_async(cleaned_prompt_text) if needs_ginza else asyncio.sleep(0, result={}),
        analysis.analyze_genders_async(full_context_text) if needs_genders else asyncio.sleep(0, result={}),
        analysis.translate_with_mtl_async(cleaned_prompt_text) if opt_toggles.get("add_mtl_assist", {}).get("enabled") else asyncio.sleep(0, result=""),
        analysis.analyze_character_voice_async(speaker, messages) if opt_toggles.get("add_char_voice", {}).get("enabled") else asyncio.sleep(0, result={})
    ]
    
    definitions, analyses, genders, mtl, voice = await asyncio.gather(*tasks)
    
    is_onomatopoeia = False
    clean_jp_for_check = re.sub(r"[「」『』……、。！？]", "", cleaned_prompt_text).strip()
    if clean_jp_for_check:
        if not re.search(r'[\u4e00-\u9faf\u30a0-\u30ff]', clean_jp_for_check):
            hiragana_words = re.split(r'[\s　]', clean_jp_for_check)
            if all(len(word) <= 3 for word in hiragana_words if word):
                is_onomatopoeia = True
                print(f"--- Detected onomatopoeia based on character type and length analysis ---")

    ner, linguistics = analyses.get("ner", []), analyses.get("linguistics", {})

    if ner:
        for ent in ner:
            if ent.get("label") == "PERSON":
                config.SCENE_CONTEXT["present_characters"].add(ent.get("text"))

    subject_pov_cfg = opt_toggles.get("add_subject_pov", {})
    subject_pov = analysis.resolve_subject_and_pov(messages, cleaned_prompt_text) if subject_pov_cfg.get("enabled") else {}
    
    appendix = ""
    summary_cfg = settings.get("summarization_settings", {})
    summary_file = config.SUMMARIES_DIR / f"{config.ACTIVE_PROFILE_NAME}.summary.txt"
    if summary_cfg.get("enabled") and summary_file.exists():
        summary = summary_file.read_text(encoding='utf-8')
        if summary:
            appendix += f"--- Story Summary (Long-Term Context) ---\n{summary}\n\n"

    appendix += "This is an automated analysis to assist translation. **You MUST provide an English translation. Do NOT output romaji.**\n"
    if character_profile_injection:
        appendix += character_profile_injection
    if config.SCENE_CONTEXT["present_characters"]:
        appendix += f"\n--- Scene Context ---\nCharacters currently present: {', '.join(sorted(list(config.SCENE_CONTEXT['present_characters'])))}\n"
    if subject_pov_cfg.get("enabled") and subject_pov:
        appendix += "\n--- Subject & POV Analysis (CRITICAL) ---\n"
        if subject_pov_cfg.get("pov_character") and "pov_character" in subject_pov: appendix += f"- POV Character: {subject_pov['pov_character']}\n"
        if subject_pov_cfg.get("likely_subject") and "likely_subject" in subject_pov: appendix += f"- Unspoken SUBJECT is likely: {subject_pov['likely_subject']}\n"
    
    if opt_toggles.get("detect_text_type", {}).get("enabled"):
        if is_dialogue_flag:
            appendix += "Instruction: SPOKEN DIALOGUE... Output only dialogue in quotes.\n"; is_dialogue = True
        else:
            appendix += "Instruction: INTERNAL MONOLOGUE... Do NOT use quotes.\n"
            if speaker != "Unknown":
                appendix += f"INSTRUCTION: Since this is POV narration and the subject is omitted, the subject of this action is {speaker}.\n"
    
    if voice:
        appendix += "\n--- Character Voice Profile (Speaker) ---\n"
        if voice.get("pronoun"): appendix += f"- Uses pronoun: {voice['pronoun']}\n"
        if voice.get("ender"): appendix += f"- Common sentence ender: {voice['ender']}\n"
        appendix += "Instruction: Use this to maintain a consistent character voice.\n"
    if opt_toggles.get("add_mtl_assist", {}).get("enabled") and mtl: appendix += f"\n--- Machine Translation (for reference) ---\n{mtl}\n"
    if ner_cfg.get("enabled"):
        ner_str = consolidate_entities(genders, ner, ner_cfg)
        if ner_str: appendix += f"\n--- Named Entities (with Confidence) ---\n{ner_str}\n"
    if ling_cfg.get("enabled"):
        if (ling_cfg.get("honorifics") and linguistics.get("honorifics")) or (ling_cfg.get("formality") and linguistics.get("formality")):
            appendix += "\n--- Honorifics & Formality ---\n"
            if ling_cfg.get("honorifics") and linguistics.get("honorifics"): appendix += f"Honorifics detected: {', '.join(linguistics['honorifics'])}\n"
            if ling_cfg.get("formality") and linguistics.get("formality"): appendix += f"Sentence Formality: {linguistics['formality']}\n"
    if opt_toggles.get("add_vocabulary", {}).get("enabled") and definitions:
        appendix += "\n--- Vocabulary Appendix ---\n"
        custom_keys = analysis.load_custom_dictionary().keys()
        for word, defi in definitions.items(): appendix += f"- {word}{' (custom)' if word in custom_keys else ''}: {defi}\n"
    if grammar_cfg.get("enabled") and analyses.get("dependency"): appendix += f"\n--- Compact Grammar Analysis ---\n{analyses['dependency']}\n"

    new_words = {w: d for w, d in definitions.items() if w not in analysis.load_custom_dictionary()} if opt_toggles.get("add_vocabulary", {}).get("enabled") and definitions else {}
    
    if len(appendix) > 100:
        enhanced_body["messages"].append({"role": "assistant", "content": appendix})
    
    enhanced_body["messages"].append(last_user_message)
    return enhanced_body, appendix, new_words, is_dialogue, is_onomatopoeia

def parse_sse_chunk(chunk: str) -> str:
    if not chunk.startswith('data:'):
        return ""
    try:
        json_str = chunk[len('data:'):].strip()
        if json_str == "[DONE]":
            return ""
        data = json.loads(json_str)
        return data['choices'][0]['delta'].get('content', '')
    except (json.JSONDecodeError, KeyError, IndexError):
        return ""

async def corrective_llm_pass(original_jp: str, stream: httpx.Response.aiter_raw, is_dialogue: bool, messages: list):
    print("--- Corrective Pass Enabled: Buffering initial translation... ---")
    
    flawed_en_parts = [parse_sse_chunk(chunk.decode('utf-8')) async for chunk in stream]
    flawed_en = "".join(flawed_en_parts)

    name_pattern = re.compile(r"^\s*[\w\s\(\)]+:\s*(?=\")")
    if is_dialogue:
        flawed_en = name_pattern.sub("", flawed_en, count=1)
    flawed_en = flawed_en.strip().strip('"')

    if not flawed_en:
        print("--- Corrective Pass: Initial translation was empty. ---")
        async def empty_stream():
            if False: yield
        return empty_stream()

    print(f"--- Running Corrective LLM Pass on: \"{flawed_en}\" ---")
    
    correction_cfg = config.SETTINGS.get("optimization_toggles", {}).get("use_corrective_pass", {})
    body = {
        "model": correction_cfg.get("model_name", "correction/model"),
        "temperature": correction_cfg.get("temperature", 0.4),
        "max_tokens": 200,
        "stream": True
    }

    if correction_cfg.get("add_context"):
        system_prompt = (
            "You are a master-level Japanese-to-English translator and editor, reviewing the work of a junior translator. "
            "Your task is to correct it, using the conversation history (backlog) provided.\n"
            "CRITICAL: Ensure the subject and object of the sentence are correct based on the context. Do not change the meaning.\n"
            "Focus on fixing: 1. Contextual errors. 2. Unnatural phrasing. 3. Untranslated romaji.\n"
            "Output ONLY the final, corrected English translation and nothing else."
        )
        history = messages[:-1]
        if len(history) > 8: history = history[-8:]
        correction_messages = [{"role": "system", "content": system_prompt}] + history
        correction_messages.append({
            "role": "user",
            "content": f"Review this translation.\nOriginal Japanese: 「{original_jp}」\nFlawed English: \"{flawed_en}\"\nCorrected English:"
        })
        body["messages"] = correction_messages
    else:
        simple_prompt = (
            "You are a proofreading expert. Correct the flawed English translation based on the original Japanese.\n"
            "Output ONLY the corrected English translation.\n\n"
            f"Original Japanese: 「{original_jp}」\n"
            f"Flawed English: \"{flawed_en}\"\n"
            "Corrected English:"
        )
        body["messages"] = [{"role": "user", "content": simple_prompt}]
    
    try:
        url = f"{config.SETTINGS['target_api_base_url'].rstrip('/')}/v1/chat/completions"
        headers = {"Host": "localhost:5001", "User-Agent": "condom-proxy/corrector"}
        req = config.llm_client.build_request("POST", url, headers=headers, content=json.dumps(body, ensure_ascii=False).encode('utf-8'))
        resp = await config.llm_client.send(req, stream=True)
        resp.raise_for_status()
        return resp.aiter_raw()
            
    except Exception as e:
        print(f"!!! CORRECTIVE PASS FAILED: {e}. Yielding original translation. !!!")
        async def fallback_stream():
            yield f"data: {json.dumps({'choices': [{'delta': {'content': flawed_en}}]})}\n\n".encode('utf-8')
        return fallback_stream()

async def final_processing_streamer(stream: httpx.Response.aiter_raw, is_dialogue: bool, original_jp: str, speaker: str):
    name_pattern = re.compile(r"^\s*[\w\s\(\)]+:\s*(?=\")")
    is_first_chunk = True
    full_response_parts = []

    async for chunk in stream:
        chunk_text = parse_sse_chunk(chunk.decode('utf-8'))
        if not chunk_text:
            continue

        if is_first_chunk:
            is_first_chunk = False
            cleaned_text = name_pattern.sub("", chunk_text, count=1) if is_dialogue else chunk_text
            full_response_parts.append(cleaned_text)
            yield f"data: {json.dumps({'choices': [{'delta': {'content': cleaned_text}}]})}\n\n".encode('utf-8')
        else:
            full_response_parts.append(chunk_text)
            yield chunk

    final_text = "".join(full_response_parts).strip().strip('"')

    if final_text:
        config.ENGLISH_TRANSLATION_BACKLOG.append(final_text)
        try:
            last_jp = config.JAPANESE_SOURCE_BACKLOG[-1]
            await asyncio.to_thread(_save_pair_to_backlog_file, speaker, last_jp, final_text)
        except IndexError:
            print("!!! WARNING: Backlogs out of sync, could not save pair. !!!")

async def run_benchmark(request: Request):
    raise HTTPException(status_code=501, detail="Benchmark function not updated for new logic.")

async def proxy_to_local_llm(request: Request, path: str):
    forced_app_name = config.SETTINGS.get("force_app_name", "")
    app_name = forced_app_name if forced_app_name else request.headers.get("X-Application-Name", "default")
    
    req_body_bytes = await request.body()
    is_chat_completion = path.rstrip('/').endswith("v1/chat/completions")

    if not is_chat_completion:
        url = f"{config.SETTINGS['target_api_base_url'].rstrip('/')}/{path.lstrip('/')}"
        req = config.llm_client.build_request(request.method, url, headers=request.headers, content=req_body_bytes)
        resp = await config.llm_client.send(req, stream=True)
        return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, headers=resp.headers)

    is_dialogue, speaker, original_prompt, cleaned_prompt = False, "Unknown", "", ""
    is_onomatopoeia_final = False
    req_body_json = {}
    try:
        req_body_json = json.loads(req_body_bytes)
        messages = req_body_json.get("messages", [])
        if not (isinstance(messages, list) and messages and messages[-1].get("content")):
            raise ValueError("Empty or invalid messages list")

        original_prompt = messages[-1]["content"]
        
        cleaned_messages = []
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content"):
                cleaned_content = clean_and_restructure_prompt(msg["content"])
                cleaned_messages.append({"role": "user", "content": cleaned_content})
            else:
                cleaned_messages.append(msg)
        
        req_body_json["messages"] = cleaned_messages
        cleaned_prompt = cleaned_messages[-1]["content"] if cleaned_messages else ""

        if cleaned_prompt:
            config.JAPANESE_SOURCE_BACKLOG.append(cleaned_prompt.strip())
            config.TOTAL_LINE_COUNT += 1
            config.save_counters(config.ACTIVE_PROFILE_NAME)
            
            summary_cfg = config.SETTINGS.get("summarization_settings", {})
            lines_since = config.TOTAL_LINE_COUNT - config.LAST_SUMMARY_LINE_COUNT
            trigger = summary_cfg.get("lines_per_summary", 50)
            
            if summary_cfg.get("enabled") and lines_since >= trigger:
                print(f"--- AUTO-SUMMARY TRIGGERED: New lines ({lines_since}) >= Trigger ({trigger}) ---")
                try:
                    await asyncio.wait_for(update_summary(), timeout=120.0)
                except asyncio.TimeoutError:
                    print("!!! SUMMARIZATION TIMED OUT !!!")

        enhanced_body, appendix, new_words, is_dialogue, is_onomatopoeia = await build_enhanced_prompt(req_body_json, config.SETTINGS, app_name)
        is_onomatopoeia_final = is_onomatopoeia
        
        if appendix:
            config.LOG_BUFFER.append({"timestamp": datetime.now().isoformat(), "original_prompt": original_prompt, "injected_analysis": appendix, "new_words": new_words})
            print(f"--- Original Final Prompt ---\n{original_prompt}")
            if original_prompt != cleaned_prompt: print(f"--- Cleaned Final Prompt ---\n{cleaned_prompt}")
            print(f"\n--- Injected Assistant Analysis ---\n{appendix}\n------------------------")
            req_body_bytes = json.dumps(enhanced_body, ensure_ascii=False).encode('utf-8')
        
        speaker_match = re.match(r"(\w+)", cleaned_prompt)
        speaker = speaker_match.group(1) if speaker_match else "Unknown"

    except (json.JSONDecodeError, ValueError) as e:
        print(f"--- Could not process request: {e}. Passing original request through. ---")
        pass

    headers = {"Host": "localhost:5001", "User-Agent": "condom-proxy/live", "Accept": "text/event-stream"}
    url = f"{config.SETTINGS['target_api_base_url'].rstrip('/')}/{path.lstrip('/')}"
    try:
        req = config.llm_client.build_request("POST", url, headers=headers, content=req_body_bytes)
        resp = await config.llm_client.send(req, stream=True)

        use_correction = config.SETTINGS.get("optimization_toggles", {}).get("use_corrective_pass", {}).get("enabled")
        
        jp_dialogue_match = re.search(r"[「『](.*?)[」』]", cleaned_prompt)
        jp_dialogue = jp_dialogue_match.group(1) if jp_dialogue_match else cleaned_prompt

        if use_correction and not is_onomatopoeia_final:
            stream_to_process = await corrective_llm_pass(jp_dialogue, resp.aiter_raw(), is_dialogue, req_body_json.get("messages", []))
        else:
            if is_onomatopoeia_final:
                print("--- Onomatopoeia detected, SKIPPING corrective pass. ---")
            stream_to_process = resp.aiter_raw()

        return StreamingResponse(final_processing_streamer(stream_to_process, is_dialogue, original_prompt, speaker), status_code=resp.status_code, media_type="text/event-stream")

    except httpx.ConnectError as e:
        return JSONResponse(content={"error": f"Failed to connect to target API: {e}"}, status_code=502)