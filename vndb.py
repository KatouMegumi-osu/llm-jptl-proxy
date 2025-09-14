# vndb.py
import json
import re
import httpx
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

import config

VNDB_TRAIT_DB = {}

# <-- FIX: Lazy-load the trait DB only when it's needed -->
async def get_trait_db():
    if not VNDB_TRAIT_DB:
        print("VNDB trait database is empty. Fetching now...")
        page = 1
        while True:
            payload = {"fields": "id, name, parents", "results": 100, "page": page}
            try:
                resp = await config.vndb_client.post("https://api.vndb.org/kana/trait", json=payload)
                resp.raise_for_status()
                data = resp.json()
                for trait in data['results']:
                    VNDB_TRAIT_DB[trait['id']] = {'name': trait['name'], 'parents': trait.get('parents', [])}
                if not data.get('more'):
                    break
                page += 1
            except Exception as e:
                print(f"!!! FAILED to fetch page {page} of VNDB traits: {e} !!!")
                break
        print(f"VNDB trait database loaded with {len(VNDB_TRAIT_DB)} traits.")
    return VNDB_TRAIT_DB

async def fetch_vndb_data(request: Request):
    data = await request.json()
    vndb_id_raw = data.get("vndb_id")
    if not vndb_id_raw:
        raise HTTPException(status_code=400, detail="VNDB ID not provided.")
    
    vndb_id = vndb_id_raw.lower().strip()
    if not vndb_id.startswith('v'):
        vndb_id = 'v' + vndb_id
        
    try:
        trait_db = await get_trait_db()
        if not trait_db:
            raise HTTPException(status_code=503, detail="VNDB trait database could not be fetched. Please try again later.")

        char_payload = {
            "filters": ["vn", "=", ["id", "=", vndb_id]],
            "fields": "name, original, aliases, traits.id, traits.spoiler",
            "results": 100
        }
        char_resp = await config.vndb_client.post("https://api.vndb.org/kana/character", json=char_payload)
        char_resp.raise_for_status()
        characters = char_resp.json().get('results', [])

        if not characters:
            return JSONResponse({"status": "success", "message": "VN ID is valid, but no characters were found."})

        character_profiles = {}
        for char in characters:
            personality_traits = []
            for trait in char.get('traits', []):
                trait_id = trait['id']
                if trait_id in trait_db and trait.get('spoiler', 1) == 0:
                    trait_info = trait_db[trait_id]
                    is_personality = any(p_id in config.PERSONALITY_TRAIT_PARENT_IDS for p_id in trait_info.get('parents', []))
                    if is_personality:
                        traits.append(trait_info['name'])
            
            character_profiles[char['id']] = {
                "name": char.get('name', ''),
                "original": char.get('original', ''),
                "aliases": char.get('aliases', []),
                "traits": sorted(list(set(personality_traits)))
            }
        
        charinfo_file = config.SUMMARIES_DIR / f"{config.ACTIVE_PROFILE_NAME}.charinfo.json"
        with open(charinfo_file, 'w', encoding='utf-8') as f:
            json.dump(character_profiles, f, indent=2, ensure_ascii=False)
            
        return JSONResponse({"status": "success", "message": f"Saved {len(character_profiles)} character profiles to {charinfo_file}."})
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"VNDB API Error: {e.response.text}")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse key '{e}' from VNDB response.")