# run.py
import json
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from copy import deepcopy
import asyncio

# Import modularized components
import config
import analysis
import vndb
import core_logic

# --- 4. FastAPI Application Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    config.SUMMARIES_DIR.mkdir(exist_ok=True)
    config.load_profiles_from_disk()
    config.load_app_state()
    config.activate_profile(config.ACTIVE_PROFILE_NAME)
    
    analysis.setup_analyzers()
    analysis.load_dictionary()
    analysis.load_ignore_list()
    
    config.vndb_client = httpx.AsyncClient(timeout=30.0)
    
    timeout_cfg = config.SETTINGS.get("timeout_settings", config.DEFAULT_SETTINGS["timeout_settings"])
    timeout = httpx.Timeout(
        connect=timeout_cfg.get("connect", 10.0),
        read=timeout_cfg.get("read", 300.0),
        write=timeout_cfg.get("write", 300.0),
        pool=timeout_cfg.get("pool", 10.0),
    )
    config.llm_client = httpx.AsyncClient(timeout=timeout)
    
    yield
    
    await config.vndb_client.aclose()
    await config.llm_client.aclose()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# --- 5. Web UI and API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/logs", response_class=JSONResponse)
async def get_logs():
    return list(config.LOG_BUFFER)

@app.get("/api/custom-dict", response_class=JSONResponse)
async def get_custom_dict():
    return analysis.load_custom_dictionary()

@app.post("/api/custom-dict")
async def save_custom_dict(request: Request):
    try:
        data = await request.json()
        content = data.get('content')
        json.loads(content)
        with open(config.CUSTOM_DICT_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        return JSONResponse({"status": "success"})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")

# --- Profile Management Endpoints ---
@app.get("/api/settings", response_class=JSONResponse)
async def get_settings():
    full_settings = deepcopy(config.SETTINGS)
    charinfo_file = config.SUMMARIES_DIR / f"{config.ACTIVE_PROFILE_NAME}.charinfo.json"
    if charinfo_file.exists():
        try:
            with open(charinfo_file, 'r', encoding='utf-8') as f:
                full_settings['character_profiles'] = json.load(f)
        except (json.JSONDecodeError, OSError):
            full_settings['character_profiles'] = {}
    else:
        full_settings['character_profiles'] = {}
        
    return full_settings

@app.get("/api/profiles", response_class=JSONResponse)
async def get_profiles():
    return {"profiles": list(config.PROFILES.keys()), "active_profile": config.ACTIVE_PROFILE_NAME}

@app.post("/api/profiles/switch")
async def switch_active_profile(request: Request):
    data = await request.json()
    profile_name = data.get("name")
    if config.activate_profile(profile_name):
        config.save_app_state()
        return JSONResponse({"status": "success", "active_profile": profile_name})
    raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found.")

@app.post("/api/profiles/save")
async def save_active_profile(request: Request):
    new_settings = await request.json()
    
    character_profiles_data = new_settings.pop("character_profiles", None)

    current_profile = config.PROFILES.get(config.ACTIVE_PROFILE_NAME, deepcopy(config.DEFAULT_SETTINGS))
    
    for key, value in new_settings.items():
        if key in current_profile and isinstance(value, dict):
            current_profile[key].update(value)
        else:
            current_profile[key] = value
            
    config.PROFILES[config.ACTIVE_PROFILE_NAME] = current_profile
    config.activate_profile(config.ACTIVE_PROFILE_NAME)
    config.save_profiles_to_disk()
    
    return JSONResponse({"status": "success", "message": f"Profile '{config.ACTIVE_PROFILE_NAME}' saved."})


@app.post("/api/profiles/create")
async def create_new_profile(request: Request):
    data = await request.json()
    profile_name = data.get("name")
    if not profile_name: raise HTTPException(status_code=400, detail="Profile name not provided.")
    if profile_name in config.PROFILES: raise HTTPException(status_code=409, detail="Profile with this name already exists.")
    config.PROFILES[profile_name] = deepcopy(config.DEFAULT_SETTINGS)
    config.save_profiles_to_disk()
    return JSONResponse({"status": "success", "message": f"Profile '{profile_name}' created."})

@app.delete("/api/profiles/{profile_name}")
async def delete_profile(profile_name: str):
    if profile_name == "default": raise HTTPException(status_code=400, detail="Cannot delete default profile.")
    if profile_name not in config.PROFILES: raise HTTPException(status_code=404, detail="Profile not found.")
    del config.PROFILES[profile_name]
    config.save_profiles_to_disk()
    (config.SUMMARIES_DIR / f"{profile_name}.summary.txt").unlink(missing_ok=True)
    (config.SUMMARIES_DIR / f"{profile_name}.charinfo.json").unlink(missing_ok=True)
    (config.SUMMARIES_DIR / f"{profile_name}.backlog.csv").unlink(missing_ok=True)
    (config.SUMMARIES_DIR / f"{profile_name}.counter.json").unlink(missing_ok=True)
    if config.ACTIVE_PROFILE_NAME == profile_name:
        config.activate_profile("default")
        config.save_app_state()
    return JSONResponse({"status": "success", "message": f"Profile '{profile_name}' deleted."})

# --- Feature Endpoints ---
@app.post("/api/profiles/fetch_vndb", response_class=JSONResponse)
async def fetch_vndb_data_endpoint(request: Request):
    return await vndb.fetch_vndb_data(request)

@app.post("/api/summarize", response_class=JSONResponse)
async def trigger_summary_generation():
    if config.SUMMARY_LOCK.locked():
        raise HTTPException(status_code=429, detail="Summarization already in progress.")
    if not config.JAPANESE_SOURCE_BACKLOG:
        return JSONResponse({"status": "no_new_content", "message": "The translation backlog is empty."})
    asyncio.create_task(core_logic.update_summary())
    return JSONResponse({"status": "started", "message": "Summarization process started."})

@app.post("/api/benchmark", response_class=JSONResponse)
async def run_benchmark(request: Request):
    return await core_logic.run_benchmark(request)

# --- Main Proxy Endpoint ---
@app.post("/{path:path}")
async def proxy_to_local_llm(request: Request, path: str):
    return await core_logic.proxy_to_local_llm(request, path)