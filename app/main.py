import json, os, secrets, shutil, sys, threading
from pathlib import Path
from typing import List, Literal
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from . import georef

DATA = Path.home() / "Documents" / "Georef"; DATA.mkdir(parents=True, exist_ok=True)
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent.parent))
STATIC = BASE / "static"

app = FastAPI()
app.mount("/data", StaticFiles(directory=DATA), name="data")

class GCP(BaseModel):
    px: float; py: float; lon: float; lat: float
class ProcessReq(BaseModel):
    gcps: List[GCP]
    method: Literal["affine", "poly2", "tps"] = "affine"

def mdir(mid):
    if not mid.isalnum(): raise HTTPException(400, "id invalide")
    d = DATA / mid
    if not d.exists(): raise HTTPException(404, "introuvable")
    return d
def meta(d): return json.loads((d / "meta.json").read_text())
def save(d, **kw):
    m = meta(d) if (d / "meta.json").exists() else {}
    m.update(kw); (d / "meta.json").write_text(json.dumps(m))

@app.post("/api/maps")
async def upload(file: UploadFile = File(...)):
    mid = secrets.token_urlsafe(8).replace("-", "a").replace("_", "b")
    d = DATA / mid; d.mkdir()
    raw = d / "raw"
    raw.write_bytes(await file.read())
    try:
        with Image.open(raw) as im:
            w, h = im.size
            im.convert("RGB").save(d / "source.jpg", quality=95)
    except Exception as e:
        shutil.rmtree(d); raise HTTPException(400, f"image invalide : {e}")
    raw.unlink()
    save(d, status="uploaded", name=Path(file.filename).stem, width=w, height=h)
    return {"id": mid, "width": w, "height": h, "image": f"/data/{mid}/source.jpg"}

def pipeline(d, req):
    try:
        g = [x.model_dump() for x in req.gcps]
        errs = georef.rmse(g, req.method)
        save(d, status="processing", progress=0, errors=errs)
        shutil.rmtree(d / "tiles", ignore_errors=True)
        b, zmin, zmax = georef.make_tiles(d / "source.jpg", g, req.method, d / "tiles",
                                          lambda f: save(d, progress=round(f, 3)))
        georef.to_mbtiles(d / "tiles", d / "carte.mbtiles", b, zmin, zmax, meta(d)["name"])
        save(d, status="done", bounds=b, zmin=zmin, zmax=zmax)
    except Exception as e:
        save(d, status="error", error=str(e))

@app.post("/api/maps/{mid}/process")
def process(mid: str, req: ProcessReq):
    d = mdir(mid)
    need = georef.MIN_PTS[req.method]
    if len(req.gcps) < need: raise HTTPException(400, f"{need} points minimum pour cette méthode")
    save(d, status="queued")
    threading.Thread(target=pipeline, args=(d, req), daemon=True).start()
    return {"status": "queued"}

@app.get("/api/maps/{mid}")
def status(mid: str): return meta(mdir(mid))

@app.post("/api/maps/{mid}/open")
def open_folder(mid: str):
    d = mdir(mid)
    if sys.platform == "win32": os.startfile(d)
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home(): return (STATIC / "index.html").read_text(encoding="utf-8")
