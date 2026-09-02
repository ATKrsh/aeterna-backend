"""
server.py — Aeterna Avatar Generation API
FastAPI server that accepts photo uploads and returns a realistic textured 3D face GLB.
"""

import os
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from face_avatar import generate_avatar_glb

app = FastAPI(title="Aeterna Avatar API", version="1.0.0")

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Serve generated GLB files as static
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Aeterna Avatar API"}


@app.post("/generate-avatar")
async def generate_avatar(photos: List[UploadFile] = File(...)):
    """
    Accept 1–10 face photos and return a textured 3D GLB avatar URL.
    
    Returns:
        { "avatarUrl": "http://localhost:8000/output/{uuid}.glb", "id": "{uuid}" }
    """
    if not photos:
        raise HTTPException(status_code=400, detail="No photos uploaded.")
    if len(photos) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 photos allowed.")

    # Read all photo bytes
    photo_bytes_list = []
    for photo in photos:
        content_type = photo.content_type or ""
        if not (content_type.startswith("image/") or
                photo.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {photo.filename}. Only images are accepted."
            )
        raw = await photo.read()
        if len(raw) == 0:
            continue
        photo_bytes_list.append(raw)

    if not photo_bytes_list:
        raise HTTPException(status_code=400, detail="All uploaded files were empty or invalid.")

    avatar_id = str(uuid.uuid4())

    try:
        glb_path = generate_avatar_glb(
            photo_files=photo_bytes_list,
            output_dir=OUTPUT_DIR,
            avatar_id=avatar_id
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Avatar generation failed: {str(e)}")

    return JSONResponse({
        "avatarUrl": f"http://localhost:8000/output/{avatar_id}.glb",
        "id": avatar_id,
        "message": "Avatar generated successfully."
    })


@app.get("/output/{filename}")
async def serve_avatar(filename: str):
    """Explicit route to serve GLB files with correct headers."""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Avatar not found.")
    return FileResponse(
        path=str(filepath),
        media_type="model/gltf-binary",
        headers={"Access-Control-Allow-Origin": "*"}
    )


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Aeterna Avatar Generation API")
    print("  Running at: http://localhost:8000")
    print("  Upload photos to: POST /generate-avatar")
    print("=" * 60)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
