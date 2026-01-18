#!/usr/bin/env python3
"""
SAM3 Segmentation Server
Single prompt controls all segmentation requests.

Usage:
    conda activate sam3
    python sam3_server.py
"""

import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

# ============== GLOBAL STATE ==============
model = None
processor = None
CURRENT_PROMPT = "object"
# ==========================================


class SegmentRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None
    confidence_threshold: float = 0.3
    return_visualization: bool = False


class SegmentResponse(BaseModel):
    success: bool
    prompt: str
    num_objects: int
    masks_base64: list[str]
    boxes: list[list[float]]
    scores: list[float]
    inference_time_ms: float
    visualization_base64: Optional[str] = None
    error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    
    print("=" * 60)
    print("Loading SAM3 model...")
    print("=" * 60)
    
    start = time.time()
    model = build_sam3_image_model()
    model = model.cuda().eval()
    processor = Sam3Processor(model)
    
    print(f"✓ SAM3 loaded in {time.time() - start:.1f}s")
    print(f"✓ Device: cuda:0")
    print(f"✓ Default prompt: '{CURRENT_PROMPT}'")
    print(f"✓ Server ready at http://0.0.0.0:8100")
    print("=" * 60)
    
    yield
    
    del model, processor
    torch.cuda.empty_cache()


app = FastAPI(title="SAM3 Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def decode_image(base64_str: str) -> Image.Image:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")


def encode_mask(mask: np.ndarray) -> str:
    while mask.ndim > 2:
        mask = mask[0]
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    img = Image.fromarray(mask, mode='L')
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============== PROMPT ENDPOINTS ==============

@app.get("/prompt")
async def get_prompt():
    return {"prompt": CURRENT_PROMPT}


@app.post("/prompt/{new_prompt:path}")
async def set_prompt(new_prompt: str):
    global CURRENT_PROMPT
    CURRENT_PROMPT = new_prompt
    print(f"✓ Prompt changed to: '{CURRENT_PROMPT}'")
    return {"prompt": CURRENT_PROMPT}


# ============== HEALTH ==============

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "prompt": CURRENT_PROMPT,
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ============== SEGMENTATION ==============

@app.post("/segment", response_model=SegmentResponse)
async def segment_image(request: SegmentRequest):
    global CURRENT_PROMPT
    
    prompt = request.prompt if request.prompt else CURRENT_PROMPT
    
    try:
        start_time = time.time()
        
        image = decode_image(request.image_base64)
        
        # Use autocast for faster FP16 inference
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        inference_time = (time.time() - start_time) * 1000
        
        if masks is None or len(masks) == 0:
            return SegmentResponse(
                success=True, prompt=prompt, num_objects=0,
                masks_base64=[], boxes=[], scores=[],
                inference_time_ms=inference_time,
            )
        
        masks_np = masks.cpu().numpy()
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        keep = scores_np >= request.confidence_threshold
        masks_np, boxes_np, scores_np = masks_np[keep], boxes_np[keep], scores_np[keep]
        
        encoded_masks = []
        for m in masks_np:
            try:
                encoded_masks.append(encode_mask(m))
            except:
                pass
        
        return SegmentResponse(
            success=True,
            prompt=prompt,
            num_objects=len(masks_np),
            masks_base64=encoded_masks,
            boxes=boxes_np.tolist(),
            scores=scores_np.tolist(),
            inference_time_ms=inference_time,
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return SegmentResponse(
            success=False, prompt=prompt, num_objects=0,
            masks_base64=[], boxes=[], scores=[],
            inference_time_ms=0, error=str(e),
        )


if __name__ == "__main__":
    uvicorn.run("sam3_server:app", host="0.0.0.0", port=8100, reload=False, workers=1)