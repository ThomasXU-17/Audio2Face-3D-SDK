# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Audio2Face Web API Backend

This FastAPI application provides a web interface for the Audio2Face SDK.
It allows users to:
1. Upload audio files
2. Select inference models (regression/diffusion)
3. Run inference and download results
"""

import os
import json
import uuid
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "_data"
BUILD_DIR = PROJECT_ROOT / "_build" / "release"
WEB_INFERENCE_EXECUTABLE = BUILD_DIR / "audio2face-sdk" / "bin" / "a2f-web-inference"
SAMPLE_EXECUTABLE = BUILD_DIR / "audio2face-sdk" / "bin" / "sample-a2f-executor"
UPLOADS_DIR = PROJECT_ROOT / "webui" / "uploads"
RESULTS_DIR = PROJECT_ROOT / "webui" / "results"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Available models configuration
AVAILABLE_MODELS = {
    "mark": {
        "name": "Mark (Regression v2.3)",
        "type": "regression",
        "model_path": str(DATA_DIR / "generated" / "audio2face-sdk" / "samples" / "data" / "mark" / "model.json"),
        "description": "Regression-based model for Mark character"
    },
    "claire": {
        "name": "Claire (Regression v2.3.1)",
        "type": "regression", 
        "model_path": str(DATA_DIR / "generated" / "audio2face-sdk" / "samples" / "data" / "claire" / "model.json"),
        "description": "Regression-based model for Claire character"
    },
    "james": {
        "name": "James (Regression v2.3.1)",
        "type": "regression",
        "model_path": str(DATA_DIR / "generated" / "audio2face-sdk" / "samples" / "data" / "james" / "model.json"),
        "description": "Regression-based model for James character"
    },
    "multi-diffusion": {
        "name": "Multi-Diffusion (v3.0)",
        "type": "diffusion",
        "model_path": str(DATA_DIR / "generated" / "audio2face-sdk" / "samples" / "data" / "multi-diffusion" / "model.json"),
        "description": "Diffusion-based model with multiple identities"
    }
}

app = FastAPI(
    title="Audio2Face Web API",
    description="Web API for Audio2Face SDK - Convert audio to facial animation",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ModelInfo(BaseModel):
    id: str
    name: str
    type: str
    description: str
    available: bool

class InferenceRequest(BaseModel):
    model_id: str
    audio_file_id: str

class InferenceResult(BaseModel):
    job_id: str
    status: str
    message: str
    result_file: Optional[str] = None
    frames_processed: Optional[int] = None

# Job storage (in production, use a proper database)
jobs = {}


def check_model_availability(model_id: str) -> bool:
    """Check if a model's files exist."""
    if model_id not in AVAILABLE_MODELS:
        return False
    model_path = Path(AVAILABLE_MODELS[model_id]["model_path"])
    return model_path.exists()


def convert_audio_to_16k(input_path: Path, output_path: Path) -> bool:
    """Convert audio file to 16kHz mono WAV format required by Audio2Face."""
    try:
        # Try using ffmpeg for conversion
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            str(output_path)
        ], capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except FileNotFoundError:
        # ffmpeg not found, try using scipy/pydub
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(input_path))
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(str(output_path), format="wav")
            return True
        except Exception as e:
            print(f"Audio conversion error: {e}")
            return False
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return False


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Audio2Face Web API",
        "version": "1.0.0",
        "endpoints": {
            "models": "/api/models",
            "upload": "/api/upload",
            "inference": "/api/inference",
            "results": "/api/results/{job_id}"
        }
    }


@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
    models = []
    for model_id, model_info in AVAILABLE_MODELS.items():
        models.append(ModelInfo(
            id=model_id,
            name=model_info["name"],
            type=model_info["type"],
            description=model_info["description"],
            available=check_model_availability(model_id)
        ))
    return models


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload an audio file for processing."""
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    original_path = UPLOADS_DIR / f"{file_id}_original{file_ext}"
    converted_path = UPLOADS_DIR / f"{file_id}.wav"
    
    # Save uploaded file
    try:
        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Convert to 16kHz WAV if needed
    if file_ext != ".wav":
        success = convert_audio_to_16k(original_path, converted_path)
        if not success:
            original_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500, 
                detail="Failed to convert audio file. Please upload a 16kHz WAV file."
            )
        # Remove original file after conversion
        original_path.unlink(missing_ok=True)
    else:
        # For WAV files, still ensure 16kHz format
        temp_path = UPLOADS_DIR / f"{file_id}_temp.wav"
        success = convert_audio_to_16k(original_path, temp_path)
        if success:
            original_path.unlink(missing_ok=True)
            temp_path.rename(converted_path)
        else:
            # Just use the original if conversion fails
            original_path.rename(converted_path)
    
    return {
        "file_id": file_id,
        "filename": file.filename,
        "message": "File uploaded successfully"
    }


def run_inference_task(job_id: str, model_id: str, audio_path: Path, result_path: Path):
    """Background task to run inference using the C++ executable."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Initializing inference..."
        
        # Check if the C++ executable exists
        if not WEB_INFERENCE_EXECUTABLE.exists():
            # Fall back to mock inference if executable not found
            jobs[job_id]["message"] = "C++ executable not found, using mock inference..."
            run_mock_inference(job_id, model_id, audio_path, result_path)
            return
        
        jobs[job_id]["message"] = "Running Audio2Face inference..."
        
        # Build command to run the C++ inference executable
        cmd = [
            str(WEB_INFERENCE_EXECUTABLE),
            "--model", model_id,
            "--audio", str(audio_path),
            "--output", str(result_path),
            "--data-dir", str(DATA_DIR),
            "--fps", "60"
        ]
        
        # Set up environment with library paths
        env = os.environ.copy()
        
        # Add library paths for CUDA and TensorRT
        tensorrt_lib = os.environ.get("TENSORRT_ROOT_DIR", "")
        if tensorrt_lib:
            tensorrt_lib = os.path.join(tensorrt_lib, "lib")
        
        cuda_lib = os.environ.get("CUDA_PATH", "/usr/local/cuda")
        if cuda_lib:
            cuda_lib = os.path.join(cuda_lib, "lib64")
        
        # Build library path
        ld_library_path = env.get("LD_LIBRARY_PATH", "")
        additional_paths = [
            str(BUILD_DIR / "audio2x-sdk" / "lib"),
            str(BUILD_DIR / "audio2face-sdk" / "lib"),
            str(BUILD_DIR / "audio2emotion-sdk" / "lib"),
        ]
        if tensorrt_lib:
            additional_paths.append(tensorrt_lib)
        if cuda_lib:
            additional_paths.append(cuda_lib)
        
        env["LD_LIBRARY_PATH"] = ":".join(additional_paths + [ld_library_path])
        
        # Execute the C++ inference program
        print(f"Running inference command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for long audio files
            env=env,
            cwd=str(PROJECT_ROOT)
        )
        
        # Log stderr for debugging (C++ program logs to stderr)
        if result.stderr:
            print(f"Inference stderr: {result.stderr}")
        
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = f"Inference failed: {error_msg}"
            return
        
        # Check if result file was created
        if not result_path.exists():
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = "Inference completed but no result file was created"
            return
        
        # Read result file to get frame count
        try:
            with open(result_path, "r") as f:
                result_data = json.load(f)
            # Support both new format (numFrames) and legacy format (total_frames)
            frames_processed = result_data.get("numFrames", result_data.get("total_frames", 0))
        except Exception as e:
            frames_processed = 0
            print(f"Error reading result file: {e}")
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Inference completed successfully"
        jobs[job_id]["result_file"] = str(result_path)
        jobs[job_id]["frames_processed"] = frames_processed
        
    except subprocess.TimeoutExpired:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = "Inference timed out (exceeded 10 minutes)"
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Error: {str(e)}"


def run_mock_inference(job_id: str, model_id: str, audio_path: Path, result_path: Path):
    """Fallback mock inference when C++ executable is not available."""
    import wave
    
    try:
        model_info = AVAILABLE_MODELS[model_id]
        model_type = model_info["type"]
        
        # Calculate approximate number of frames based on audio duration
        with wave.open(str(audio_path), "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            duration = frames / float(rate)
            num_frames = int(duration * 60)  # 60 FPS
        
        # 52 FACS names matching Apple ARKit format
        facs_names = [
            "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
            "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight",
            "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
            "jawForward", "jawLeft", "jawRight", "jawOpen",
            "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
            "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
            "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
            "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
            "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
            "mouthUpperUpLeft", "mouthUpperUpRight",
            "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
            "noseSneerLeft", "noseSneerRight", "tongueOut"
        ]
        
        # Generate mock result in a2f_export format
        result = {
            "exportFps": 60.0,
            "trackPath": str(audio_path),
            "numPoses": len(facs_names),
            "numFrames": num_frames,
            "facsNames": facs_names,
            "weightMat": [],
            "joints": ["jaw", "eye_L", "eye_R"],
            "rotations": [],
            "translations": []
        }
        
        # Generate mock frame data (blendshape weights)
        np.random.seed(42)
        for i in range(num_frames):
            # Generate random blendshape weights (mostly zeros with some small values)
            weights = [max(0.0, float(np.random.random() * 0.3 - 0.1)) for _ in range(len(facs_names))]
            result["weightMat"].append(weights)
        
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Mock inference completed (C++ executable not available)"
        jobs[job_id]["result_file"] = str(result_path)
        jobs[job_id]["frames_processed"] = num_frames
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Mock inference error: {str(e)}"



@app.post("/api/inference", response_model=InferenceResult)
async def start_inference(
    background_tasks: BackgroundTasks,
    model_id: str = Form(...),
    audio_file_id: str = Form(...)
):
    """Start inference on an uploaded audio file."""
    # Validate model
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    
    if not check_model_availability(model_id):
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model_id}' is not available. Please run gen_testdata.sh first."
        )
    
    # Validate audio file
    audio_path = UPLOADS_DIR / f"{audio_file_id}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Create job
    job_id = str(uuid.uuid4())
    result_path = RESULTS_DIR / f"{job_id}_result.json"
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": "Job queued for processing",
        "model_id": model_id,
        "audio_file_id": audio_file_id,
        "created_at": datetime.now().isoformat(),
        "result_file": None,
        "frames_processed": None
    }
    
    # Start background task
    background_tasks.add_task(
        run_inference_task, 
        job_id, 
        model_id, 
        audio_path, 
        result_path
    )
    
    return InferenceResult(
        job_id=job_id,
        status="queued",
        message="Inference job started"
    )


@app.get("/api/results/{job_id}", response_model=InferenceResult)
async def get_result(job_id: str):
    """Get the status/result of an inference job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return InferenceResult(
        job_id=job_id,
        status=job["status"],
        message=job["message"],
        result_file=job.get("result_file"),
        frames_processed=job.get("frames_processed")
    )


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """Download the result file of a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    result_path = Path(job["result_file"])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        path=result_path,
        filename=f"audio2face_result_{job_id[:8]}.json",
        media_type="application/json"
    )


@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up job files after download."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Remove result file
    if job.get("result_file"):
        Path(job["result_file"]).unlink(missing_ok=True)
    
    # Remove audio file
    audio_path = UPLOADS_DIR / f"{job['audio_file_id']}.wav"
    audio_path.unlink(missing_ok=True)
    
    # Remove job record
    del jobs[job_id]
    
    return {"message": "Job cleaned up successfully"}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "web_inference_available": WEB_INFERENCE_EXECUTABLE.exists(),
        "sample_executable_available": SAMPLE_EXECUTABLE.exists(),
        "models_available": sum(1 for m in AVAILABLE_MODELS if check_model_availability(m)),
        "inference_mode": "real" if WEB_INFERENCE_EXECUTABLE.exists() else "mock"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
