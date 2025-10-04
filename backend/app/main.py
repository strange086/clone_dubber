"""
Main FastAPI application for EduDub AI Platform
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import aiofiles
import uuid
import os
from pathlib import Path
from typing import Optional, List
import logging

from .config import settings
from .models import DubbingRequest, DubbingResponse, ProcessingStatus
from .dubbing_pipeline import DubbingPipeline
from .utils import validate_file_format, get_file_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI-powered dubbing system for Indian languages",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")

# Initialize dubbing pipeline
dubbing_pipeline = DubbingPipeline()

# In-memory storage for job status (use Redis in production)
job_status = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    await dubbing_pipeline.initialize()

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application")
    await dubbing_pipeline.cleanup()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.version,
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "version": settings.version,
        "supported_languages": settings.supported_languages,
        "models_loaded": await dubbing_pipeline.get_model_status()
    }

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "languages": settings.supported_languages,
        "total_count": len(settings.supported_languages)
    }

@app.post("/upload", response_model=dict)
async def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: str = Form(default="auto"),
    preserve_emotion: bool = Form(default=True),
    enable_lipsync: bool = Form(default=True)
):
    """Upload video file and start dubbing process"""
    
    # Validate file format
    if not validate_file_format(file.filename, settings.supported_video_formats):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {settings.supported_video_formats}"
        )
    
    # Validate target language
    if target_language not in settings.supported_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language. Supported languages: {list(settings.supported_languages.keys())}"
        )
    
    # Check file size
    if file.size > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size // (1024*1024)}MB"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = settings.upload_dir / f"{job_id}_{file.filename}"
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Initialize job status
        job_status[job_id] = {
            "status": "uploaded",
            "progress": 0,
            "message": "File uploaded successfully",
            "original_filename": file.filename,
            "file_path": str(file_path),
            "target_language": target_language,
            "source_language": source_language,
            "preserve_emotion": preserve_emotion,
            "enable_lipsync": enable_lipsync,
            "created_at": str(Path(file_path).stat().st_mtime)
        }
        
        logger.info(f"File uploaded successfully: {job_id}")
        
        return {
            "job_id": job_id,
            "message": "File uploaded successfully",
            "filename": file.filename,
            "size": file.size,
            "target_language": settings.supported_languages[target_language]
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process/{job_id}")
async def start_dubbing_process(
    job_id: str,
    background_tasks: BackgroundTasks
):
    """Start the dubbing process for uploaded video"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_status[job_id]
    
    if job_info["status"] != "uploaded":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is in {job_info['status']} state, cannot start processing"
        )
    
    # Update status to processing
    job_status[job_id]["status"] = "processing"
    job_status[job_id]["progress"] = 5
    job_status[job_id]["message"] = "Starting dubbing pipeline..."
    
    # Start background processing
    background_tasks.add_task(process_dubbing_job, job_id)
    
    return {
        "job_id": job_id,
        "message": "Dubbing process started",
        "status": "processing"
    }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing status of a job"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download the dubbed video"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_status[job_id]
    
    if job_info["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is in {job_info['status']} state, download not available"
        )
    
    output_path = job_info.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_path,
        filename=f"dubbed_{job_info['original_filename']}",
        media_type="video/mp4"
    )

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated files"""
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_status[job_id]
    
    # Delete files
    try:
        file_path = Path(job_info["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        output_path = job_info.get("output_path")
        if output_path and Path(output_path).exists():
            Path(output_path).unlink()
            
        # Remove from job status
        del job_status[job_id]
        
        return {"message": "Job deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting job: {str(e)}")

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "jobs": job_status,
        "total_count": len(job_status)
    }

async def process_dubbing_job(job_id: str):
    """Background task to process dubbing job"""
    
    try:
        job_info = job_status[job_id]
        file_path = job_info["file_path"]
        
        # Create dubbing request
        request = DubbingRequest(
            input_file=file_path,
            target_language=job_info["target_language"],
            source_language=job_info["source_language"],
            preserve_emotion=job_info["preserve_emotion"],
            enable_lipsync=job_info["enable_lipsync"]
        )
        
        # Process through pipeline
        async for update in dubbing_pipeline.process(request, job_id):
            if job_id in job_status:  # Check if job still exists
                job_status[job_id].update(update)
        
        logger.info(f"Dubbing completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        if job_id in job_status:
            job_status[job_id].update({
                "status": "failed",
                "message": f"Processing failed: {str(e)}",
                "error": str(e)
            })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )