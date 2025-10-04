"""
Configuration settings for EduDub AI Platform
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Configuration
    app_name: str = "EduDub AI - Indian Language Dubbing Platform"
    version: str = "1.0.0"
    debug: bool = True
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # File Storage
    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("outputs")
    temp_dir: Path = Path("temp")
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    
    # Supported formats
    supported_video_formats: list = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    supported_audio_formats: list = [".wav", ".mp3", ".m4a", ".flac"]
    
    # AI Model Configuration
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"  # or "cpu"
    
    # Indian Languages Support
    supported_languages: dict = {
        "hi": "Hindi",
        "bn": "Bengali", 
        "te": "Telugu",
        "mr": "Marathi",
        "ta": "Tamil",
        "gu": "Gujarati",
        "kn": "Kannada",
        "ml": "Malayalam",
        "pa": "Punjabi",
        "or": "Odia",
        "as": "Assamese",
        "ur": "Urdu",
        "en": "English"
    }
    
    # TTS Configuration
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_device: str = "cuda"
    
    # Processing Configuration
    sample_rate: int = 22050
    chunk_duration: int = 30  # seconds
    
    # Quality Thresholds (from PRD)
    target_asr_accuracy: float = 0.88
    target_bleu_score: float = 42.0
    target_voice_mos: float = 4.2
    target_lipsync_accuracy: float = 0.92
    
    # API Keys (optional, for external services)
    openai_api_key: str = ""
    google_translate_api_key: str = ""
    elevenlabs_api_key: str = ""
    bhashini_api_key: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Ensure directories exist
settings.upload_dir.mkdir(exist_ok=True)
settings.output_dir.mkdir(exist_ok=True) 
settings.temp_dir.mkdir(exist_ok=True)