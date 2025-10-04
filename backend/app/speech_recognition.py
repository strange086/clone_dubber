"""
Speech Recognition System using Whisper
"""
import whisper
import torch
import logging
from typing import Dict, List, Optional
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)

class SpeechRecognitionSystem:
    """Speech recognition using OpenAI Whisper"""
    
    def __init__(self):
        self.model = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize Whisper model"""
        try:
            logger.info("Loading Whisper model...")
            self.model = whisper.load_model(
                settings.whisper_model,
                device=settings.whisper_device
            )
            self.initialized = True
            logger.info("Speech recognition system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize speech recognition: {str(e)}")
            raise
    
    async def transcribe(self, audio_path: str, language: str = "auto") -> Dict:
        """Transcribe audio to text"""
        try:
            result = self.model.transcribe(
                audio_path,
                language=None if language == "auto" else language,
                word_timestamps=True
            )
            
            return {
                "text": result["text"],
                "language": result["language"],
                "segments": result["segments"],
                "confidence": 0.9  # Whisper doesn't provide confidence scores
            }
        except Exception as e:
            logger.error(f"Error in speech recognition: {str(e)}")
            raise