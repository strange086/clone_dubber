"""
Pydantic models for EduDub AI Platform
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class ProcessingStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EXTRACTING_AUDIO = "extracting_audio"
    SOURCE_SEPARATION = "source_separation"
    SPEECH_RECOGNITION = "speech_recognition"
    SPEAKER_DIARIZATION = "speaker_diarization"
    EMOTION_DETECTION = "emotion_detection"
    VOICE_CLONING = "voice_cloning"
    TRANSLATION = "translation"
    TEXT_TO_SPEECH = "text_to_speech"
    PROSODY_MATCHING = "prosody_matching"
    AUDIO_MIXING = "audio_mixing"
    LIP_SYNC = "lip_sync"
    FINAL_VIDEO = "final_video"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SpeakerInfo(BaseModel):
    """Information about detected speakers"""
    speaker_id: str
    gender: str = Field(..., description="Detected gender: male/female/unknown")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    confidence: float = Field(..., description="Confidence score 0-1")
    voice_embedding: Optional[List[float]] = Field(None, description="Voice embedding vector")
    emotion: Optional[str] = Field(None, description="Detected emotion")

class VoiceProfile(BaseModel):
    """Voice profile for cloning"""
    profile_id: str
    speaker_name: Optional[str] = None
    gender: str
    language: str
    voice_embedding: List[float] = Field(..., description="Speaker embedding vector")
    reference_audio_path: str = Field(..., description="Path to reference audio sample")
    sample_rate: int = 22050
    duration: float = Field(..., description="Duration of reference audio in seconds")
    quality_score: float = Field(..., description="Quality score of voice sample 0-1")
    created_at: datetime = Field(default_factory=datetime.now)

class DubbingRequest(BaseModel):
    """Request model for dubbing process"""
    input_file: str = Field(..., description="Path to input video file")
    target_language: str = Field(..., description="Target language code")
    source_language: str = Field(default="auto", description="Source language code")
    preserve_emotion: bool = Field(default=True, description="Preserve emotional tone")
    enable_lipsync: bool = Field(default=True, description="Enable lip synchronization")
    use_voice_cloning: bool = Field(default=True, description="Use voice cloning for speaker preservation")
    custom_voice_profile: Optional[str] = Field(None, description="Custom voice profile ID to use")
    voice_similarity_threshold: float = Field(default=0.75, description="Minimum similarity for voice matching")

class DubbingResponse(BaseModel):
    """Response model for dubbing process"""
    job_id: str
    status: ProcessingStatus
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    message: str
    current_step: Optional[str] = None
    speakers_detected: Optional[List[SpeakerInfo]] = None
    voice_profiles_created: Optional[List[str]] = None
    translation_quality: Optional[Dict[str, float]] = None
    output_path: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

class VoiceCloneRequest(BaseModel):
    """Request for voice cloning"""
    audio_file: str = Field(..., description="Path to audio file for voice cloning")
    speaker_name: Optional[str] = None
    target_language: str = Field(..., description="Target language for cloning")
    min_duration: float = Field(default=10.0, description="Minimum audio duration required")
    quality_threshold: float = Field(default=0.7, description="Minimum quality threshold")

class VoiceCloneResponse(BaseModel):
    """Response for voice cloning"""
    profile_id: str
    status: str
    quality_score: float
    duration: float
    message: str
    voice_embedding_size: int
    supported_languages: List[str]

class EmotionAnalysis(BaseModel):
    """Emotion analysis result"""
    emotion: str = Field(..., description="Detected emotion")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    intensity: float = Field(..., ge=0, le=1, description="Emotion intensity")

class TranslationQuality(BaseModel):
    """Translation quality metrics"""
    bleu_score: float = Field(..., ge=0, le=100, description="BLEU score")
    confidence: float = Field(..., ge=0, le=1, description="Translation confidence")
    source_text: str
    translated_text: str
    language_pair: str = Field(..., description="source-target language pair")

class AudioSegment(BaseModel):
    """Audio segment information"""
    segment_id: str
    start_time: float
    end_time: float
    speaker_id: str
    text: str
    translated_text: Optional[str] = None
    emotion: Optional[str] = None
    confidence: float
    audio_path: Optional[str] = None