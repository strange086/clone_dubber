"""
Main Dubbing Pipeline for EduDub AI Platform
Orchestrates the complete dubbing workflow with voice cloning
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator
import uuid
import json

from .config import settings
from .models import DubbingRequest, ProcessingStatus, SpeakerInfo
from .audio_processing import AudioProcessor
from .speech_recognition import SpeechRecognitionSystem
from .speaker_diarization import SpeakerDiarizationSystem
from .translation import TranslationSystem
from .voice_cloning import VoiceCloningSystem
from .prosody_matching import ProsodyMatcher
from .lipsync import LipSyncProcessor
from .video_processing import VideoProcessor

logger = logging.getLogger(__name__)

class DubbingPipeline:
    """
    Complete dubbing pipeline with voice cloning and speaker preservation
    """
    
    def __init__(self):
        # Initialize all processing components
        self.audio_processor = AudioProcessor()
        self.speech_recognition = SpeechRecognitionSystem()
        self.speaker_diarization = SpeakerDiarizationSystem()
        self.translation_system = TranslationSystem()
        self.voice_cloning_system = VoiceCloningSystem()
        self.prosody_matcher = ProsodyMatcher()
        self.lipsync_processor = LipSyncProcessor()
        self.video_processor = VideoProcessor()
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize all pipeline components"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing dubbing pipeline...")
            
            # Initialize components in parallel where possible
            await asyncio.gather(
                self.audio_processor.initialize(),
                self.speech_recognition.initialize(),
                self.speaker_diarization.initialize(),
                self.translation_system.initialize(),
                self.prosody_matcher.initialize(),
                self.lipsync_processor.initialize(),
                self.video_processor.initialize()
            )
            
            self.initialized = True
            logger.info("Dubbing pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize dubbing pipeline: {str(e)}")
            raise
    
    async def process(
        self, 
        request: DubbingRequest, 
        job_id: str
    ) -> AsyncGenerator[Dict, None]:
        """
        Process video through complete dubbing pipeline
        
        Args:
            request: Dubbing request parameters
            job_id: Unique job identifier
            
        Yields:
            Status updates throughout the process
        """
        start_time = time.time()
        temp_dir = settings.temp_dir / job_id
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: Extract audio from video
            yield {
                "status": ProcessingStatus.EXTRACTING_AUDIO,
                "progress": 5,
                "message": "Extracting audio from video...",
                "current_step": "audio_extraction"
            }
            
            audio_path = await self.audio_processor.extract_audio_from_video(
                request.input_file,
                str(temp_dir / "extracted_audio.wav")
            )
            
            # Step 2: Source separation (separate speech from music/effects)
            yield {
                "status": ProcessingStatus.SOURCE_SEPARATION,
                "progress": 10,
                "message": "Separating speech from background audio...",
                "current_step": "source_separation"
            }
            
            separated_audio = await self.audio_processor.separate_sources(
                audio_path,
                str(temp_dir / "separated")
            )
            speech_audio = separated_audio["vocals"]
            background_audio = separated_audio["accompaniment"]
            
            # Step 3: Speech recognition
            yield {
                "status": ProcessingStatus.SPEECH_RECOGNITION,
                "progress": 20,
                "message": "Converting speech to text...",
                "current_step": "speech_recognition"
            }
            
            transcription_result = await self.speech_recognition.transcribe(
                speech_audio,
                language=request.source_language
            )
            
            # Step 4: Speaker diarization and voice tracking
            yield {
                "status": ProcessingStatus.SPEAKER_DIARIZATION,
                "progress": 30,
                "message": "Identifying and tracking speakers...",
                "current_step": "speaker_diarization"
            }
            
            speakers_info, speaker_audio_files = await self.speaker_diarization.diarize_speakers(
                speech_audio,
                num_speakers=None
            )
            
            # Step 5: Voice cloning - Create voice profiles for each speaker
            yield {
                "status": ProcessingStatus.VOICE_CLONING,
                "progress": 40,
                "message": "Creating voice profiles for speakers...",
                "current_step": "voice_cloning"
            }
            
            speaker_voice_profiles = {}
            if request.use_voice_cloning:
                speaker_voice_profiles = await self.speaker_diarization.create_speaker_voice_profiles(
                    speaker_audio_files,
                    speakers_info,
                    request.target_language
                )
            
            # Step 6: Emotion detection
            yield {
                "status": ProcessingStatus.EMOTION_DETECTION,
                "progress": 45,
                "message": "Analyzing emotional content...",
                "current_step": "emotion_detection"
            }
            
            # Emotion detection is integrated in speaker diarization
            
            # Step 7: Translation
            yield {
                "status": ProcessingStatus.TRANSLATION,
                "progress": 55,
                "message": "Translating text to target language...",
                "current_step": "translation"
            }
            
            translation_result = await self.translation_system.translate_segments(
                transcription_result["segments"],
                source_language=request.source_language,
                target_language=request.target_language,
                preserve_emotion=request.preserve_emotion
            )
            
            # Step 8: Text-to-Speech with voice cloning
            yield {
                "status": ProcessingStatus.TEXT_TO_SPEECH,
                "progress": 70,
                "message": "Generating speech with cloned voices...",
                "current_step": "text_to_speech"
            }
            
            tts_segments = await self._generate_dubbed_speech(
                translation_result,
                speakers_info,
                speaker_voice_profiles,
                str(temp_dir / "tts_segments"),
                request.target_language
            )
            
            # Step 9: Prosody matching
            yield {
                "status": ProcessingStatus.PROSODY_MATCHING,
                "progress": 80,
                "message": "Matching speech timing and prosody...",
                "current_step": "prosody_matching"
            }
            
            matched_audio_segments = await self.prosody_matcher.match_prosody(
                tts_segments,
                transcription_result["segments"],
                str(temp_dir / "prosody_matched")
            )
            
            # Step 10: Audio mixing
            yield {
                "status": ProcessingStatus.AUDIO_MIXING,
                "progress": 85,
                "message": "Mixing dubbed speech with background audio...",
                "current_step": "audio_mixing"
            }
            
            final_audio_path = await self.audio_processor.mix_audio_segments(
                matched_audio_segments,
                background_audio,
                str(temp_dir / "final_audio.wav")
            )
            
            # Step 11: Lip sync (if enabled)
            final_video_path = str(settings.output_dir / f"dubbed_{job_id}.mp4")
            
            if request.enable_lipsync:
                yield {
                    "status": ProcessingStatus.LIP_SYNC,
                    "progress": 90,
                    "message": "Synchronizing lip movements...",
                    "current_step": "lip_sync"
                }
                
                final_video_path = await self.lipsync_processor.sync_lips(
                    request.input_file,
                    final_audio_path,
                    final_video_path
                )
            else:
                # Step 12: Final video assembly without lip sync
                yield {
                    "status": ProcessingStatus.FINAL_VIDEO,
                    "progress": 90,
                    "message": "Assembling final video...",
                    "current_step": "final_video"
                }
                
                final_video_path = await self.video_processor.replace_audio(
                    request.input_file,
                    final_audio_path,
                    final_video_path
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate quality report
            quality_report = await self._generate_quality_report(
                transcription_result,
                translation_result,
                speakers_info,
                speaker_voice_profiles
            )
            
            # Final completion
            yield {
                "status": ProcessingStatus.COMPLETED,
                "progress": 100,
                "message": "Dubbing completed successfully!",
                "current_step": "completed",
                "output_path": final_video_path,
                "processing_time": processing_time,
                "speakers_detected": speakers_info,
                "voice_profiles_created": list(speaker_voice_profiles.values()),
                "translation_quality": quality_report,
                "quality_metrics": {
                    "total_speakers": len(set(s.speaker_id for s in speakers_info)),
                    "total_duration": sum(s.end_time - s.start_time for s in speakers_info),
                    "voice_cloning_enabled": request.use_voice_cloning,
                    "lip_sync_enabled": request.enable_lipsync
                }
            }
            
        except Exception as e:
            logger.error(f"Error in dubbing pipeline: {str(e)}")
            yield {
                "status": ProcessingStatus.FAILED,
                "progress": 0,
                "message": f"Processing failed: {str(e)}",
                "error": str(e)
            }
        finally:
            # Cleanup temporary files
            await self._cleanup_temp_files(temp_dir)
    
    async def _generate_dubbed_speech(
        self,
        translation_result: Dict,
        speakers_info: List[SpeakerInfo],
        speaker_voice_profiles: Dict[str, str],
        output_dir: str,
        target_language: str
    ) -> List[Dict]:
        """
        Generate dubbed speech using voice cloning
        
        Args:
            translation_result: Translation results
            speakers_info: Speaker information
            speaker_voice_profiles: Voice profile IDs by speaker
            output_dir: Output directory for audio segments
            target_language: Target language
            
        Returns:
            List of generated audio segments
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            tts_segments = []
            
            for i, segment in enumerate(translation_result["segments"]):
                speaker_id = segment.get("speaker_id", "unknown")
                translated_text = segment["translated_text"]
                emotion = segment.get("emotion", "neutral")
                
                # Get voice profile for this speaker
                voice_profile_id = speaker_voice_profiles.get(speaker_id)
                
                if voice_profile_id:
                    # Use cloned voice
                    voice_profile = self.voice_cloning_system.get_voice_profile(voice_profile_id)
                    
                    if voice_profile:
                        output_path = output_dir / f"segment_{i:04d}_{speaker_id}.wav"
                        
                        await self.voice_cloning_system.clone_voice(
                            text=translated_text,
                            voice_profile=voice_profile,
                            output_path=str(output_path),
                            emotion=emotion
                        )
                        
                        tts_segments.append({
                            "segment_id": f"segment_{i:04d}",
                            "speaker_id": speaker_id,
                            "audio_path": str(output_path),
                            "start_time": segment["start"],
                            "end_time": segment["end"],
                            "text": translated_text,
                            "emotion": emotion,
                            "voice_cloned": True
                        })
                    else:
                        logger.warning(f"Voice profile not found: {voice_profile_id}")
                        # Fallback to default TTS
                        tts_segments.append(await self._generate_default_tts(
                            segment, i, output_dir, target_language
                        ))
                else:
                    # Use default TTS
                    tts_segments.append(await self._generate_default_tts(
                        segment, i, output_dir, target_language
                    ))
            
            return tts_segments
            
        except Exception as e:
            logger.error(f"Error generating dubbed speech: {str(e)}")
            raise
    
    async def _generate_default_tts(
        self,
        segment: Dict,
        segment_index: int,
        output_dir: Path,
        target_language: str
    ) -> Dict:
        """Generate TTS using default voice"""
        try:
            output_path = output_dir / f"segment_{segment_index:04d}_default.wav"
            
            # Use basic TTS (implement based on available TTS system)
            # This is a placeholder - implement with your preferred TTS
            
            return {
                "segment_id": f"segment_{segment_index:04d}",
                "speaker_id": segment.get("speaker_id", "unknown"),
                "audio_path": str(output_path),
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["translated_text"],
                "emotion": segment.get("emotion", "neutral"),
                "voice_cloned": False
            }
            
        except Exception as e:
            logger.error(f"Error generating default TTS: {str(e)}")
            raise
    
    async def _generate_quality_report(
        self,
        transcription_result: Dict,
        translation_result: Dict,
        speakers_info: List[SpeakerInfo],
        speaker_voice_profiles: Dict[str, str]
    ) -> Dict:
        """Generate quality assessment report"""
        try:
            return {
                "transcription_confidence": transcription_result.get("confidence", 0.0),
                "translation_quality": translation_result.get("bleu_score", 0.0),
                "speakers_detected": len(set(s.speaker_id for s in speakers_info)),
                "voice_profiles_created": len(speaker_voice_profiles),
                "total_segments": len(translation_result.get("segments", [])),
                "average_segment_duration": sum(
                    s.end_time - s.start_time for s in speakers_info
                ) / len(speakers_info) if speakers_info else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating quality report: {str(e)}")
            return {}
    
    async def _cleanup_temp_files(self, temp_dir: Path):
        """Clean up temporary files"""
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")
    
    async def get_model_status(self) -> Dict[str, bool]:
        """Get status of all loaded models"""
        return {
            "audio_processor": self.audio_processor.initialized if hasattr(self.audio_processor, 'initialized') else False,
            "speech_recognition": self.speech_recognition.initialized if hasattr(self.speech_recognition, 'initialized') else False,
            "speaker_diarization": self.speaker_diarization.initialized if hasattr(self.speaker_diarization, 'initialized') else False,
            "translation_system": self.translation_system.initialized if hasattr(self.translation_system, 'initialized') else False,
            "voice_cloning_system": self.voice_cloning_system.initialized if hasattr(self.voice_cloning_system, 'initialized') else False,
            "prosody_matcher": self.prosody_matcher.initialized if hasattr(self.prosody_matcher, 'initialized') else False,
            "lipsync_processor": self.lipsync_processor.initialized if hasattr(self.lipsync_processor, 'initialized') else False,
            "video_processor": self.video_processor.initialized if hasattr(self.video_processor, 'initialized') else False
        }
    
    async def cleanup(self):
        """Cleanup all pipeline components"""
        try:
            await asyncio.gather(
                self.speaker_diarization.cleanup(),
                self.voice_cloning_system.cleanup(),
                return_exceptions=True
            )
            logger.info("Dubbing pipeline cleaned up")
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {str(e)}")