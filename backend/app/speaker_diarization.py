"""
Speaker Diarization and Voice Tracking System
Identifies and tracks multiple speakers in audio content
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import timedelta

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Annotation, Segment
import librosa
import soundfile as sf

from .config import settings
from .models import SpeakerInfo, EmotionAnalysis
from .voice_cloning import VoiceCloningSystem

logger = logging.getLogger(__name__)

class SpeakerDiarizationSystem:
    """
    Advanced speaker diarization system for multi-speaker content
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diarization_pipeline = None
        self.speaker_embedding_model = None
        self.emotion_classifier = None
        self.voice_cloning_system = None
        
        # Diarization parameters
        self.min_segment_duration = 1.0  # seconds
        self.max_speakers = 10
        self.similarity_threshold = 0.8
        
    async def initialize(self):
        """Initialize diarization models"""
        try:
            logger.info("Initializing speaker diarization system...")
            
            # Initialize pyannote diarization pipeline
            logger.info("Loading pyannote diarization pipeline...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=None  # Add HuggingFace token if needed
            )
            
            # Initialize speaker embedding model
            logger.info("Loading speaker embedding model...")
            self.speaker_embedding_model = PretrainedSpeakerEmbedding(
                "speechbrain/spkrec-ecapa-voxceleb",
                device=self.device
            )
            
            # Initialize emotion classifier
            logger.info("Loading emotion classifier...")
            from transformers import pipeline
            self.emotion_classifier = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize voice cloning system
            self.voice_cloning_system = VoiceCloningSystem()
            await self.voice_cloning_system.initialize()
            
            logger.info("Speaker diarization system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize speaker diarization system: {str(e)}")
            raise
    
    async def diarize_speakers(
        self, 
        audio_path: str,
        num_speakers: Optional[int] = None
    ) -> Tuple[List[SpeakerInfo], Dict[str, str]]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            num_speakers: Expected number of speakers (optional)
            
        Returns:
            Tuple of (speaker_info_list, speaker_audio_files)
        """
        try:
            logger.info(f"Starting speaker diarization for: {audio_path}")
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Prepare audio for pyannote (mono, 16kHz)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Create temporary file for pyannote
            temp_audio_path = Path(audio_path).parent / f"temp_{Path(audio_path).stem}.wav"
            torchaudio.save(temp_audio_path, waveform, sample_rate)
            
            # Run diarization
            diarization_params = {}
            if num_speakers:
                diarization_params["num_speakers"] = min(num_speakers, self.max_speakers)
            
            diarization = self.diarization_pipeline(
                str(temp_audio_path),
                **diarization_params
            )
            
            # Process diarization results
            speakers_info = []
            speaker_audio_files = {}
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.duration < self.min_segment_duration:
                    continue
                
                # Extract speaker segment
                start_sample = int(turn.start * sample_rate)
                end_sample = int(turn.end * sample_rate)
                speaker_audio = waveform[:, start_sample:end_sample]
                
                # Save speaker segment
                speaker_audio_path = Path(audio_path).parent / f"speaker_{speaker}_{turn.start:.2f}_{turn.end:.2f}.wav"
                torchaudio.save(speaker_audio_path, speaker_audio, sample_rate)
                
                # Detect gender
                gender = await self._detect_gender(speaker_audio, sample_rate)
                
                # Detect emotion
                emotion_info = await self._detect_emotion(str(speaker_audio_path))
                
                # Create speaker info
                speaker_info = SpeakerInfo(
                    speaker_id=speaker,
                    gender=gender,
                    start_time=turn.start,
                    end_time=turn.end,
                    confidence=0.9,  # pyannote doesn't provide confidence scores
                    emotion=emotion_info.emotion if emotion_info else None
                )
                
                speakers_info.append(speaker_info)
                
                if speaker not in speaker_audio_files:
                    speaker_audio_files[speaker] = []
                speaker_audio_files[speaker].append(str(speaker_audio_path))
            
            # Clean up temporary file
            if temp_audio_path.exists():
                temp_audio_path.unlink()
            
            logger.info(f"Diarization completed. Found {len(set(s.speaker_id for s in speakers_info))} speakers")
            
            return speakers_info, speaker_audio_files
            
        except Exception as e:
            logger.error(f"Error in speaker diarization: {str(e)}")
            raise
    
    async def _detect_gender(self, audio_tensor: torch.Tensor, sample_rate: int) -> str:
        """
        Detect speaker gender from audio
        
        Args:
            audio_tensor: Audio tensor
            sample_rate: Sample rate
            
        Returns:
            Detected gender: 'male', 'female', or 'unknown'
        """
        try:
            # Convert to numpy for analysis
            audio_np = audio_tensor.squeeze().numpy()
            
            # Calculate fundamental frequency (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_np,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate
            )
            
            # Remove NaN values
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) == 0:
                return "unknown"
            
            # Calculate average F0
            avg_f0 = np.mean(f0_clean)
            
            # Gender classification based on F0
            # Typical ranges: Male: 85-180 Hz, Female: 165-265 Hz
            if avg_f0 < 150:
                return "male"
            elif avg_f0 > 180:
                return "female"
            else:
                # Use spectral features for ambiguous cases
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate))
                return "female" if spectral_centroid > 2000 else "male"
                
        except Exception as e:
            logger.error(f"Error detecting gender: {str(e)}")
            return "unknown"
    
    async def _detect_emotion(self, audio_path: str) -> Optional[EmotionAnalysis]:
        """
        Detect emotion from audio segment
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Emotion analysis result
        """
        try:
            # Load audio for emotion detection
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Run emotion classification
            result = self.emotion_classifier(audio)
            
            if result and len(result) > 0:
                top_emotion = result[0]
                
                return EmotionAnalysis(
                    emotion=top_emotion['label'].lower(),
                    confidence=top_emotion['score'],
                    start_time=0.0,
                    end_time=len(audio) / sr,
                    intensity=top_emotion['score']
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting emotion: {str(e)}")
            return None
    
    async def create_speaker_voice_profiles(
        self,
        speaker_audio_files: Dict[str, List[str]],
        speakers_info: List[SpeakerInfo],
        target_language: str
    ) -> Dict[str, str]:
        """
        Create voice profiles for all detected speakers
        
        Args:
            speaker_audio_files: Dictionary of speaker audio files
            speakers_info: List of speaker information
            target_language: Target language for voice profiles
            
        Returns:
            Dictionary mapping speaker_id to voice_profile_id
        """
        try:
            logger.info("Creating voice profiles for detected speakers...")
            
            speaker_profiles = {}
            speaker_info_dict = {s.speaker_id: s for s in speakers_info}
            
            for speaker_id, audio_files in speaker_audio_files.items():
                if speaker_id not in speaker_info_dict:
                    continue
                
                speaker_info = speaker_info_dict[speaker_id]
                
                # Find the longest audio segment for this speaker
                best_audio_file = None
                best_duration = 0
                
                for audio_file in audio_files:
                    audio, sr = librosa.load(audio_file, sr=None)
                    duration = len(audio) / sr
                    
                    if duration > best_duration:
                        best_duration = duration
                        best_audio_file = audio_file
                
                if best_audio_file and best_duration >= 3.0:  # Minimum 3 seconds
                    try:
                        # Create voice profile
                        voice_profile = await self.voice_cloning_system.create_voice_profile(
                            audio_path=best_audio_file,
                            speaker_info=speaker_info,
                            target_language=target_language,
                            speaker_name=f"Speaker_{speaker_id}"
                        )
                        
                        speaker_profiles[speaker_id] = voice_profile.profile_id
                        logger.info(f"Created voice profile for speaker {speaker_id}: {voice_profile.profile_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to create voice profile for speaker {speaker_id}: {str(e)}")
                        continue
                else:
                    logger.warning(f"Insufficient audio for speaker {speaker_id} (duration: {best_duration:.1f}s)")
            
            return speaker_profiles
            
        except Exception as e:
            logger.error(f"Error creating speaker voice profiles: {str(e)}")
            raise
    
    async def track_speaker_consistency(
        self,
        speakers_info: List[SpeakerInfo],
        audio_path: str
    ) -> Dict[str, float]:
        """
        Track speaker consistency across the audio
        
        Args:
            speakers_info: List of speaker information
            audio_path: Path to original audio file
            
        Returns:
            Dictionary of speaker consistency scores
        """
        try:
            logger.info("Tracking speaker consistency...")
            
            consistency_scores = {}
            speaker_embeddings = {}
            
            # Group segments by speaker
            speaker_segments = {}
            for speaker_info in speakers_info:
                speaker_id = speaker_info.speaker_id
                if speaker_id not in speaker_segments:
                    speaker_segments[speaker_id] = []
                speaker_segments[speaker_id].append(speaker_info)
            
            # Load original audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Calculate embeddings for each speaker segment
            for speaker_id, segments in speaker_segments.items():
                embeddings = []
                
                for segment in segments:
                    # Extract segment audio
                    start_sample = int(segment.start_time * sample_rate)
                    end_sample = int(segment.end_time * sample_rate)
                    segment_audio = waveform[:, start_sample:end_sample]
                    
                    # Get speaker embedding
                    embedding = self.speaker_embedding_model(segment_audio)
                    embeddings.append(embedding.cpu().numpy())
                
                if len(embeddings) > 1:
                    # Calculate consistency as average pairwise similarity
                    similarities = []
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            sim = np.dot(embeddings[i], embeddings[j]) / (
                                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                            )
                            similarities.append(sim)
                    
                    consistency_scores[speaker_id] = np.mean(similarities)
                else:
                    consistency_scores[speaker_id] = 1.0  # Single segment = perfect consistency
                
                # Store average embedding for this speaker
                speaker_embeddings[speaker_id] = np.mean(embeddings, axis=0)
            
            logger.info(f"Speaker consistency analysis completed for {len(consistency_scores)} speakers")
            
            return consistency_scores
            
        except Exception as e:
            logger.error(f"Error tracking speaker consistency: {str(e)}")
            return {}
    
    async def merge_similar_speakers(
        self,
        speakers_info: List[SpeakerInfo],
        similarity_threshold: float = 0.85
    ) -> List[SpeakerInfo]:
        """
        Merge speakers that are likely the same person
        
        Args:
            speakers_info: List of speaker information
            similarity_threshold: Threshold for merging speakers
            
        Returns:
            Updated list of speaker information with merged speakers
        """
        try:
            logger.info("Checking for similar speakers to merge...")
            
            # Group by speaker ID and calculate average embeddings
            speaker_groups = {}
            for speaker_info in speakers_info:
                speaker_id = speaker_info.speaker_id
                if speaker_id not in speaker_groups:
                    speaker_groups[speaker_id] = []
                speaker_groups[speaker_id].append(speaker_info)
            
            # This is a simplified version - in practice, you'd need to
            # extract embeddings and compare them across different speaker IDs
            # For now, we'll return the original list
            
            return speakers_info
            
        except Exception as e:
            logger.error(f"Error merging similar speakers: {str(e)}")
            return speakers_info
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.voice_cloning_system:
                await self.voice_cloning_system.cleanup()
            
            # Clear models from memory
            if self.diarization_pipeline:
                del self.diarization_pipeline
            if self.speaker_embedding_model:
                del self.speaker_embedding_model
            if self.emotion_classifier:
                del self.emotion_classifier
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Speaker diarization system cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_speaker_statistics(self, speakers_info: List[SpeakerInfo]) -> Dict:
        """Get statistics about detected speakers"""
        try:
            total_speakers = len(set(s.speaker_id for s in speakers_info))
            total_duration = sum(s.end_time - s.start_time for s in speakers_info)
            
            gender_counts = {}
            emotion_counts = {}
            
            for speaker_info in speakers_info:
                # Count genders
                gender = speaker_info.gender
                gender_counts[gender] = gender_counts.get(gender, 0) + 1
                
                # Count emotions
                if speaker_info.emotion:
                    emotion = speaker_info.emotion
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            return {
                "total_speakers": total_speakers,
                "total_segments": len(speakers_info),
                "total_duration": total_duration,
                "gender_distribution": gender_counts,
                "emotion_distribution": emotion_counts,
                "average_segment_duration": total_duration / len(speakers_info) if speakers_info else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating speaker statistics: {str(e)}")
            return {}