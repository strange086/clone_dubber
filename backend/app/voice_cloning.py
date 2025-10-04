"""
Voice Cloning and Speaker Tracking System for EduDub AI
Implements voice preservation and cloning using Coqui XTTS-v2
"""
import torch
import torchaudio
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json
import uuid
from datetime import datetime

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from speechbrain.pretrained import EncoderClassifier
import soundfile as sf

from .config import settings
from .models import VoiceProfile, VoiceCloneRequest, VoiceCloneResponse, SpeakerInfo

logger = logging.getLogger(__name__)

class VoiceCloningSystem:
    """
    Advanced voice cloning system for preserving artist's voice across languages
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tts_model = None
        self.speaker_encoder = None
        self.voice_profiles = {}
        self.profiles_dir = Path("voice_profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        
        # Voice quality thresholds
        self.min_audio_duration = 10.0  # seconds
        self.quality_threshold = 0.7
        self.similarity_threshold = 0.75
        
    async def initialize(self):
        """Initialize voice cloning models"""
        try:
            logger.info("Initializing voice cloning system...")
            
            # Initialize Coqui XTTS-v2 for multilingual TTS
            logger.info("Loading XTTS-v2 model...")
            self.tts_model = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=True
            ).to(self.device)
            
            # Initialize speaker encoder for voice embeddings
            logger.info("Loading speaker encoder...")
            self.speaker_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            
            # Load existing voice profiles
            await self._load_voice_profiles()
            
            logger.info("Voice cloning system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice cloning system: {str(e)}")
            raise
    
    async def _load_voice_profiles(self):
        """Load existing voice profiles from disk"""
        try:
            profiles_file = self.profiles_dir / "profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                    
                for profile_data in profiles_data:
                    profile = VoiceProfile(**profile_data)
                    self.voice_profiles[profile.profile_id] = profile
                    
                logger.info(f"Loaded {len(self.voice_profiles)} voice profiles")
        except Exception as e:
            logger.error(f"Error loading voice profiles: {str(e)}")
    
    async def _save_voice_profiles(self):
        """Save voice profiles to disk"""
        try:
            profiles_file = self.profiles_dir / "profiles.json"
            profiles_data = []
            
            for profile in self.voice_profiles.values():
                profile_dict = profile.dict()
                profile_dict['created_at'] = profile.created_at.isoformat()
                profiles_data.append(profile_dict)
            
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving voice profiles: {str(e)}")
    
    async def extract_voice_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract voice embedding from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Voice embedding vector
        """
        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Extract speaker embedding
            with torch.no_grad():
                embedding = self.speaker_encoder.encode_batch(waveform)
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting voice embedding: {str(e)}")
            raise
    
    async def analyze_audio_quality(self, audio_path: str) -> Dict[str, float]:
        """
        Analyze audio quality for voice cloning
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Quality metrics dictionary
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            
            # Calculate quality metrics
            # 1. Signal-to-noise ratio
            rms = librosa.feature.rms(y=audio)[0]
            snr = np.mean(rms) / (np.std(rms) + 1e-8)
            
            # 2. Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            brightness = np.mean(spectral_centroids)
            
            # 3. Zero crossing rate (speech clarity)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            clarity = 1.0 - np.mean(zcr)  # Lower ZCR = clearer speech
            
            # 4. Spectral rolloff (frequency content)
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            frequency_content = np.mean(rolloff) / (sr / 2)  # Normalized
            
            # Calculate overall quality score
            quality_score = (
                min(snr / 10.0, 1.0) * 0.3 +  # SNR component
                min(clarity, 1.0) * 0.3 +      # Clarity component
                min(brightness / 2000, 1.0) * 0.2 +  # Brightness component
                min(frequency_content, 1.0) * 0.2     # Frequency content
            )
            
            return {
                "duration": duration,
                "quality_score": quality_score,
                "snr": snr,
                "brightness": brightness,
                "clarity": clarity,
                "frequency_content": frequency_content
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio quality: {str(e)}")
            return {"duration": 0, "quality_score": 0}
    
    async def create_voice_profile(
        self, 
        audio_path: str, 
        speaker_info: SpeakerInfo,
        target_language: str,
        speaker_name: Optional[str] = None
    ) -> VoiceProfile:
        """
        Create a voice profile for cloning
        
        Args:
            audio_path: Path to reference audio
            speaker_info: Speaker information
            target_language: Target language for cloning
            speaker_name: Optional speaker name
            
        Returns:
            Created voice profile
        """
        try:
            logger.info(f"Creating voice profile for speaker {speaker_info.speaker_id}")
            
            # Analyze audio quality
            quality_metrics = await self.analyze_audio_quality(audio_path)
            
            if quality_metrics["duration"] < self.min_audio_duration:
                raise ValueError(f"Audio too short: {quality_metrics['duration']:.1f}s (minimum: {self.min_audio_duration}s)")
            
            if quality_metrics["quality_score"] < self.quality_threshold:
                logger.warning(f"Low audio quality: {quality_metrics['quality_score']:.2f}")
            
            # Extract voice embedding
            voice_embedding = await self.extract_voice_embedding(audio_path)
            
            # Create profile
            profile_id = str(uuid.uuid4())
            profile = VoiceProfile(
                profile_id=profile_id,
                speaker_name=speaker_name or f"Speaker_{speaker_info.speaker_id}",
                gender=speaker_info.gender,
                language=target_language,
                voice_embedding=voice_embedding.tolist(),
                reference_audio_path=audio_path,
                duration=quality_metrics["duration"],
                quality_score=quality_metrics["quality_score"]
            )
            
            # Save profile
            self.voice_profiles[profile_id] = profile
            await self._save_voice_profiles()
            
            logger.info(f"Voice profile created: {profile_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating voice profile: {str(e)}")
            raise
    
    async def find_similar_voice(
        self, 
        target_embedding: np.ndarray, 
        language: str,
        gender: Optional[str] = None
    ) -> Optional[VoiceProfile]:
        """
        Find similar voice profile based on embedding
        
        Args:
            target_embedding: Target voice embedding
            language: Target language
            gender: Optional gender filter
            
        Returns:
            Most similar voice profile or None
        """
        try:
            best_profile = None
            best_similarity = 0.0
            
            for profile in self.voice_profiles.values():
                # Filter by language and gender if specified
                if profile.language != language:
                    continue
                if gender and profile.gender != gender:
                    continue
                
                # Calculate cosine similarity
                profile_embedding = np.array(profile.voice_embedding)
                similarity = np.dot(target_embedding, profile_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(profile_embedding)
                )
                
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_profile = profile
            
            if best_profile:
                logger.info(f"Found similar voice: {best_profile.profile_id} (similarity: {best_similarity:.3f})")
            
            return best_profile
            
        except Exception as e:
            logger.error(f"Error finding similar voice: {str(e)}")
            return None
    
    async def clone_voice(
        self, 
        text: str, 
        voice_profile: VoiceProfile,
        output_path: str,
        emotion: Optional[str] = None,
        speed: float = 1.0
    ) -> str:
        """
        Generate speech using cloned voice
        
        Args:
            text: Text to synthesize
            voice_profile: Voice profile to use
            output_path: Output audio file path
            emotion: Optional emotion to apply
            speed: Speech speed multiplier
            
        Returns:
            Path to generated audio file
        """
        try:
            logger.info(f"Cloning voice for profile {voice_profile.profile_id}")
            
            # Prepare reference audio
            ref_audio_path = voice_profile.reference_audio_path
            
            # Generate speech with XTTS-v2
            self.tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=ref_audio_path,
                language=voice_profile.language,
                speed=speed
            )
            
            # Apply emotion if specified
            if emotion and emotion != "neutral":
                await self._apply_emotion(output_path, emotion)
            
            logger.info(f"Voice cloning completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error cloning voice: {str(e)}")
            raise
    
    async def _apply_emotion(self, audio_path: str, emotion: str):
        """
        Apply emotional characteristics to synthesized speech
        
        Args:
            audio_path: Path to audio file
            emotion: Emotion to apply
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Apply emotion-based modifications
            if emotion == "happy":
                # Increase pitch and tempo slightly
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
                audio = librosa.effects.time_stretch(audio, rate=1.1)
            elif emotion == "sad":
                # Decrease pitch and tempo
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
                audio = librosa.effects.time_stretch(audio, rate=0.9)
            elif emotion == "angry":
                # Increase intensity and slightly higher pitch
                audio = audio * 1.2  # Increase volume
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.5)
            elif emotion == "fear":
                # Higher pitch, faster tempo
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
                audio = librosa.effects.time_stretch(audio, rate=1.15)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Save modified audio
            sf.write(audio_path, audio, sr)
            
        except Exception as e:
            logger.error(f"Error applying emotion: {str(e)}")
    
    async def batch_clone_voices(
        self, 
        segments: List[Dict], 
        voice_profiles: Dict[str, VoiceProfile],
        output_dir: str
    ) -> List[str]:
        """
        Batch clone voices for multiple segments
        
        Args:
            segments: List of text segments with speaker info
            voice_profiles: Dictionary of voice profiles by speaker ID
            output_dir: Output directory for audio files
            
        Returns:
            List of generated audio file paths
        """
        try:
            output_paths = []
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            for i, segment in enumerate(segments):
                speaker_id = segment["speaker_id"]
                text = segment["translated_text"]
                emotion = segment.get("emotion", "neutral")
                
                if speaker_id not in voice_profiles:
                    logger.warning(f"No voice profile for speaker {speaker_id}")
                    continue
                
                voice_profile = voice_profiles[speaker_id]
                output_path = output_dir / f"segment_{i:04d}_{speaker_id}.wav"
                
                await self.clone_voice(
                    text=text,
                    voice_profile=voice_profile,
                    output_path=str(output_path),
                    emotion=emotion
                )
                
                output_paths.append(str(output_path))
            
            return output_paths
            
        except Exception as e:
            logger.error(f"Error in batch voice cloning: {str(e)}")
            raise
    
    def get_voice_profiles(self) -> List[VoiceProfile]:
        """Get all voice profiles"""
        return list(self.voice_profiles.values())
    
    def get_voice_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """Get specific voice profile"""
        return self.voice_profiles.get(profile_id)
    
    async def delete_voice_profile(self, profile_id: str) -> bool:
        """Delete voice profile"""
        try:
            if profile_id in self.voice_profiles:
                profile = self.voice_profiles[profile_id]
                
                # Delete reference audio file
                ref_path = Path(profile.reference_audio_path)
                if ref_path.exists():
                    ref_path.unlink()
                
                # Remove from memory
                del self.voice_profiles[profile_id]
                
                # Save updated profiles
                await self._save_voice_profiles()
                
                logger.info(f"Voice profile deleted: {profile_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting voice profile: {str(e)}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear models from memory
            if self.tts_model:
                del self.tts_model
            if self.speaker_encoder:
                del self.speaker_encoder
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Voice cloning system cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")