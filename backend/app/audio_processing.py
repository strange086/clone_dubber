"""
Audio Processing System for EduDub AI Platform
Handles audio extraction, source separation, and mixing
"""
import subprocess
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import ffmpeg

from .config import settings

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Audio processing system for dubbing pipeline
    """
    
    def __init__(self):
        self.demucs_model = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize audio processing models"""
        try:
            logger.info("Initializing audio processing system...")
            
            # Load Demucs model for source separation
            logger.info("Loading Demucs model for source separation...")
            self.demucs_model = get_model('htdemucs')
            
            self.initialized = True
            logger.info("Audio processing system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio processing: {str(e)}")
            raise
    
    async def extract_audio_from_video(self, video_path: str, output_path: str) -> str:
        """
        Extract audio from video file using FFmpeg
        
        Args:
            video_path: Path to input video file
            output_path: Path for output audio file
            
        Returns:
            Path to extracted audio file
        """
        try:
            logger.info(f"Extracting audio from video: {video_path}")
            
            # Use ffmpeg-python for audio extraction
            (
                ffmpeg
                .input(video_path)
                .output(
                    output_path,
                    acodec='pcm_s16le',
                    ac=1,  # mono
                    ar=settings.sample_rate  # sample rate
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            logger.info(f"Audio extracted successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise
    
    async def separate_sources(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        """
        Separate audio sources (vocals, drums, bass, other) using Demucs
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory for separated audio files
            
        Returns:
            Dictionary with paths to separated audio files
        """
        try:
            logger.info(f"Separating audio sources: {audio_path}")
            
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Load audio
            waveform, sample_rate = librosa.load(audio_path, sr=None, mono=False)
            
            # Convert to tensor and ensure correct shape
            if len(waveform.shape) == 1:
                waveform = waveform[None, :]  # Add channel dimension
            
            waveform_tensor = torch.from_numpy(waveform).float()
            
            # Apply Demucs model
            with torch.no_grad():
                sources = apply_model(self.demucs_model, waveform_tensor[None], device='cpu')[0]
            
            # Save separated sources
            source_names = ['drums', 'bass', 'other', 'vocals']
            separated_files = {}
            
            for i, source_name in enumerate(source_names):
                output_path = output_dir / f"{source_name}.wav"
                source_audio = sources[i].numpy()
                
                # Convert to mono if stereo
                if len(source_audio.shape) > 1:
                    source_audio = np.mean(source_audio, axis=0)
                
                sf.write(str(output_path), source_audio, sample_rate)
                separated_files[source_name] = str(output_path)
            
            # Create accompaniment (everything except vocals)
            accompaniment = sources[0] + sources[1] + sources[2]  # drums + bass + other
            accompaniment_audio = accompaniment.numpy()
            
            if len(accompaniment_audio.shape) > 1:
                accompaniment_audio = np.mean(accompaniment_audio, axis=0)
            
            accompaniment_path = output_dir / "accompaniment.wav"
            sf.write(str(accompaniment_path), accompaniment_audio, sample_rate)
            separated_files['accompaniment'] = str(accompaniment_path)
            
            logger.info("Audio source separation completed")
            return separated_files
            
        except Exception as e:
            logger.error(f"Error separating audio sources: {str(e)}")
            raise
    
    async def mix_audio_segments(
        self, 
        audio_segments: List[Dict], 
        background_audio: str, 
        output_path: str
    ) -> str:
        """
        Mix dubbed audio segments with background audio
        
        Args:
            audio_segments: List of audio segment information
            background_audio: Path to background audio file
            output_path: Path for output mixed audio
            
        Returns:
            Path to mixed audio file
        """
        try:
            logger.info("Mixing dubbed audio with background...")
            
            # Load background audio
            bg_audio, sr = librosa.load(background_audio, sr=settings.sample_rate)
            
            # Create empty audio array with same length as background
            mixed_audio = bg_audio.copy()
            
            # Mix in each dubbed segment
            for segment in audio_segments:
                if 'audio_path' not in segment:
                    continue
                
                segment_audio, _ = librosa.load(segment['audio_path'], sr=settings.sample_rate)
                start_sample = int(segment['start_time'] * settings.sample_rate)
                end_sample = start_sample + len(segment_audio)
                
                # Ensure we don't exceed the background audio length
                if end_sample > len(mixed_audio):
                    segment_audio = segment_audio[:len(mixed_audio) - start_sample]
                    end_sample = len(mixed_audio)
                
                # Mix the segment (reduce background volume where speech occurs)
                if start_sample < len(mixed_audio):
                    # Reduce background volume by 70% where speech occurs
                    mixed_audio[start_sample:end_sample] *= 0.3
                    
                    # Add the dubbed speech
                    segment_length = min(len(segment_audio), end_sample - start_sample)
                    mixed_audio[start_sample:start_sample + segment_length] += segment_audio[:segment_length]
            
            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val
            
            # Save mixed audio
            sf.write(output_path, mixed_audio, settings.sample_rate)
            
            logger.info(f"Audio mixing completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error mixing audio: {str(e)}")
            raise
    
    async def normalize_audio(self, audio_path: str, target_lufs: float = -23.0) -> str:
        """
        Normalize audio to target LUFS level
        
        Args:
            audio_path: Path to input audio file
            target_lufs: Target LUFS level
            
        Returns:
            Path to normalized audio file
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Calculate current RMS
            rms = np.sqrt(np.mean(audio**2))
            
            # Calculate target RMS (simplified LUFS approximation)
            target_rms = 10**(target_lufs/20)
            
            # Apply gain
            if rms > 0:
                gain = target_rms / rms
                audio = audio * gain
            
            # Prevent clipping
            audio = np.clip(audio, -1.0, 1.0)
            
            # Save normalized audio
            sf.write(audio_path, audio, sr)
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {str(e)}")
            return audio_path
    
    async def apply_audio_effects(
        self, 
        audio_path: str, 
        effects: Dict[str, float]
    ) -> str:
        """
        Apply audio effects like reverb, compression, etc.
        
        Args:
            audio_path: Path to input audio file
            effects: Dictionary of effects and their parameters
            
        Returns:
            Path to processed audio file
        """
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Apply reverb if specified
            if 'reverb' in effects and effects['reverb'] > 0:
                # Simple reverb simulation using convolution
                reverb_strength = effects['reverb']
                impulse_length = int(0.1 * sr)  # 100ms impulse
                impulse = np.exp(-np.linspace(0, 5, impulse_length)) * np.random.randn(impulse_length)
                reverb_audio = np.convolve(audio, impulse, mode='same') * reverb_strength
                audio = audio + reverb_audio * 0.3
            
            # Apply compression if specified
            if 'compression' in effects and effects['compression'] > 0:
                threshold = -20  # dB
                ratio = 4.0
                
                # Simple compression
                audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
                over_threshold = audio_db > threshold
                
                compressed_db = np.where(
                    over_threshold,
                    threshold + (audio_db - threshold) / ratio,
                    audio_db
                )
                
                audio = np.sign(audio) * 10**(compressed_db / 20)
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            
            # Save processed audio
            sf.write(audio_path, audio, sr)
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Error applying audio effects: {str(e)}")
            return audio_path
    
    def get_audio_info(self, audio_path: str) -> Dict:
        """Get information about audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if len(audio.shape) == 1 else audio.shape[0],
                "samples": len(audio),
                "rms": float(np.sqrt(np.mean(audio**2))),
                "max_amplitude": float(np.max(np.abs(audio)))
            }
            
        except Exception as e:
            logger.error(f"Error getting audio info: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.demucs_model:
                del self.demucs_model
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Audio processing system cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")