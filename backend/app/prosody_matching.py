"""
Prosody Matching System
"""
import librosa
import soundfile as sf
import numpy as np
import logging
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class ProsodyMatcher:
    """Match prosody between original and synthesized speech"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize prosody matcher"""
        self.initialized = True
        logger.info("Prosody matcher initialized")
    
    async def match_prosody(
        self, 
        tts_segments: List[Dict], 
        original_segments: List[Dict],
        output_dir: str
    ) -> List[Dict]:
        """Match prosody of TTS segments to original"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            matched_segments = []
            
            for i, (tts_seg, orig_seg) in enumerate(zip(tts_segments, original_segments)):
                if 'audio_path' not in tts_seg:
                    continue
                
                # Load TTS audio
                tts_audio, sr = librosa.load(tts_seg['audio_path'], sr=None)
                
                # Calculate target duration
                target_duration = orig_seg['end'] - orig_seg['start']
                current_duration = len(tts_audio) / sr
                
                # Time stretch to match duration
                if current_duration > 0:
                    stretch_factor = target_duration / current_duration
                    matched_audio = librosa.effects.time_stretch(tts_audio, rate=stretch_factor)
                else:
                    matched_audio = tts_audio
                
                # Save matched audio
                output_path = output_dir / f"matched_{i:04d}.wav"
                sf.write(str(output_path), matched_audio, sr)
                
                matched_segment = tts_seg.copy()
                matched_segment['audio_path'] = str(output_path)
                matched_segments.append(matched_segment)
            
            return matched_segments
            
        except Exception as e:
            logger.error(f"Error in prosody matching: {str(e)}")
            raise