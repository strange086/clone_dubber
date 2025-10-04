"""
Utility functions for EduDub AI Platform
"""
import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def validate_file_format(filename: str, supported_formats: List[str]) -> bool:
    """
    Validate if file format is supported
    
    Args:
        filename: Name of the file
        supported_formats: List of supported file extensions
        
    Returns:
        True if format is supported, False otherwise
    """
    if not filename:
        return False
    
    file_ext = Path(filename).suffix.lower()
    return file_ext in supported_formats

def get_file_info(file_path: str) -> Dict:
    """
    Get detailed information about a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return {"error": "File not found"}
        
        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(str(path))
        
        return {
            "filename": path.name,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "extension": path.suffix.lower(),
            "mime_type": mime_type,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_video": mime_type and mime_type.startswith("video/") if mime_type else False,
            "is_audio": mime_type and mime_type.startswith("audio/") if mime_type else False
        }
        
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        return {"error": str(e)}

def generate_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """
    Generate hash for a file
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (md5, sha1, sha256)
        
    Returns:
        File hash as hex string
    """
    try:
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
        
    except Exception as e:
        logger.error(f"Error generating file hash: {str(e)}")
        return ""

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2m 30s")
    """
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
            
    except Exception:
        return "0s"

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename

def create_directory_structure(base_path: str, subdirs: List[str]) -> Dict[str, Path]:
    """
    Create directory structure
    
    Args:
        base_path: Base directory path
        subdirs: List of subdirectory names
        
    Returns:
        Dictionary mapping subdir names to Path objects
    """
    try:
        base = Path(base_path)
        base.mkdir(exist_ok=True)
        
        paths = {}
        for subdir in subdirs:
            path = base / subdir
            path.mkdir(exist_ok=True)
            paths[subdir] = path
        
        return paths
        
    except Exception as e:
        logger.error(f"Error creating directory structure: {str(e)}")
        return {}

def load_json_file(file_path: str) -> Optional[Dict]:
    """
    Load JSON file safely
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return None

def save_json_file(data: Dict, file_path: str) -> bool:
    """
    Save data to JSON file safely
    
    Args:
        data: Data to save
        file_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        return False

def calculate_audio_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two audio embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        import numpy as np
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Convert to 0-1 range
        return (similarity + 1) / 2
        
    except Exception as e:
        logger.error(f"Error calculating audio similarity: {str(e)}")
        return 0.0

def estimate_processing_time(
    video_duration: float,
    num_speakers: int,
    enable_lipsync: bool,
    use_voice_cloning: bool
) -> float:
    """
    Estimate processing time based on video characteristics
    
    Args:
        video_duration: Duration of video in seconds
        num_speakers: Number of speakers detected
        enable_lipsync: Whether lip sync is enabled
        use_voice_cloning: Whether voice cloning is enabled
        
    Returns:
        Estimated processing time in seconds
    """
    try:
        # Base processing time (1.5x video duration)
        base_time = video_duration * 1.5
        
        # Add time for multiple speakers
        speaker_multiplier = 1 + (num_speakers - 1) * 0.3
        
        # Add time for voice cloning
        if use_voice_cloning:
            base_time *= 1.8
        
        # Add time for lip sync
        if enable_lipsync:
            base_time *= 2.5
        
        # Apply speaker multiplier
        total_time = base_time * speaker_multiplier
        
        return max(total_time, 30)  # Minimum 30 seconds
        
    except Exception as e:
        logger.error(f"Error estimating processing time: {str(e)}")
        return 300  # Default 5 minutes

def validate_language_code(language_code: str, supported_languages: Dict[str, str]) -> bool:
    """
    Validate language code against supported languages
    
    Args:
        language_code: Language code to validate
        supported_languages: Dictionary of supported language codes
        
    Returns:
        True if language is supported, False otherwise
    """
    return language_code in supported_languages

def get_language_name(language_code: str, supported_languages: Dict[str, str]) -> str:
    """
    Get language name from code
    
    Args:
        language_code: Language code
        supported_languages: Dictionary of supported languages
        
    Returns:
        Language name or the code itself if not found
    """
    return supported_languages.get(language_code, language_code)

def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
    """
    Clean up old files from a directory
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep in hours
        
    Returns:
        Number of files deleted
    """
    try:
        from datetime import datetime, timedelta
        
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {str(e)}")
        
        logger.info(f"Cleaned up {deleted_count} old files from {directory}")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error cleaning up old files: {str(e)}")
        return 0

def get_system_info() -> Dict:
    """
    Get system information for diagnostics
    
    Returns:
        Dictionary with system information
    """
    try:
        import platform
        import psutil
        import torch
        
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent,
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.device_count() > 0 else None
            })
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {"error": str(e)}