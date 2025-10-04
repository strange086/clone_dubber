# 🎙️ EduDub AI - Voice Cloning & Speaker Tracking (Sprint 5)

## Overview

This implementation focuses on **Sprint 5: Voice Tracking & Artist Voice Preservation** - a sophisticated voice cloning system that maintains the original artist's vocal characteristics when dubbing content to different Indian languages.

## 🚀 Key Features

### Voice Cloning System
- **Artist Voice Preservation**: Maintains original speaker's vocal characteristics
- **Multi-Speaker Support**: Handles multiple speakers with individual voice profiles
- **Emotion Preservation**: Maintains emotional tone across languages
- **High-Quality Synthesis**: Uses Coqui XTTS-v2 for natural-sounding speech
- **Voice Quality Analysis**: Automatic quality assessment of voice samples

### Speaker Tracking
- **Advanced Diarization**: Identifies and tracks multiple speakers using pyannote.audio
- **Gender Detection**: Automatic gender classification for voice matching
- **Speaker Consistency**: Tracks speaker identity across the entire video
- **Voice Embedding**: Creates unique voice fingerprints for each speaker

### Complete Pipeline
1. **Audio Extraction** - FFmpeg-based video-to-audio conversion
2. **Source Separation** - Demucs separates speech from background audio
3. **Speech Recognition** - Whisper Large-v3 for accurate transcription
4. **Speaker Diarization** - pyannote.audio identifies multiple speakers
5. **Voice Profile Creation** - Automatic voice cloning profile generation
6. **Emotion Detection** - wav2vec2-emotion for emotional analysis
7. **Translation** - IndicTrans2 for Indian language translation
8. **Voice Cloning** - XTTS-v2 generates speech with original voice
9. **Prosody Matching** - Matches timing and speech patterns
10. **Audio Mixing** - Combines dubbed speech with background audio
11. **Lip Sync** - Wav2Lip for visual-audio synchronization
12. **Final Assembly** - Complete dubbed video generation

## 🛠️ Technical Architecture

### Backend Components
- **FastAPI** - Async web framework with real-time processing
- **Voice Cloning System** - Core voice preservation engine
- **Speaker Diarization** - Multi-speaker identification and tracking
- **Audio Processing** - Complete audio pipeline with source separation
- **Translation System** - Multi-language translation with context preservation

### AI Models Used
- **Coqui XTTS-v2** - Multilingual text-to-speech with voice cloning
- **pyannote.audio** - Speaker diarization and voice activity detection
- **Whisper Large-v3** - Speech recognition for Indian languages
- **Demucs** - Source separation for music/speech isolation
- **wav2vec2-emotion** - Emotion detection and preservation
- **SpeechBrain ECAPA** - Speaker embedding extraction

### Frontend Features
- **Voice Profile Management** - Create, test, and manage voice profiles
- **Real-time Processing** - Live progress tracking with WebSocket updates
- **Quality Metrics** - Voice quality scores and processing statistics
- **Multi-language Support** - 12+ Indian languages supported

## 📋 Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- FFmpeg
- CUDA-capable GPU (recommended)

### Quick Start with Docker
```bash
# Clone repository
git clone https://github.com/yash0539s/EduDubAI.git
cd EduDubAI

# Start with Docker Compose
docker-compose up -d

# Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

### Manual Installation

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Start backend server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🎯 Usage Guide

### 1. Create Voice Profiles
1. Navigate to "Voice Profiles" tab
2. Click "Create Profile"
3. Upload clear audio sample (minimum 10 seconds)
4. Specify speaker name, language, and gender
5. System automatically analyzes quality and creates voice embedding

### 2. Process Video with Voice Cloning
1. Go to "Video Dubbing" tab
2. Upload your video file
3. Select target language
4. Enable "Use Voice Cloning" option
5. Configure lip sync and emotion preservation
6. Start processing and monitor real-time progress

### 3. Monitor System Status
1. Check "System Status" tab for:
   - AI model loading status
   - GPU/CPU utilization
   - Voice profile statistics
   - Performance metrics

## 🔧 API Endpoints

### Voice Profile Management
- `POST /voice-profiles/create` - Create new voice profile
- `GET /voice-profiles` - List all voice profiles
- `GET /voice-profiles/{id}` - Get specific profile details
- `DELETE /voice-profiles/{id}` - Delete voice profile
- `POST /voice-profiles/{id}/test` - Test voice with sample text

### Video Processing
- `POST /upload` - Upload video for processing
- `POST /process/{job_id}` - Start dubbing process
- `GET /status/{job_id}` - Get processing status
- `GET /download/{job_id}` - Download completed video

### System Information
- `GET /health` - System health check
- `GET /languages` - Supported languages
- `GET /system/info` - Detailed system information

## 📊 Performance Metrics (PRD Targets)

| Metric | Target | Implementation |
|--------|--------|----------------|
| ASR Accuracy | ≥88% | Whisper Large-v3 |
| Translation BLEU | >42 | IndicTrans2 |
| Voice Naturalness | ≥4.2/5 MOS | XTTS-v2 |
| Lip-sync Accuracy | ≥92% | Wav2Lip |
| Processing Speed | ≤1.5x real-time | GPU acceleration |

## 🌍 Supported Languages

- **Hindi** (hi) - हिंदी
- **Bengali** (bn) - বাংলা
- **Telugu** (te) - తెలుగు
- **Marathi** (mr) - मराठी
- **Tamil** (ta) - தமிழ்
- **Gujarati** (gu) - ગુજરાતી
- **Kannada** (kn) - ಕನ್ನಡ
- **Malayalam** (ml) - മലയാളം
- **Punjabi** (pa) - ਪੰਜਾਬੀ
- **Odia** (or) - ଓଡ଼ିଆ
- **Assamese** (as) - অসমীয়া
- **Urdu** (ur) - اردو
- **English** (en)

## 🔬 Voice Cloning Technical Details

### Voice Profile Creation Process
1. **Audio Analysis**: Quality assessment, duration check, noise analysis
2. **Speaker Embedding**: ECAPA-TDNN extracts unique voice fingerprint
3. **Voice Characteristics**: Pitch, formants, speaking rate analysis
4. **Quality Scoring**: Automatic quality assessment (0-1 scale)
5. **Profile Storage**: Secure storage with metadata

### Voice Synthesis Process
1. **Text Processing**: Language-specific text normalization
2. **Emotion Mapping**: Apply detected emotions to synthesis
3. **Voice Cloning**: XTTS-v2 generates speech with target voice
4. **Prosody Matching**: Adjust timing to match original speech
5. **Quality Enhancement**: Post-processing for natural sound

### Speaker Diarization Pipeline
1. **Voice Activity Detection**: Identify speech segments
2. **Speaker Segmentation**: Split audio by speaker changes
3. **Speaker Clustering**: Group segments by speaker identity
4. **Gender Classification**: Automatic gender detection
5. **Consistency Tracking**: Maintain speaker identity across video

## 🚀 Advanced Features

### Emotion Preservation
- Real-time emotion detection during speech
- Emotion-aware voice synthesis
- Contextual emotional mapping across languages

### Multi-Speaker Handling
- Automatic speaker identification and tracking
- Individual voice profile creation for each speaker
- Consistent voice assignment throughout video

### Quality Optimization
- Automatic audio quality assessment
- Voice sample validation and filtering
- Real-time quality metrics during processing

## 🔧 Configuration

### Environment Variables
```env
# API Keys
MURF_API_KEY=your_murf_key
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_TOKEN=your_hf_token

# Model Configuration
WHISPER_MODEL=large-v3
TTS_DEVICE=cuda
WHISPER_DEVICE=cuda

# Quality Thresholds
TARGET_ASR_ACCURACY=0.88
TARGET_BLEU_SCORE=42.0
TARGET_VOICE_MOS=4.2
TARGET_LIPSYNC_ACCURACY=0.92
```

## 🐛 Troubleshooting

### Common Issues
1. **GPU Memory Issues**: Reduce batch size or use CPU fallback
2. **Voice Quality Low**: Ensure clean audio samples (>10s, low noise)
3. **Processing Slow**: Check GPU availability and model loading
4. **Language Not Supported**: Verify language code in supported list

### Performance Optimization
- Use GPU acceleration for faster processing
- Optimize audio quality for better voice cloning
- Monitor system resources during processing
- Use appropriate model sizes based on hardware

## 📈 Future Enhancements

- **Real-time Voice Cloning**: Live voice conversion during streaming
- **Advanced Lip Sync**: Improved facial animation and expression matching
- **Custom Voice Training**: User-specific voice model fine-tuning
- **Multi-modal Emotion**: Visual emotion detection integration
- **Cloud Scaling**: Distributed processing for large-scale deployment

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/voice-enhancement`)
3. Commit changes (`git commit -m "Add voice quality improvements"`)
4. Push to branch (`git push origin feature/voice-enhancement`)
5. Create Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Coqui TTS** for excellent voice cloning capabilities
- **pyannote.audio** for robust speaker diarization
- **OpenAI Whisper** for accurate speech recognition
- **AI4Bharat** for Indian language support
- **Meta Demucs** for source separation

---

**EduDub AI** - Preserving voices across languages with AI-powered dubbing technology.