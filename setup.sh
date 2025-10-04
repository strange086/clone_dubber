#!/bin/bash

# EduDub AI Setup Script
# Sprint 5: Voice Cloning & Speaker Tracking System

echo "🎙️ Setting up EduDub AI - Voice Cloning Platform"
echo "================================================="

# Check if Python 3.9+ is installed
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
if [[ $(echo "$python_version >= 3.9" | bc -l) -eq 0 ]]; then
    echo "❌ Python 3.9+ required. Current version: $python_version"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg is required but not installed."
    echo "Please install FFmpeg: https://ffmpeg.org/download.html"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Setup backend
echo ""
echo "🔧 Setting up backend..."
cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads outputs temp voice_profiles
echo "✅ Backend directories created"

# Setup environment file
if [ ! -f ".env" ]; then
    echo "🔑 Creating environment file..."
    cat > .env << EOL
# API Keys (replace with your actual keys)
MURF_API_KEY=your_murf_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
WHISPER_API_KEY=your_whisper_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Application Configuration
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
TTS_DEVICE=cuda

# File Storage
MAX_FILE_SIZE=524288000
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
TEMP_DIR=temp

# Quality Thresholds
TARGET_ASR_ACCURACY=0.88
TARGET_BLEU_SCORE=42.0
TARGET_VOICE_MOS=4.2
TARGET_LIPSYNC_ACCURACY=0.92
EOL
    echo "✅ Environment file created (.env)"
    echo "⚠️  Please edit .env file with your actual API keys"
fi

cd ..

# Setup frontend
echo ""
echo "🎨 Setting up frontend..."
cd frontend

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

cd ..

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit backend/.env with your API keys"
echo "2. Start the backend: cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
echo "3. Start the frontend: cd frontend && npm run dev"
echo "4. Open http://localhost:5173 in your browser"
echo ""
echo "🐳 Alternative: Use Docker Compose"
echo "   docker-compose up -d"
echo ""
echo "📚 Documentation:"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Voice Cloning Guide: README_VOICE_CLONING.md"
echo ""
echo "🎙️ Voice Cloning Features:"
echo "   ✅ Artist voice preservation across languages"
echo "   ✅ Multi-speaker diarization and tracking"
echo "   ✅ Emotion detection and preservation"
echo "   ✅ High-quality voice synthesis with XTTS-v2"
echo "   ✅ Real-time processing with progress tracking"
echo ""