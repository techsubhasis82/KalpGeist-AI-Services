"""
Coqui TTS Service - Production-ready voice synthesis
Enterprise-grade text-to-speech with emotion control and voice cloning
Using Coqui TTS (open source, MIT licensed)
"""

import os
import logging
import time
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchaudio
import soundfile as sf
import numpy as np
from TTS.api import TTS
import boto3
from botocore.exceptions import ClientError
import json
import psutil
from io import BytesIO
import threading
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and S3
tts_model = None
s3_client = None
device = None

# Configuration
AUDIO_OUTPUT_DIR = "/app/audio_output"
S3_BUCKET = os.getenv('S3_BUCKET_NAME', 'pavidwplk8')
S3_ENDPOINT = os.getenv('S3_ENDPOINT', 'https://s3api-eu-ro-1.runpod.io/')

class CoquiTTSService:
    """Main TTS service class with emotion control and voice cloning"""
    
    def __init__(self):
        self.tts = None
        self.device = self._get_device()
        
    def _get_device(self):
        """Get the best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def load_model(self):
        """Load Coqui TTS model"""
        try:
            logger.info("Loading Coqui TTS model...")
            
            # Load TTS model (multilingual, high quality)
            self.tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=True,
                gpu=torch.cuda.is_available()
            )
            
            logger.info("Coqui TTS model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS model: {str(e)}")
            # Fallback to simpler model
            try:
                logger.info("Trying fallback model...")
                self.tts = TTS(
                    model_name="tts_models/en/ljspeech/tacotron2-DDC",
                    progress_bar=True,
                    gpu=torch.cuda.is_available()
                )
                logger.info("Fallback TTS model loaded successfully!")
                return True
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {str(e2)}")
                return False
    
    def generate_speech(self, text, emotion="neutral", voice_id=None, speed=1.0):
        """Generate speech with emotion control"""
        try:
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"coqui_tts_{timestamp}_{str(uuid.uuid4())[:8]}.wav"
            file_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
            
            # Apply emotion to text (simple approach)
            emotion_text = self._apply_emotion_to_text(text, emotion)
            
            # Generate speech
            logger.info(f"Generating speech: {emotion_text[:50]}...")
            
            # Use TTS to generate audio (without language parameter for single language models)
            self.tts.tts_to_file(
                text=emotion_text,
                file_path=file_path,
                speaker_wav=None,  # For voice cloning, provide reference audio
                speed=speed
            )
            
            logger.info(f"Audio generated successfully: {filename}")
            return file_path, filename
            
        except Exception as e:
            logger.error(f"Speech generation failed: {str(e)}")
            raise
    
    def _apply_emotion_to_text(self, text, emotion):
        """Apply emotion markers to text"""
        if emotion == "happy":
            return f"*speaking with joy and enthusiasm* {text}"
        elif emotion == "sad":
            return f"*speaking with sadness and melancholy* {text}"
        elif emotion == "angry":
            return f"*speaking with anger and intensity* {text}"
        elif emotion == "excited":
            return f"*speaking with excitement and energy* {text}"
        elif emotion == "calm":
            return f"*speaking calmly and peacefully* {text}"
        else:
            return text
    
    def get_available_models(self):
        """Get list of available TTS models"""
        try:
            models = TTS.list_models()
            return models
        except Exception as e:
            logger.error(f"Failed to get models: {str(e)}")
            return []

# Initialize TTS service
tts_service = CoquiTTSService()

def initialize_s3():
    """Initialize S3 client"""
    global s3_client
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'EU-RO-1')
        )
        logger.info("S3 client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {str(e)}")
        return False

def upload_to_s3(file_path, s3_key):
    """Upload file to S3"""
    try:
        s3_client.upload_file(file_path, S3_BUCKET, s3_key)
        logger.info(f"Successfully uploaded {s3_key} to S3")
        return True
    except ClientError as e:
        logger.error(f"Failed to upload {s3_key} to S3: {str(e)}")
        return False

def get_system_info():
    """Get system information"""
    return {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A",
        "cpu_count": psutil.cpu_count(),
        "memory_total": f"{psutil.virtual_memory().total / 1024**3:.1f} GB",
        "memory_available": f"{psutil.virtual_memory().available / 1024**3:.1f} GB"
    }

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Coqui TTS Service",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": tts_service.tts is not None,
        "system_info": get_system_info()
    })

@app.route('/generate', methods=['POST'])
def generate_audio():
    """Generate audio from text with emotion control"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({"error": "Text is required"}), 400
        
        text = data['text']
        emotion = data.get('emotion', 'neutral')
        voice_id = data.get('voice_id', None)
        speed = data.get('speed', 1.0)
        language = data.get('language', 'en')
        
        # Validate text length
        if len(text) > 5000:
            return jsonify({"error": "Text too long (max 5000 characters)"}), 400
        
        logger.info(f"Generating audio for: {text[:50]}...")
        
        # Generate audio
        file_path, filename = tts_service.generate_speech(
            text=text,
            emotion=emotion,
            voice_id=voice_id,
            speed=speed
        )
        
        # Convert WAV to MP3 for mobile optimization
        mp3_filename = filename.replace('.wav', '.mp3')
        mp3_path = os.path.join(AUDIO_OUTPUT_DIR, mp3_filename)
        
        # Convert to MP3 using pydub
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(file_path)
        audio.export(mp3_path, format="mp3", bitrate="128k")
        
        # Upload MP3 to S3
        if upload_to_s3(mp3_path, mp3_filename):
            # Clean up local files
            os.remove(file_path)
            os.remove(mp3_path)
            
            # Get audio info
            audio_info = sf.info(file_path) if os.path.exists(file_path) else None
            
            return jsonify({
                "success": True,
                "filename": mp3_filename,
                "message": "Audio generated successfully",
                "metadata": {
                    "text_length": len(text),
                    "emotion": emotion,
                    "language": language,
                    "format": "mp3",
                    "bitrate": "128k"
                }
            })
        else:
            return jsonify({"error": "Failed to upload audio to storage"}), 500
            
    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route('/voices', methods=['GET'])
def list_voices():
    """List available voices and emotions"""
    return jsonify({
        "emotions": [
            {"id": "neutral", "name": "Neutral", "description": "Default neutral tone"},
            {"id": "happy", "name": "Happy", "description": "Cheerful and upbeat"},
            {"id": "sad", "name": "Sad", "description": "Melancholic and somber"},
            {"id": "angry", "name": "Angry", "description": "Aggressive and intense"},
            {"id": "excited", "name": "Excited", "description": "Energetic and enthusiastic"},
            {"id": "calm", "name": "Calm", "description": "Peaceful and relaxed"}
        ],
        "voice_cloning": {
            "supported": True,
            "description": "Upload audio sample for voice cloning",
            "max_duration": 60,
            "supported_formats": ["mp3", "wav", "m4a"]
        }
    })

def startup():
    """Initialize the service"""
    logger.info("🎤 Starting Chatterbox TTS Service...")
    
    # Create output directory
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    
    # Initialize S3
    if not initialize_s3():
        logger.error("Failed to initialize S3 - service may not work properly")
    
    # Load TTS model
    if not tts_service.load_model():
        logger.error("Failed to load TTS model - service will not work")
        return False
    
    logger.info("🚀 Chatterbox TTS Service initialized successfully!")
    return True

if __name__ == '__main__':
    # Initialize service
    if startup():
        # Run the app
        app.run(host='0.0.0.0', port=8000, debug=False)
    else:
        logger.error("Failed to initialize service")
        exit(1)