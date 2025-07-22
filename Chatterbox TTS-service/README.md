# 🎤 Chatterbox TTS Service

Enterprise-grade text-to-speech service with emotion control and voice cloning capabilities.

## Features

- **Emotion Control**: Happy, sad, angry, excited, calm, neutral
- **Voice Cloning**: Zero-shot voice cloning from audio samples
- **High Quality**: Production-ready audio synthesis
- **GPU Accelerated**: RTX 5090 optimized for fast generation
- **S3 Integration**: Seamless file storage and retrieval
- **RESTful API**: Easy integration with KalpGeist AI app

## API Endpoints

### Health Check
```
GET /health
```

### Generate Audio
```
POST /generate
{
    "text": "Hello, this is a test message!",
    "emotion": "happy",
    "voice_id": "optional_voice_id",
    "speed": 1.0
}
```

### List Voices
```
GET /voices
```

## Supported Emotions

- `neutral` - Default neutral tone
- `happy` - Cheerful and upbeat
- `sad` - Melancholic and somber
- `angry` - Aggressive and intense
- `excited` - Energetic and enthusiastic
- `calm` - Peaceful and relaxed

## Docker Deployment

```bash
# Build the image
docker build -t chatterbox-tts-service .

# Run the container
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e S3_BUCKET_NAME=pavidwplk8 \
  chatterbox-tts-service
```

## RunPod Deployment

1. Push Docker image to registry
2. Create RunPod serverless endpoint
3. Configure environment variables
4. Deploy with RTX 5090 GPU

## Performance

- **Processing Time**: 1-2 seconds per sentence
- **Audio Quality**: 22kHz sample rate
- **Output Format**: MP3 (optimized for mobile)
- **Concurrent Requests**: Up to 10 simultaneous generations

## Cost Efficiency

- **Per Request**: ~$0.005-0.01 (serverless)
- **Storage**: Shared S3 bucket with video service
- **GPU Usage**: Only during generation (pay-per-use)

## Integration with KalpGeist AI

This service integrates with your existing infrastructure:
- Same S3 bucket as video service
- Same Railway download server
- Same environment variables
- Compatible with your Android app architecture

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

## License

MIT License - Free for commercial use