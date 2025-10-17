# Audio VAD Service

Voice Activity Detection service using Silero VAD to extract speech from noisy recordings.

## Deployment on Railway

1. Create Railway account at railway.app
2. Install Railway CLI: `npm install -g @railway/cli`
3. Login: `railway login`
4. Initialize: `railway init`
5. Deploy: `railway up`

## Testing Locally
```bash
pip install -r requirements.txt
python main.py
```

Test endpoint:
```bash
curl http://localhost:8000/
```

## API Usage
```bash
curl -X POST http://your-app.railway.app/extract-speech \
  -F "audio=@recording.wav" \
  -o cleaned.wav
```# audio-vad-service
# audio-vad-service
# audio-vad-service
# audio-vad-service
