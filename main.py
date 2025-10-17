from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Activity Detection Service")

# Enable CORS for your Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Silero VAD model on startup
logger.info("Loading Silero VAD model...")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)
(get_speech_timestamps, _, read_audio, *_) = utils
logger.info("✓ Silero VAD model loaded successfully")

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "voice-activity-detection",
        "model": "silero-vad",
        "version": "1.0"
    }

@app.post("/extract-speech")
async def extract_speech(audio: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {audio.filename}, content_type: {audio.content_type}")
        
        # Read uploaded audio
        audio_data = await audio.read()
        logger.info(f"Original file size: {len(audio_data)} bytes ({len(audio_data)/1024/1024:.2f} MB)")
        
        # Load audio with torchaudio
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))
        original_duration = waveform.shape[1] / sample_rate
        logger.info(f"Sample rate: {sample_rate}Hz, Duration: {original_duration:.2f}s ({original_duration/60:.2f} min)")
        
        # Resample to 16kHz (Silero VAD requirement)
        if sample_rate != 16000:
            logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            logger.info("Converting stereo to mono")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Get speech timestamps using Silero VAD
        logger.info("Running VAD analysis...")
        speech_timestamps = get_speech_timestamps(
            waveform[0],
            model,
            sampling_rate=sample_rate,
            threshold=0.4,              # Lower = more sensitive (good for quiet speech)
            min_speech_duration_ms=250,  # Catch short utterances
            min_silence_duration_ms=700, # Allow for drilling pauses
            window_size_samples=512,
            speech_pad_ms=50            # Padding to not cut words
        )
        
        logger.info(f"Detected {len(speech_timestamps)} speech segments")
        
        if not speech_timestamps:
            raise HTTPException(
                status_code=400, 
                detail="No speech detected in audio. Recording may contain only noise/silence."
            )
        
        # Extract and concatenate speech segments
        speech_segments = []
        total_speech_samples = 0
        
        for i, timestamp in enumerate(speech_timestamps):
            start = timestamp['start']
            end = timestamp['end']
            segment = waveform[:, start:end]
            speech_segments.append(segment)
            total_speech_samples += (end - start)
            
            segment_duration = (end - start) / sample_rate
            logger.info(f"  Speech segment {i+1}: {start/sample_rate:.2f}s - {end/sample_rate:.2f}s (duration: {segment_duration:.2f}s)")
        
        # Concatenate all speech segments
        speech_only = torch.cat(speech_segments, dim=1)
        
        # Calculate statistics
        speech_duration = speech_only.shape[1] / sample_rate
        reduction_percent = ((original_duration - speech_duration) / original_duration) * 100
        
        logger.info(f"=== RESULTS ===")
        logger.info(f"Original duration: {original_duration:.2f}s ({original_duration/60:.2f} min)")
        logger.info(f"Speech duration: {speech_duration:.2f}s ({speech_duration/60:.2f} min)")
        logger.info(f"Silence removed: {reduction_percent:.1f}%")
        logger.info(f"Speech segments: {len(speech_timestamps)}")
        
        # Export as WAV
        output = io.BytesIO()
        torchaudio.save(output, speech_only, sample_rate, format="wav")
        output.seek(0)
        
        output_size = len(output.getvalue())
        logger.info(f"Output file size: {output_size} bytes ({output_size/1024/1024:.2f} MB)")
        logger.info("✓ Processing complete")
        
        return Response(
            content=output.read(),
            media_type="audio/wav",
            headers={
                "X-Original-Duration-Sec": f"{original_duration:.2f}",
                "X-Speech-Duration-Sec": f"{speech_duration:.2f}",
                "X-Reduction-Percent": f"{reduction_percent:.1f}",
                "X-Speech-Segments": str(len(speech_timestamps)),
                "X-Original-Size-Bytes": str(len(audio_data)),
                "X-Output-Size-Bytes": str(output_size)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

