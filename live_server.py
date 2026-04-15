import os
import sys
import json
import asyncio
import tempfile
import whisper
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Initialize FastAPI
app = FastAPI()

# Mount templates
templates = Jinja2Templates(directory="templates")

# Load Whisper Model (Global)
print("Loading Whisper model...")
model = whisper.load_model("small")
print("Whisper model loaded!")

def detect_filler_words(text):
    """Detect filler words in transcribed text."""
    filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'literally', 'er', 'ah']
    text_lower = text.lower()
    count = 0
    for filler in filler_words:
        count += text_lower.count(filler)
    return count

def calculate_fluency_score(text, duration_seconds):
    """Calculate overall fluency score."""
    if duration_seconds == 0 or not text.strip():
        return {
            'wpm': 0,
            'filler_count': 0,
            'fluency_score': 0,
            'rating': 'N/A'
        }
    
    word_count = len(text.split())
    wpm = (word_count / duration_seconds) * 60
    filler_count = detect_filler_words(text)
    filler_percentage = (filler_count / max(1, word_count)) * 100
    
    # WPM Score
    if 130 <= wpm <= 170: wpm_score = 50
    elif 100 <= wpm < 130 or 170 < wpm <= 200: wpm_score = 35
    elif 70 <= wpm < 100 or 200 < wpm <= 230: wpm_score = 20
    else: wpm_score = 10
    
    # Filler Score
    if filler_percentage < 2: filler_score = 50
    elif filler_percentage < 5: filler_score = 35
    elif filler_percentage < 10: filler_score = 20
    else: filler_score = 10
    
    fluency_score = wpm_score + filler_score
    
    if fluency_score >= 85: rating = 'Excellent'
    elif fluency_score >= 70: rating = 'Good'
    elif fluency_score >= 50: rating = 'Fair'
    else: rating = 'Needs Improvement'
    
    return {
        'wpm': round(wpm, 1),
        'filler_count': filler_count,
        'fluency_score': fluency_score,
        'rating': rating
    }

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("live.html", {"request": request})

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    try:
        while True:
            # Receive audio blob
            data = await websocket.receive_bytes()
            print(f"Received audio chunk size: {len(data)} bytes")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
                tmp_file.write(data)
                tmp_path = tmp_file.name
            
            try:
                # Transcribe
                print("Starting transcription...")
                # Note: Whisper handles various audio formats via ffmpeg
                
                try:
                    # Attempt with word_timestamps for mispronunciation detection
                    result = model.transcribe(tmp_path, language="en", word_timestamps=True)
                except TypeError:
                     # Fallback for older whisper versions
                     print("Warning: word_timestamps not supported. Falling back to standard transcription.")
                     result = model.transcribe(tmp_path, language="en")
                except Exception as e:
                     # Fallback for other errors (e.g. alignment failure)
                     print(f"Transcription error with timestamps: {e}. Falling back...")
                     result = model.transcribe(tmp_path, language="en")

                text = result["text"]
                print(f"Transcription result: {text[:50]}...")
                
                # Calculate metrics
                # Whisper result has 'segments' with start/end and logprob
                segments = result.get('segments', [])
                
                duration = segments[-1].get('end', 0) if segments else 0
                if duration == 0: duration = 10.0 # Fallback
                
                # --- Word-Level Analysis for Mispronunciation ---
                words_analysis = []
                low_conf_threshold = 0.6
                
                all_logprobs = []
                
                for segment in segments:
                    # Collect segment-level logprobs for overall pronunciation score
                    all_logprobs.append(segment.get('avg_logprob', -1.0))
                    
                    # Process individual words if available
                    for word_info in segment.get('words', []):
                        word_text = word_info.get('word', '').strip()
                        # probability is exp(logprob) NOT stored directly in 'words' usually, 
                        # but 'probability' key might be present depending on whisper version/wrapper.
                        # If not, we might rely on segment avg_logprob, but 'words' usually has 'probability' in newer whisper.
                        # Let's check keys. Standard openai-whisper 'words' has: word, start, end, probability.
                        
                        word_conf = float(word_info.get('probability', 0.0))
                        
                        words_analysis.append({
                            'word': word_text,
                            'confidence': round(word_conf, 2),
                            'is_mispronounced': bool(word_conf < low_conf_threshold)
                        })

                # Calculate Pronunciation Score (Overall Confidence)
                if all_logprobs:
                    avg_prob = np.mean([np.exp(lp) for lp in all_logprobs])
                    pronunciation_score = round(float(avg_prob) * 100, 1)
                else:
                    pronunciation_score = 0
                
                metrics = calculate_fluency_score(text, duration)
                metrics['pronunciation_score'] = pronunciation_score
                
                response = {
                    "text": text,
                    "metrics": metrics,
                    "words": words_analysis
                }
                
                await websocket.send_json(response)
                print("Sent response back to client")
                
            except Exception as e:
                print(f"Error processing audio: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({"error": str(e)})
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
