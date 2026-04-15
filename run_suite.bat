@echo off
echo ===================================================
echo 🚀 Starting Whisper German Translator Suite
echo ===================================================

echo.
echo 1. Starting Live Fluency Server (FastAPI) on port 8000...
start "Live Fluency Server" cmd /k "uvicorn live_server:app --reload --port 8000"

echo.
echo 2. Starting Streamlit App on port 8501...
streamlit run app.py

pause
