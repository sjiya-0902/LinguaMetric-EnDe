# 🇩🇪 Whisper German Translator Suite

A comprehensive language learning application combining English-to-German translation, pronunciation practice, and real-time fluency analysis. This suite integrates OpenAI's Whisper for speech recognition, a custom Transformer model for translation, and advanced audio processing for feedback.

## ✨ Features

### 1. 🗣️ English to German Translation
- **Dual Input:** Accepts both text input and audio recordings (via microphone or file upload).
- **AI Translation:** Uses a custom-trained Transformer model (`GerTran.keras`) to translate English to German.
- **IPA Support:** Diplays International Phonetic Alphabet (IPA) transcription for the translated German text using `epitran`.
- **Text-to-Speech:** Generates audio playback of the German translation using `gTTS`.

### 2. 🎙️ Pronunciation Practice
- **Compare & Learn:** Record yourself speaking the German translations.
- **Detailed Scoring:**
  - **Pronunciation Score:** Overall accuracy percentage.
  - **WER (Word Error Rate):** Measures word-level accuracy.
  - **PER (Phone Error Rate):** Measures phonetic accuracy.
- **Visual Feedback:** Shows your transcribed text and IPA alongside the target for comparison.

### 3. ⚡ Live Fluency Analysis
- **Real-Time Feedback:** A dedicated live mode (accessed via `http://localhost:8000`) that analyzes speech as you talk.
- **Fluency Metrics:**
  - **WPM (Words Per Minute):** Tracks speaking pace.
  - **Filler Word Detection:** Counts fillers like "um", "uh", "like", etc.
  - **Fluency Score:** A composite score (0-100) rating your overall fluency.
- **Mispronunciation Detection:** Highlights words with low confidence scores in real-time.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Dependencies:**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```

3.  **FFmpeg:**
    The project includes a local FFmpeg binary in `ffmpeg_bin`. The application is configured to automatically add this to the system PATH at runtime.

## 🚀 Usage

The easiest way to run the full suite is using the provided batch script:

1.  **Double-click `run_suite.bat`**
    
    *OR* run it from the command line:
    ```cmd
    run_suite.bat
    ```

This script will automatically:
1.  Start the **FastAPI Live Server** on `http://localhost:8000` (Backend for live fluency).
2.  Start the **Streamlit Application** on `http://localhost:8501` (Main UI).

The Streamlit interface will open in your default browser. You can access the "Live Fluency App" via the link in the "Fluency Check" tab or directly at `http://localhost:8000`.

## 📂 Project Structure

- **`app.py`**: Main Streamlit application entry point. Handles the UI for translation and practice.
- **`live_server.py`**: FastAPI server handling WebSocket connections for real-time audio transcription and analysis.
- **`run_suite.bat`**: Startup script to launch both services simultaneously.
- **`model_utils.py`**: Utilities for loading and using the custom Keras translation model.
- **`GerTran.keras`**: The pre-trained English-to-German translation model.
- **`templates/live.html`**: Frontend interface for the real-time fluency feature.

## 🤖 Technologies Used

-   **Frontend:** [Streamlit](https://streamlit.io/), HTML/JS/CSS
-   **Backend:** [FastAPI](https://fastapi.tiangolo.com/), WebSockets
-   **AI/ML:**
    -   [OpenAI Whisper](https://github.com/openai/whisper) (ASR)
    -   [TensorFlow/Keras](https://www.tensorflow.org/) (Translation Model)
    -   [Epitran](https://github.com/dmort27/epitran) (Grapheme-to-Phoneme)
-   **Audio:** [FFmpeg](https://ffmpeg.org/), [gTTS](https://gtts.readthedocs.io/), [Pydub](https://github.com/jiaaro/pydub)
