import os
import sys
import locale
import builtins
import io
import time

# Force UTF-8 encoding for the entire app to fix Windows issues
os.environ["PYTHONUTF8"] = "1"
def getpreferredencoding(do_setlocale=True):
    return "utf-8"
locale.getpreferredencoding = getpreferredencoding
if hasattr(locale, 'getencoding'):
    locale.getencoding = lambda: "utf-8"

import codecs
# Monkeypatch codecs.open
original_codecs_open = codecs.open
def utf8_codecs_open(*args, **kwargs):
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return original_codecs_open(*args, **kwargs)
codecs.open = utf8_codecs_open

# Monkeypatch open globally to force utf-8 - AGGRESSIVE VERSION
original_open = builtins.open
def utf8_open(*args, **kwargs):
    # Convert args to list for manipulation
    args_list = list(args)
    
    # Determine mode (position 2, index 1)
    mode = kwargs.get('mode', 'r')
    if len(args_list) > 1:
        mode = args_list[1]
    
    # Only apply UTF-8 to text mode (not binary)
    if 'b' not in mode:
        # Handle encoding parameter: open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None)
        # Position 4 (index 3) is encoding
        if len(args_list) >= 4:
            # Encoding is positional - replace it
            args_list[3] = 'utf-8'
            # Convert back to tuple
            args = tuple(args_list)
            # Don't add to kwargs to avoid duplicate
        else:
            # Encoding not in positional args, use kwargs
            kwargs['encoding'] = 'utf-8'
        
        # Set errors parameter if not already set
        if len(args_list) < 5 and 'errors' not in kwargs:
            kwargs['errors'] = 'replace'
        
    return original_open(*args, **kwargs)

builtins.open = utf8_open
io.open = utf8_open

import pandas as pd
# Monkeypatch pandas.read_csv to force utf-8 - enhanced version
original_read_csv = pd.read_csv
def utf8_read_csv(*args, **kwargs):
    # Force UTF-8 encoding
    kwargs['encoding'] = 'utf-8'
    
    # If first argument is a file handle, wrap it with UTF-8 TextIOWrapper
    if args:
        first_arg = args[0]
        # Check if it's a file handle (has read method but not a string)
        if hasattr(first_arg, 'read') and not isinstance(first_arg, str):
            import io
            # If it's a binary file handle, wrap it
            if hasattr(first_arg, 'mode') and 'b' in first_arg.mode:
                first_arg = io.TextIOWrapper(first_arg, encoding='utf-8')
                args = (first_arg,) + args[1:]
    
    return original_read_csv(*args, **kwargs)
pd.read_csv = utf8_read_csv

# Also monkeypatch pandas IO common to set default encoding
try:
    import pandas.io.common as pd_io_common
    if hasattr(pd_io_common, 'get_handle'):
        original_get_handle = pd_io_common.get_handle
        def utf8_get_handle(*args, **kwargs):
            if 'encoding' not in kwargs:
                kwargs['encoding'] = 'utf-8'
            return original_get_handle(*args, **kwargs)
        pd_io_common.get_handle = utf8_get_handle
except:
    pass

import streamlit as st
import numpy as np
import whisper
import tempfile
from gtts import gTTS
import epitran
import editdistance
from jiwer import wer
import io as _io 

# Add ffmpeg to PATH
# Update path to point to the bin folder inside the extracted directory
ffmpeg_path = os.path.abspath(os.path.join("ffmpeg_bin", "ffmpeg-8.0.1-essentials_build", "bin"))
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + ffmpeg_path

# Import our custom model utility
from model_utils import GermanTranslator

# Page Config
st.set_page_config(page_title="Whisper German Translator", page_icon="🇩🇪", layout="wide")

# --- Caching Resources ---

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_translator():
    return GermanTranslator(
        model_path="GerTran.keras",
        source_vocab_path="source_vocab_re.json",
        target_vocab_path="target_vocab_re.json"
    )

@st.cache_resource
def load_epitran():
    return epitran.Epitran('deu-Latn')

# --- Helper Functions ---

def text_to_ipa(text, epi_instance):
    # Clean text
    text = text.replace("[start]", "").replace("[end]", "").strip()
    return epi_instance.transliterate(text)

def generate_tts(text):
    text = text.replace("[start]", "").replace("[end]", "").strip()
    if not text:
        return None
    tts = gTTS(text=text, lang='de')
    fp = _io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def normalize_text(text):
    """Normalize text for WER calculation by lowercasing and removing punctuation."""
    import string
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def calculate_per(ref_ipa, hyp_ipa):
    r = "".join(ref_ipa.split())
    h = "".join(hyp_ipa.split())
    if len(r) == 0:
        return 1.0 if len(h) > 0 else 0.0
    dist = editdistance.eval(r, h)
    return dist / max(1, len(r))

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def detect_filler_words(text):
    """Detect filler words in transcribed text."""
    filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'literally', 'er', 'ah']
    text_lower = text.lower()
    count = 0
    for filler in filler_words:
        count += text_lower.count(filler)
    return count

def calculate_speaking_rate(text, duration_seconds):
    """Calculate words per minute (WPM)."""
    if duration_seconds == 0:
        return 0
    words = len(text.split())
    wpm = (words / duration_seconds) * 60
    return wpm

def calculate_fluency_score(text, duration_seconds):
    """Calculate overall fluency score based on multiple metrics."""
    if duration_seconds == 0 or not text.strip():
        return {
            'wpm': 0,
            'filler_count': 0,
            'filler_percentage': 0,
            'fluency_score': 0,
            'rating': 'N/A'
        }
    
    # Calculate metrics
    wpm = calculate_speaking_rate(text, duration_seconds)
    filler_count = detect_filler_words(text)
    word_count = len(text.split())
    filler_percentage = (filler_count / max(1, word_count)) * 100
    
    # Optimal WPM range: 130-170
    # Score WPM (0-50 points)
    if 130 <= wpm <= 170:
        wpm_score = 50
    elif 100 <= wpm < 130 or 170 < wpm <= 200:
        wpm_score = 35
    elif 70 <= wpm < 100 or 200 < wpm <= 230:
        wpm_score = 20
    else:
        wpm_score = 10
    
    # Score filler words (0-50 points)
    # Less than 2% filler words is excellent
    if filler_percentage < 2:
        filler_score = 50
    elif filler_percentage < 5:
        filler_score = 35
    elif filler_percentage < 10:
        filler_score = 20
    else:
        filler_score = 10
    
    # Total fluency score (0-100)
    fluency_score = wpm_score + filler_score
    
    # Rating
    if fluency_score >= 85:
        rating = 'Excellent'
    elif fluency_score >= 70:
        rating = 'Good'
    elif fluency_score >= 50:
        rating = 'Fair'
    else:
        rating = 'Needs Improvement'
    
    return {
        'wpm': round(wpm, 1),
        'filler_count': filler_count,
        'filler_percentage': round(filler_percentage, 2),
        'fluency_score': fluency_score,
        'rating': rating
    }

# --- Main App ---

st.title("🇩🇪 Learn how to speak German")
st.markdown("Translate English to German and practice your pronunciation!")

# Load models
with st.spinner("Loading models..."):
    whisper_model = None
    translator = None
    epi = None
    
    try:
        whisper_model = load_whisper_model()
    except Exception as e:
        st.error(f"Error loading Whisper: {e}")
        st.stop()
        
    try:
        translator = load_translator()
    except Exception as e:
        st.error(f"Error loading Translator: {e}")
        st.stop()
        
    try:
        epi = load_epitran()
    except Exception as e:
        import traceback
        st.error(f"Error loading Epitran: {e}")
        st.code(traceback.format_exc())
        st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["Translate", "Practice", "Fluency Check"])

# --- Tab 1: Translate ---
with tab1:
    st.header("English to German Translation")
    
    # Initialize session state for english text if not exists
    if 'english_text' not in st.session_state:
        st.session_state['english_text'] = ""

    input_method = st.radio("Input Method", ["Text", "Audio"], horizontal=True)
    
    if input_method == "Text":
        st.session_state['english_text'] = st.text_area("Enter English Text", value=st.session_state['english_text'], height=100)
    else:
        st.markdown("### Record or Upload Audio")
        col_audio1, col_audio2 = st.columns(2)
        
        with col_audio1:
            audio_input = st.audio_input("Record English Audio")
        
        with col_audio2:
            uploaded_audio = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])
            
        # Determine which audio source to use
        audio_source = audio_input if audio_input else uploaded_audio
        
        if audio_source:
            st.audio(audio_source)
            if st.button("Transcribe Audio"):
                with st.spinner("Transcribing..."):
                    # Save to temp file for Whisper
                    suffix = os.path.splitext(audio_source.name)[1] if hasattr(audio_source, 'name') else ".wav"
                    if not suffix: suffix = ".wav"
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(audio_source.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        result = whisper_model.transcribe(tmp_path, language="en")
                        st.session_state['english_text'] = result["text"]
                        st.success("Transcription Complete!")
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                            
        if st.session_state['english_text']:
            st.text_area("Transcribed Text", value=st.session_state['english_text'], disabled=True)

    if st.button("Translate"):
        english_text = st.session_state['english_text']
        if english_text.strip():
            with st.spinner("Translating..."):
                # 1. Translate
                german_translation = translator.decode_sequence(english_text)
                clean_german = german_translation.replace("[start]", "").replace("[end]", "").strip()
                
                # 2. IPA
                ipa_text = text_to_ipa(clean_german, epi)
                
                # 3. TTS
                tts_audio = generate_tts(clean_german)
                
                # Display Results
                st.subheader("Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("*German Translation:*")
                    st.info(clean_german)
                
                with col2:
                    st.markdown("*IPA Pronunciation:*")
                    st.code(ipa_text)
                
                if tts_audio:
                    st.markdown("*Listen to Pronunciation:*")
                    st.audio(tts_audio, format="audio/mp3")
                    
                # Store in session state for Practice tab
                st.session_state['last_translation'] = clean_german
                st.session_state['last_ipa'] = ipa_text
        else:
            st.warning("Please enter some text or record/upload audio first.")

# --- Tab 2: Practice ---
with tab2:
    st.header("Pronunciation Practice")
    
    if 'last_translation' in st.session_state:
        target_text = st.session_state['last_translation']
        target_ipa = st.session_state['last_ipa']
        
        st.markdown(f"*Target Text:* {target_text}")
        st.markdown(f"*Target IPA:* {target_ipa}")
        
        st.markdown("---")
        st.markdown("### Record your pronunciation")
        
        practice_audio = st.audio_input("Record yourself speaking the German text above")
        
        if practice_audio:
            st.audio(practice_audio)
            
            if st.button("Analyze Pronunciation"):
                with st.spinner("Analyzing..."):
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(practice_audio.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Transcribe with Whisper (German)
                    result = whisper_model.transcribe(tmp_path, language="de")
                    user_transcript = result["text"]
                    
                    # Generate IPA for user transcript
                    user_ipa = text_to_ipa(user_transcript, epi)
                    
                    # Calculate Metrics
                    per = calculate_per(target_ipa, user_ipa)
                    # Normalize both texts before WER calculation for fair comparison
                    normalized_target = normalize_text(target_text)
                    normalized_transcript = normalize_text(user_transcript)
                    wer_val = wer(normalized_target, normalized_transcript)
                    score = max(0.0, 1.0 - per) * 100.0
                    
                    # Display Results
                    st.subheader("Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Pronunciation Score", f"{score:.1f}%")
                        st.metric("Phone Error Rate (PER)", f"{per:.3f}")
                    with col2:
                        st.metric("Word Error Rate (WER)", f"{wer_val:.3f}")
                    
                    st.markdown("*Your Transcript:*")
                    st.text(user_transcript)
                    
                    st.markdown("*Your IPA:*")
                    st.code(user_ipa)
                    
                    os.remove(tmp_path)
                    
    else:
        st.info("Please translate something in the 'Translate' tab first to practice it!")

# --- Tab 3: Fluency Check ---
with tab3:
    st.header("English Fluency Analysis")
    
    # Link to Live App
    st.info("Real-Time Live Fluency Mode")
    st.link_button("Open Live Fluency App 🎙️", "http://localhost:8000")
    st.markdown("---")
    
    st.markdown("Or record yourself speaking below (Standard Mode):")
    
    fluency_audio = st.audio_input("Record your English speech")
    
    if fluency_audio:
        st.audio(fluency_audio)
        
        if st.button("Analyze Fluency"):
            # Create a placeholder for progressive updates
            results_container = st.container()
            
            with st.spinner("Transcribing full audio..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(fluency_audio.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Transcribe with Whisper (English)
                    result = whisper_model.transcribe(tmp_path, language="en")
                    full_transcript = result["text"]
                    segments = result.get("segments", [])
                    
                    # Display full transcript first
                    with results_container:
                        st.subheader("Full Transcript")
                        st.write(full_transcript)
                        st.markdown("---")
                        st.subheader("Segment Analysis")
                    
                    # Process in 10-second chunks
                    chunk_duration = 10.0
                    chunks = []
                    current_chunk = {"text": "", "start": 0, "end": 0}
                    
                    # Group segments into chunks
                    for segment in segments:
                        seg_start = segment['start']
                        seg_end = segment['end']
                        seg_text = segment['text']
                        
                        chunk_index = int(seg_start // chunk_duration)
                        chunk_start = chunk_index * chunk_duration
                        chunk_end = (chunk_index + 1) * chunk_duration
                        
                        if current_chunk["text"] and seg_start >= current_chunk["end"]:
                            chunks.append(current_chunk)
                            current_chunk = {"text": "", "start": chunk_start, "end": chunk_end}
                        
                        if not current_chunk["text"]:
                            current_chunk["start"] = chunk_start
                            current_chunk["end"] = chunk_end
                        current_chunk["text"] += seg_text + " "
                    
                    if current_chunk["text"]:
                        chunks.append(current_chunk)
                    
                    if not chunks:
                        total_duration = result.get("duration", 10.0)
                        chunks = [{"text": full_transcript, "start": 0, "end": total_duration}]
                    
                    # Display results
                    for i, chunk in enumerate(chunks):
                        chunk_text = chunk["text"].strip()
                        chunk_duration_sec = chunk["end"] - chunk["start"]
                        
                        # Calculate metrics
                        fluency_metrics = calculate_fluency_score(chunk_text, chunk_duration_sec)
                        
                        # Display immediately
                        with results_container:
                            with st.expander(f"Segment {i+1} ({chunk['start']:.1f}s - {chunk['end']:.1f}s)", expanded=True):
                                st.text(f"Text: {chunk_text}")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Speaking Rate", f"{fluency_metrics['wpm']} WPM")
                                with col2:
                                    st.metric("Filler Words", f"{fluency_metrics['filler_count']}")
                                with col3:
                                    st.metric("Fluency Score", f"{fluency_metrics['fluency_score']}/100")
                                    st.caption(f"Rating: {fluency_metrics['rating']}")
                        
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
