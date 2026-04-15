import epitran # epitran converts written text into how it sounds when spoken
from gtts import gTTS
from pydub import AudioSegment
from IPython.display import Audio, display, HTML
import io
import os

# Initializing epitran for German (Latin script)
epi = epitran.Epitran('deu-Latn')

def german_to_ipa(text):
    text = text.replace("[start]", "").replace("[end]", "").strip()
    ipa = epi.transliterate(text)
    return 

def german_tts_and_slow(text, lang='de', slow_factor=0.6):
    tts = gTTS(text=text, lang=lang)
    tmpfile = "/tmp/german_tts.mp3"
    tts.save(tmpfile)
    aud = AudioSegment.from_file(tmpfile, format="mp3")

    # slowed: change frame_rate to slow playback
    slowed = aud._spawn(aud.raw_data, overrides={
        "frame_rate": int(aud.frame_rate * slow_factor)
    }).set_frame_rate(aud.frame_rate)

    # export to bytes for playback
    b1 = io.BytesIO()
    b2 = io.BytesIO()
    aud.export(b1, format="wav")
    slowed.export(b2, format="wav")
    b1.seek(0); b2.seek(0)
    return b1, b2

# UI wrapper: given an English input, decode using your transformer, show German + IPA + playback
def show_pronunciation_for(input_sentence):
    german = decode_sequence(input_sentence) 
    ipa = german_to_ipa(german)
    print("Input (EN):", input_sentence)
    print("Output (DE):", german)
    print("IPA (DE):", ipa)

    # Audio Generation
    normal_wav, slowed_wav = german_tts_and_slow(german.replace("[start]", "").replace("[end]",""))
    display(HTML("<b>Play normal speed:</b>"))
    display(Audio(normal_wav.read(), autoplay=False))
    normal_wav.seek(0)
    display(HTML("<b>Play slowed (for pronunciation practice):</b>"))
    display(Audio(slowed_wav.read(), autoplay=False))