from IPython.display import Javascript, display, Audio, HTML
import base64, os, io, subprocess, sys, json, tempfile
import numpy as np
import whisper
import soundfile as sf
from jiwer import wer # For computing Word Error Rate (WER)
import editdistance

# load whisper model
try:
    whisper_model
except NameError:
    whisper_model = whisper.load_model("small")  

# JavaScript code used in browser to record microphone audio
RECORD_JS = """
const sleep = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise((resolve) => {
  const reader = new FileReader()
  reader.onload = e => resolve(e.target.result)
  reader.readAsDataURL(blob)
})
var record = async function(sec=5){
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recorder = new MediaRecorder(stream);
  chunks = [];
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.start();
  await sleep(sec*1000);
  recorder.stop();
  await new Promise(resolve => recorder.onstop = resolve);
  blob = new Blob(chunks);
  base64 = await b2text(blob);
  return base64;
}
"""

def record_audio(seconds=5, filename="/content/recording.wav"):
    display(Javascript(RECORD_JS + f"record({seconds}).then(b => google.colab.kernel.invokeFunction('notebook.receiveRecording', [b], {{}}));"))
    # register callback to save the recording
    from google.colab import output
    def _save_b64(b64):
        header, data = b64.split(',', 1)
        b = base64.b64decode(data)
        with open(filename, "wb") as f:
            f.write(b)
    output.register_callback('notebook.receiveRecording', _save_b64)
    print(f"Recording for {seconds} seconds... Please speak into your microphone.")
 
    # wait until file exists (simple busy wait)
    import time
    timeout = seconds + 5
    start = time.time()
    while not os.path.exists(filename) and (time.time() - start) < timeout:
        time.sleep(0.1)
    if not os.path.exists(filename):
        raise RuntimeError("Recording not saved. Browser blocked mic or runtime error.")
    print("Saved:", filename)
    return filename

# text to IPA
def to_ipa(s):
    return epi.transliterate(s.replace("[start]","").replace("[end]","").strip())

# PER (phone error rate)
def phone_error_rate(ref_ipa, hyp_ipa):
    r = "".join(ref_ipa.split())
    h = "".join(hyp_ipa.split())
    if len(r) == 0:
        return 1.0 if len(h)>0 else 0.0
    dist = editdistance.eval(r, h)
    return dist / max(1, len(r))

# show prompt, play model TTS, record user, transcribe with whisper, compare
def interactive_pronunciation_task(input_sentence, record_seconds=5):
    target_german = decode_sequence(input_sentence)
    print("Target (DE):", target_german)

    target_ipa = to_ipa(target_german)
    print("Target IPA:", target_ipa)

    normal_wav, slowed_wav = german_tts_and_slow(target_german.replace("[start]","").replace("[end]",""))
    print("Playing model audio (normal speed). Listen and repeat:")
    display(Audio(normal_wav.read(), autoplay=False))
    normal_wav.seek(0)

    # recording user's attempt
    rec_file = record_audio(seconds=record_seconds, filename="/content/recording.wav")
    display(Audio(rec_file, autoplay=False))

    # transcribe user's audio with Whisper
    print("Transcribing user audio with Whisper...")
    res = whisper_model.transcribe(rec_file, language='de', task='transcribe')
    user_transcript = res.get("text", "").strip()
    print("Whisper transcript (user):", user_transcript)

    user_ipa = to_ipa(user_transcript)
    print("User IPA:", user_ipa)

    per = phone_error_rate(target_ipa, user_ipa)
    score = max(0.0, 1.0 - per) * 100.0
    print(f"Pronunciation score: {score:.1f}%  (phone error rate = {per:.3f})")

    # Computing WER (Word Eror Rate) for diagnostic:
    wer_val = wer(target_german.replace("[start]","").replace("[end]","").strip(), user_transcript)
    print(f"Word error rate (approx): {wer_val:.3f}")
    return {
        "target": target_german,
        "target_ipa": target_ipa,
        "user_transcript": user_transcript,
        "user_ipa": user_ipa,
        "score": score,
        "wer": wer_val,
        "audio_file": rec_file
    }