#//Source Code//
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
import numpy as np
import os
import subprocess

print("Loading TTS model...")

try:
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    embeddings_path = "static/speaker_embeddings.npy"
    if os.path.exists(embeddings_path):
        emb = np.load(embeddings_path)
        emb = np.asarray(emb).flatten()[:512]
        speaker_embeddings = torch.from_numpy(emb).float().unsqueeze(0)
    else:
        speaker_embeddings = torch.randn(1, 512)
    
    use_neural = True
    print("Neural TTS Ready!")
except Exception as e:
    print(f"Neural TTS failed: {e}, using gTTS")
    use_neural = False

def text_to_speech(text, filename=None, use_gtts=False):
    if not text or not text.strip():
        raise ValueError("No text to speak")
    
    if filename is None:
        filename = "static/audio/output.wav"
    
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)
    
    if use_gtts or not use_neural:
        from gtts import gTTS
        text = text.strip()
        if not text:
            raise ValueError("No text to speak")
        filename = filename.replace('.wav', '.mp3')
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        return filename
    
    inputs = processor(text=text, return_tensors="pt")

    speech = model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings,
        vocoder=vocoder
    )

    sf.write(filename, speech.numpy(), samplerate=16000)

    return filename
