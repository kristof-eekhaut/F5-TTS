from flask import Flask, request, send_file, jsonify
import tempfile
import uuid
import os

from importlib.resources import files
import soundfile as sf
from io import BytesIO
import socket
import struct
import torch
import torchaudio
from threading import Thread


import gc
import traceback


from infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, load_vocoder, load_model, device, target_sample_rate
from model.backbones.dit import DiT


app = Flask(__name__)


class F5TTS:
    def __init__(self, ckpt_file, vocab_file, ref_voices_dir, dtype=torch.float32):
        self.device = device
        self.target_sample_rate = target_sample_rate

        # Load the model using the provided checkpoint and vocab files
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=ckpt_file,
            mel_spec_type="vocos",  # or "bigvgan" depending on vocoder
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        ).to(self.device, dtype=dtype)

        # Load the vocoder
        self.vocoder = load_vocoder(is_local=False)

        # Set sampling rate for streaming
        self.sampling_rate = 24000  # Consistency with client

        self.ref_voices_dir = ref_voices_dir

    def generate_audio(self, text, voice_name):
        files = [f for f in os.listdir(str(self.ref_voices_dir)) if (f.endswith(".mp3") or f.endswith(".wav")) and f.startswith(voice_name)]
        ref_audio = str(self.ref_voices_dir.joinpath(files[0]))
        ref_text = open(str(self.ref_voices_dir.joinpath(voice_name + ".txt")), 'r').read()

        # Preprocess the reference audio and text
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio,ref_text)

        # Load reference audio
        audio, sr = torchaudio.load(ref_audio)

        # Run inference for the input text
        wav, sr, spect = infer_batch_process(
            (audio, sr),
            ref_text,
            [text],
            self.model,
            self.vocoder,
            device=self.device,  # Pass vocoder here
        )

        audio_buffer = BytesIO()
        sf.write(audio_buffer, wav, self.target_sample_rate, format='WAV')
        audio_buffer.seek(0)  # Reset buffer position to the start

        return audio_buffer


def generate_audio(gen_text, voice_name):
    text = gen_text.strip()

    audio_buffer = processor.generate_audio(text, voice_name)
    return audio_buffer


def warm_up(voice_name):
    print("Warming up the model...")
    gen_text = "Warm-up text for the model."
    generate_audio(gen_text, voice_name)
    print("Warm-up completed.")

@app.route('/generate_audio', methods=['POST'])
def generate_audio_api():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    gen_text = data['text']
    voice_name = data['voice']


    audio_buffer = generate_audio(gen_text, voice_name)

    return send_file(
        audio_buffer,
        as_attachment=True,
        download_name="generated_audio.wav",
        mimetype="audio/wav"
    )

if __name__ == '__main__':
    try:
        # Load the model and vocoder using the provided files
        ckpt_file = str(files("f5_tts").joinpath("../../ckpts/F5TTS_Base/model_1200000.safetensors"))
        vocab_file = ""  # Add vocab file path if needed

        ref_voices_dir = files("f5_tts").joinpath("../../voices")

        # Initialize the processor with the model and vocoder
        processor = F5TTS(
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            ref_voices_dir=ref_voices_dir,
            dtype=torch.float32,
        )

        # Start the server
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        gc.collect()
