from flask import Flask, request, send_file, jsonify
import tempfile
import uuid
import os

from importlib.resources import files
import soundfile as sf
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
    def __init__(self, ckpt_file, vocab_file, ref_audio, ref_text, temp_out_dir, dtype=torch.float32):
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

        # Set reference audio and text
        self.ref_audio = ref_audio
        self.ref_text = ref_text

        # Set audio ouput dir
        self.temp_out_dir = temp_out_dir

        # Warm up the model
        self._warm_up()

    def _warm_up(self):
        """Warm up the model with a dummy input to ensure it's ready for real-time processing."""
        print("Warming up the model...")
        ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio, self.ref_text)
        audio, sr = torchaudio.load(ref_audio)
        gen_text = "Warm-up text for the model."

        # Pass the vocoder as an argument here
        infer_batch_process((audio, sr), ref_text, [gen_text], self.model, self.vocoder, device=self.device)
        print("Warm-up completed.")

    def export_wav(self, wav, file_wave):
        sf.write(file_wave, wav, self.target_sample_rate)

    def generate_audio(self, text):
        # Preprocess the reference audio and text
        ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio, self.ref_text)

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

        audio_filename = f"{uuid.uuid4()}.wav"
        out_file = str(self.temp_out_dir.joinpath(audio_filename))
        self.export_wav(wav, out_file)

        return out_file;




# Add your TTS generation code here as a function
def generate_audio(gen_text):
    text = gen_text.strip()

    audio_path = processor.generate_audio(text)
    return audio_path

@app.route('/generate_audio', methods=['POST'])
def generate_audio_api():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    gen_text = data['text']

    try:
        audio_path = generate_audio(gen_text)

        if not os.path.exists(audio_path):
            return jsonify({"error": "Failed to generate audio"}), 500

        return send_file(audio_path, as_attachment=True)
    finally:
        # Clean up: remove the audio file after sending the response
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == '__main__':
    try:
        # Load the model and vocoder using the provided files
        ckpt_file = str(files("f5_tts").joinpath("../../ckpts/F5TTS_Base/model_1200000.safetensors"))
        vocab_file = ""  # Add vocab file path if needed
        ref_audio = str(files("f5_tts").joinpath("../../voices/example_cassandra.mp3"))  # add ref audio"./tests/ref_audio/reference.wav"

        temp_out_dir = files("f5_tts").joinpath("../../temp/")

        # Initialize the processor with the model and vocoder
        processor = F5TTS(
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            ref_audio=ref_audio,
            ref_text=ref_text,
            temp_out_dir=temp_out_dir,
            dtype=torch.float32,
        )

        # Start the server
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        gc.collect()
