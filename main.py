import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperTranscriber:
    def __init__(self, model_name="openai/whisper-tiny"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.sample_rate = 16000
        self._load_model()
    
    def _load_model(self):
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            self.model.config.forced_decoder_ids = None
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def load_audio(self, audio_path):
        try:
            audio_signal, _ = librosa.load(audio_path, sr=self.sample_rate)
            return audio_signal
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")

    def process_audio(self, audio_signal):
        return self.processor(
            audio_signal,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features

    def transcribe(self, audio_path):
        try:
            audio_signal = self.load_audio(audio_path)
            input_features = self.process_audio(audio_signal)
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )
            return "".join(transcription)
        
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, device):
        self.model.to(device)
        return self