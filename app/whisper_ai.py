import whisper
from app.utils import logger

def load_model(model_type="base"):
    """Load Whisper model based on configuration."""
    try:
        logger.info(f"Loading Whisper model: {model_type}")
        model = whisper.load_model(model_type)
        return model
    except Exception as e:
        logger.error(e)

def transcribe_audio(model, audio_path):
    """Transcribe the given audio file."""
    try:
        logger.info(f"Transcribing audio: {audio_path}")
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        logger.error(e)
