from faster_whisper import WhisperModel
from time import time
import logging
import os
import warnings

# Suppress pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

AUDIO_PATH = r"C:\Users\basti\PycharmProjects\Sophy4Lite\Voys-founder Mark Vletter_ 'AI kan geweldig systemen bouwen, maar dat kost véél moeite'.mp3"
OUTPUT_PATH = r"C:\Users\basti\Desktop\transcriptie.txt"

def format_time(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02}:{m:02}:{sec:02}.{ms:03}"

def transcribe_audio():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not os.path.isfile(AUDIO_PATH):
        logger.error(f"Audio file not found: {AUDIO_PATH}")
        return
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.isdir(output_dir):
        logger.error(f"Output directory not found: {output_dir}")
        return

    # Load model
    start_load = time()
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        logger.info(f"Model loaded in {time() - start_load:.1f} sec")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Transcribe audio
    start_trans = time()
    try:
        segments, info = model.transcribe(AUDIO_PATH, language=None, vad_filter=True)
        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return

    # Write transcription
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start_time = format_time(seg.start)
                line = f"[{start_time}] {seg.text.strip()}"
                logger.info(f"[{i}] {line}")
                f.write(line + "\n")
        logger.info(f"Transcription saved to {OUTPUT_PATH} in {time() - start_trans:.1f} sec")
    except Exception as e:
        logger.error(f"Failed to write transcription: {e}")
        return

if __name__ == "__main__":
    transcribe_audio()