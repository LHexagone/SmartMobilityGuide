import random
import numpy as np
import sounddevice as sd
from kokoro import KPipeline

class LocalTTS:
    def __init__(self, lang_code: str = "a", default_voice: str = "bm_george"):
        print("Loading Kokoro KPipeline into memory...")
        self.pipeline = KPipeline(lang_code=lang_code)
        self.default_voice = default_voice
        self.sample_rate = 24000  
        print("TTS Engine Ready!")

    def speak(self, text: str, voice: str = None):
        """Synthesizes text and streams chunk outputs to sounddevice."""
        selected_voice = voice or self.default_voice

        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        stream.start()

        try:
            for _, _, audio in self.pipeline(text, voice=selected_voice, speed=1.5, split_pattern=r'\n+'):
                if audio is not None:
                    if hasattr(audio, 'numpy'):
                        samples = audio.numpy().astype(np.float32)
                    else:
                        samples = np.array(audio, dtype=np.float32)

                    stream.write(samples)
        finally:
            stream.stop()
            stream.close()

    def close(self):
        pass


# --- Usage Example ---
if __name__ == "__main__":
    tts = LocalTTS()

    phrases = [
        "Hello, how are you?",
        "Have a great day!",
        "Thank you very much.",
        "See you later.",
        "It is a beautiful day.",
        "Keep up the good work.",
        "Wishing you the best.",
        "Take care of yourself.",
    ]

    try:
        while True:
            phrase = random.choice(phrases)
            tts.speak(phrase)
    except KeyboardInterrupt:
        print("\nStopping loop...")
