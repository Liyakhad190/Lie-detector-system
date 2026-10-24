import threading

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False


class VoiceAnalyzer:
    """
    Handles speech-to-text recognition to transcribe questions
    asked by the interviewer in the background.
    """
    def __init__(self):
        self.is_listening = False
        self.current_question = ""
        self._stop_listening_fn = None
        self.error_msg = ""
        self.speech_available = SPEECH_AVAILABLE
        
        if not self.speech_available:
            self.error_msg = "SpeechRecognition not installed. Run: pip install -r requirements.txt"
            return
            
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            # Adjust for ambient noise in background so __init__ is non-blocking
            def _calibrate():
                try:
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                except Exception as e:
                    self.error_msg = f"Mic Error: {e}"
                    self.speech_available = False
            t = threading.Thread(target=_calibrate, daemon=True)
            t.start()
        except Exception as e:
            self.error_msg = f"Mic Error: {e}"
            self.speech_available = False

    def start_listening(self):
        """Start listening for the question in the background."""
        if not self.speech_available:
            self.current_question = self.error_msg
            return

        # Cancel any previous listening session so T can be pressed repeatedly
        self.stop_listening()

        self.is_listening = True
        self.current_question = "Listening... Speak question into microphone now!"

        def callback(recognizer, audio):
            try:
                # Transcribe using Google Web Speech API
                text = recognizer.recognize_google(audio)
                self.current_question = text
                print(f"[VOICE] Transcribed Question: {text}")
            except sr.UnknownValueError:
                self.current_question = "[No speech heard. Speak closer to mic & press T to retry]"
            except sr.RequestError as e:
                self.current_question = f"[Speech API offline: {e}]"
            except Exception as e:
                self.current_question = f"[Voice error: {e}]"
            finally:
                self.is_listening = False

        try:
            # Enable dynamic energy threshold to adapt to microphone volume
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            # Non-blocking listen (listens until pause in speech, up to 10 seconds)
            self._stop_listening_fn = self.recognizer.listen_in_background(
                self.microphone, callback, phrase_time_limit=10
            )
        except Exception as e:
            self.current_question = f"[Mic Error: {e}]"
            self.is_listening = False

    def stop_listening(self):
        """Stop listening manually."""
        if self._stop_listening_fn:
            try:
                self._stop_listening_fn(wait_for_stop=False)
            except Exception:
                pass
            self._stop_listening_fn = None
        self.is_listening = False

    def get_question(self):
        return self.current_question
        
    def reset(self):
        self.stop_listening()
        self.current_question = ""
