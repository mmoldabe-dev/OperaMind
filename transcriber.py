import speech_recognition as sr
from pydub import AudioSegment
import os

def transcribe_audio_file(filepath):
    try:
        print(f"📤 Загружаю: {filepath}")
        
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        duration_min = len(audio) / 60000
        print(f"⏱️ Длительность: {duration_min:.1f} мин")
        
        recognizer = sr.Recognizer()
        temp_wav = "temp.wav"
        audio.export(temp_wav, format="wav")
        
        with sr.AudioFile(temp_wav) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language='ru-RU')
        
        os.remove(temp_wav)
        print(f"✅ Готово: {len(text)} символов")
        return text
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

def transcribe_from_text(text):
    return text.strip()