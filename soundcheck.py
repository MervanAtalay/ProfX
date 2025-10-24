import speech_recognition as sr
import threading
import time

def test_microphone():
    """Test microphone without PyAudio dependency issues."""
    
    print("="*50)
    print("🎤 MICROPHONE TEST")
    print("="*50)
    
    # Try different microphone initialization methods
    microphone = None
    
    # Method 1: Default microphone
    try:
        microphone = sr.Microphone()
        print("✅ Default microphone initialized")
    except Exception as e:
        print(f"❌ Default microphone failed: {e}")
    
    # Method 2: Try specific device
    if not microphone:
        try:
            # List available microphones
            mic_list = sr.Microphone.list_microphone_names()
            print(f"📋 Available microphones: {len(mic_list)}")
            for i, name in enumerate(mic_list[:5]):  # Show first 5
                print(f"  {i}: {name}")
            
            if mic_list:
                microphone = sr.Microphone(device_index=0)
                print(f"✅ Using microphone: {mic_list[0]}")
        except Exception as e:
            print(f"❌ Specific microphone failed: {e}")
    
    if not microphone:
        print("❌ No microphone available")
        return False
    
    # Test speech recognition
    recognizer = sr.Recognizer()
    
    # Adjust recognizer settings
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0
    
    try:
        print("\n🔧 Adjusting for background noise...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print(f"🎚️ Energy threshold: {recognizer.energy_threshold}")
        
        print("\n🎤 Please say something (you have 10 seconds):")
        print("💬 Try saying: 'Hello, this is a test'")
        
        with microphone as source:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
            print("✅ Audio captured!")
        
        # Try Google Speech Recognition
        print("🔄 Processing with Google Speech Recognition...")
        try:
            text = recognizer.recognize_google(audio)
            print(f"✅ SUCCESS! You said: '{text}'")
            return True
        except sr.UnknownValueError:
            print("❌ Could not understand the audio")
            return False
        except sr.RequestError as e:
            print(f"❌ Google API error: {e}")
            return False
            
    except sr.WaitTimeoutError:
        print("❌ No speech detected (timeout)")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def test_whisper_alternative():
    """Test with Whisper if available."""
    try:
        import whisper
        print("\n🤖 Testing Whisper...")
        
        # Load small model for testing
        model = whisper.load_model("tiny")
        print("✅ Whisper loaded successfully!")
        
        # Test with microphone
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        print("🎤 Say something for Whisper test:")
        with microphone as source:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=5)
        
        # Convert to format Whisper expects
        import numpy as np
        audio_np = np.frombuffer(
            audio.get_wav_data(), 
            dtype=np.int16
        ).astype(np.float32) / 32768.0
        
        result = model.transcribe(audio_np)
        print(f"✅ Whisper result: '{result['text']}'")
        return True
        
    except ImportError:
        print("⚠️ Whisper not available")
        return False
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        return False

def main():
    """Run comprehensive microphone test."""
    print("🎯 AI Teacher - Microphone Diagnostic Tool")
    print("="*60)
    
    # Check dependencies
    missing = []
    
    try:
        import speech_recognition
        print("✅ speech_recognition available")
    except ImportError:
        missing.append("SpeechRecognition")
    
    try:
        import pyaudio
        print("✅ pyaudio available")
        has_pyaudio = True
    except ImportError:
        print("⚠️ pyaudio not available (this might cause issues)")
        has_pyaudio = False
    
    if missing:
        print(f"❌ Missing: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return
    
    # Test microphone
    print("\n" + "="*30)
    print("MICROPHONE TEST")
    print("="*30)
    
    success = test_microphone()
    
    if success:
        print("\n🎉 MICROPHONE TEST PASSED!")
        print("✅ Your microphone is working correctly")
        print("✅ Speech recognition is functional")
        
        # Test Whisper if available
        test_whisper_alternative()
        
    else:
        print("\n❌ MICROPHONE TEST FAILED!")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check if microphone is connected")
        print("2. Check Windows microphone permissions")
        print("3. Try: pip install pyaudio (if failed, try conda)")
        print("4. Make sure no other app is using microphone")
        
        if not has_pyaudio:
            print("\n💡 PyAudio Installation Options:")
            print("Option 1: conda install pyaudio")
            print("Option 2: pip install pipwin && pipwin install pyaudio")
            print("Option 3: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()