import cv2
import numpy as np
from deepface import DeepFace
import speech_recognition as sr
import pyttsx3
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import os
from collections import deque
import sqlite3
from datetime import datetime
import queue

# Optional imports with error handling
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    print("⚠️ librosa not available. Voice analysis will be simplified.")
    LIBROSA_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("⚠️ Whisper not available. Using Google Speech Recognition only.")
    WHISPER_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    print("⚠️ PyAudio not available. Using default microphone.")
    PYAUDIO_AVAILABLE = False

class VoiceAnalyzer:
    """Analyze voice for emotional state and confidence."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.chunk_size = 1024
        
    def analyze_voice_emotion(self, audio_data):
        """Analyze voice for emotion and confidence indicators."""
        try:
            if not LIBROSA_AVAILABLE:
                # Fallback to simple analysis
                return self.simple_voice_analysis(audio_data)
            
            # Convert audio to features
            y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Extract features
            features = {}
            
            # Pitch analysis
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=self.sample_rate)
                pitch = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
                features['pitch'] = pitch
            except:
                features['pitch'] = 150  # Default pitch
            
            # Energy/Volume
            try:
                energy = np.mean(librosa.feature.rms(y=y))
                features['energy'] = energy[0][0] if energy.size > 0 else 0.1
            except:
                features['energy'] = 0.1
            
            # Speech rate (zero crossing rate)
            try:
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                features['speech_rate'] = zcr[0][0] if zcr.size > 0 else 0.1
            except:
                features['speech_rate'] = 0.1
            
            # Spectral features
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)
                features['spectral_centroid'] = np.mean(spectral_centroids)
            except:
                features['spectral_centroid'] = 1500
            
            # Classify emotional state based on features
            emotion_state = self.classify_voice_emotion(features)
            
            return emotion_state
            
        except Exception as e:
            print(f"Voice analysis error: {e}")
            return {'confidence': 0.5, 'emotion': 'neutral', 'clarity': 0.5}
    
    def simple_voice_analysis(self, audio_data):
        """Simple voice analysis without librosa."""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate basic features
            energy = np.mean(np.abs(audio_array)) / 32768.0
            
            # Simple classification
            if energy > 0.1:
                confidence = min(1.0, energy * 3)
                emotion = 'confident'
                clarity = 0.8
            elif energy < 0.05:
                confidence = 0.3
                emotion = 'uncertain'
                clarity = 0.4
            else:
                confidence = 0.6
                emotion = 'neutral'
                clarity = 0.6
            
            return {
                'confidence': confidence,
                'emotion': emotion,
                'clarity': clarity,
                'features': {'energy': energy}
            }
        except Exception as e:
            print(f"Simple voice analysis error: {e}")
            return {'confidence': 0.5, 'emotion': 'neutral', 'clarity': 0.5}
    
    def classify_voice_emotion(self, features):
        """Classify emotional state from voice features."""
        confidence = 0.5
        emotion = 'neutral'
        clarity = 0.5
        
        try:
            # Simple rule-based classification
            if features['energy'] > 0.1 and features['pitch'] > 200:
                confidence = min(1.0, features['energy'] * 5)
                emotion = 'confident'
            elif features['energy'] < 0.05:
                confidence = 0.2
                emotion = 'uncertain'
            
            # Speech clarity based on spectral centroid
            if features['spectral_centroid'] > 2000:
                clarity = 0.8
            elif features['spectral_centroid'] < 1000:
                clarity = 0.3
            else:
                clarity = 0.6
        except Exception as e:
            print(f"Voice classification error: {e}")
            
        return {
            'confidence': confidence,
            'emotion': emotion,
            'clarity': clarity,
            'features': features
        }

class SpeechEngine:
    """Handle speech recognition and text-to-speech."""
    
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
        except Exception as e:
            print(f"⚠️ Microphone initialization error: {e}")
            self.microphone = None
        
        # Initialize text-to-speech
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Speaking rate
            self.tts_engine.setProperty('volume', 0.8)
            
            # Set female voice if available
            voices = self.tts_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
        except Exception as e:
            print(f"⚠️ TTS initialization error: {e}")
            self.tts_engine = None
        
        # Initialize Whisper for better recognition
        self.use_whisper = False
        if WHISPER_AVAILABLE:
            try:
                print("Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
                self.use_whisper = True
                print("✅ Whisper loaded successfully!")
            except Exception as e:
                print(f"⚠️ Whisper initialization error: {e}")
                self.use_whisper = False
        
        # Audio queue for processing
        self.audio_queue = queue.Queue()
        self.voice_analyzer = VoiceAnalyzer()
        
    def listen_for_speech(self, timeout=8, phrase_timeout=3):
        """Listen for speech and return text + voice analysis."""
        if not self.microphone:
            return {'text': '', 'voice_analysis': None, 'success': False, 'error': 'no_microphone'}
        
        try:
            # Adjust for ambient noise
            with self.microphone as source:
                print("Adjusting for background noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
            print("🎤 Listening for student response...")
            
            with self.microphone as source:
                # Listen for speech
                audio_data = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_timeout
                )
                
            print("🔄 Processing speech...")
            
            # Convert speech to text
            text = ""
            method = "none"
            
            # Try Whisper first (best accuracy)
            if self.use_whisper:
                try:
                    audio_np = np.frombuffer(
                        audio_data.get_wav_data(), 
                        dtype=np.int16
                    ).astype(np.float32) / 32768.0
                    
                    result = self.whisper_model.transcribe(audio_np, fp16=False)
                    text = result["text"].strip()
                    method = "whisper"
                    print(f"✅ Whisper result: '{text}'")
                except Exception as e:
                    print(f"⚠️ Whisper error: {e}")
            
            # Fallback to Google Speech Recognition
            if not text:
                try:
                    text = self.recognizer.recognize_google(audio_data)
                    method = "google"
                    print(f"✅ Google result: '{text}'")
                except sr.UnknownValueError:
                    print("❌ Could not understand audio")
                except sr.RequestError as e:
                    print(f"❌ Google API error: {e}")
            
            # Analyze voice characteristics
            voice_analysis = None
            if text:
                try:
                    voice_analysis = self.voice_analyzer.analyze_voice_emotion(audio_data.get_wav_data())
                    voice_analysis['method'] = method
                except Exception as e:
                    print(f"Voice analysis error: {e}")
            
            return {
                'text': text,
                'voice_analysis': voice_analysis,
                'success': bool(text),
                'method': method
            }
            
        except sr.WaitTimeoutError:
            print("⏰ No speech detected (timeout)")
            return {'text': '', 'voice_analysis': None, 'success': False, 'error': 'timeout'}
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            return {'text': '', 'voice_analysis': None, 'success': False, 'error': str(e)}
    
    def speak_text(self, text):
        """Convert text to speech."""
        try:
            print(f"🤖 AI Teacher: {text}")
            
            if self.tts_engine:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                print("[TTS not available - text only]")
                
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            print(f"[SPEECH]: {text}")

class LessonManager:
    """Manage lessons, topics, and questions."""
    
    def __init__(self):
        self.lessons_db = self.initialize_lessons()
        self.current_lesson = None
        self.current_topic_index = 0
        self.topic_attempt_count = 0
        self.max_attempts = 2
        
    def initialize_lessons(self):
        """Initialize lesson database."""
        lessons = {
            "mathematics": {
                "title": "Basic Mathematics",
                "topics": [
                    {
                        "title": "Addition",
                        "explanation": "Addition is combining two or more numbers to get their sum. For example, 2 plus 3 equals 5.",
                        "questions": [
                            {"question": "What is 5 plus 3?", "answer": "8", "alternatives": ["eight"]},
                            {"question": "If you have 4 apples and get 2 more, how many do you have?", "answer": "6", "alternatives": ["six"]}
                        ]
                    },
                    {
                        "title": "Subtraction", 
                        "explanation": "Subtraction is taking away one number from another. For example, 10 minus 3 equals 7.",
                        "questions": [
                            {"question": "What is 10 minus 4?", "answer": "6", "alternatives": ["six"]},
                            {"question": "If you have 8 cookies and eat 3, how many are left?", "answer": "5", "alternatives": ["five"]}
                        ]
                    }
                ]
            },
            "science": {
                "title": "Basic Science",
                "topics": [
                    {
                        "title": "Water Cycle",
                        "explanation": "The water cycle shows how water moves around Earth. Water evaporates from oceans, forms clouds, and comes back as rain.",
                        "questions": [
                            {"question": "What happens when water evaporates?", "answer": "it becomes water vapor", "alternatives": ["vapor", "gas", "steam"]},
                            {"question": "What do we call water falling from clouds?", "answer": "rain", "alternatives": ["precipitation"]}
                        ]
                    }
                ]
            }
        }
        return lessons
    
    def get_available_lessons(self):
        """Get list of available lessons."""
        return list(self.lessons_db.keys())
    
    def start_lesson(self, lesson_key):
        """Start a specific lesson."""
        if lesson_key in self.lessons_db:
            self.current_lesson = lesson_key
            self.current_topic_index = 0
            self.topic_attempt_count = 0
            return True
        return False
    
    def get_current_topic(self):
        """Get current topic to teach."""
        if not self.current_lesson:
            return None
            
        lesson = self.lessons_db[self.current_lesson]
        if self.current_topic_index >= len(lesson["topics"]):
            return None  # Lesson completed
            
        return lesson["topics"][self.current_topic_index]
    
    def move_to_next_topic(self):
        """Move to next topic in lesson."""
        self.current_topic_index += 1
        self.topic_attempt_count = 0
        
    def retry_current_topic(self):
        """Increment attempt count for current topic."""
        self.topic_attempt_count += 1
        
    def should_re_explain(self):
        """Check if topic should be re-explained."""
        return self.topic_attempt_count >= self.max_attempts
    
    def get_random_question(self, topic):
        """Get a random question from topic."""
        if not topic or 'questions' not in topic:
            return None
        import random
        return random.choice(topic['questions'])

class AILanguageModel:
    """Handle AI responses and evaluation."""
    
    def __init__(self):
        # Simple rule-based responses (no OpenAI needed)
        self.responses = {
            'encouragement': [
                "Great job! You're doing wonderful!",
                "Excellent answer! You understand this well!",
                "Perfect! You've got it right!"
            ],
            'gentle_correction': [
                "That's a good try! Let me help you understand better.",
                "You're close! Let's think about this together.",
                "Not quite, but you're on the right track!"
            ],
            'confusion_detected': [
                "I can see you might be a bit confused. That's okay!",
                "Let me explain this in a different way.",
                "Don't worry, let's break this down step by step."
            ]
        }
    
    def evaluate_answer(self, question_data, student_answer, voice_analysis, face_emotion):
        """Evaluate student's answer considering multiple factors."""
        if not question_data or not student_answer:
            return {'correct': False, 'confidence': 0}
        
        # Check answer correctness
        correct_answer = question_data['answer'].lower()
        alternatives = [alt.lower() for alt in question_data.get('alternatives', [])]
        student_lower = student_answer.lower().strip()
        
        is_correct = (correct_answer in student_lower or 
                     any(alt in student_lower for alt in alternatives))
        
        # Evaluate confidence based on voice and face
        overall_confidence = 0.5
        
        if voice_analysis:
            voice_confidence = voice_analysis.get('confidence', 0.5)
            voice_clarity = voice_analysis.get('clarity', 0.5)
            overall_confidence = (voice_confidence + voice_clarity) / 2
        
        # Adjust confidence based on face emotion
        if face_emotion:
            if face_emotion in ['confident', 'focused']:
                overall_confidence = min(1.0, overall_confidence * 1.2)
            elif face_emotion in ['confused', 'anxiety']:
                overall_confidence = max(0.1, overall_confidence * 0.8)
        
        return {
            'correct': is_correct,
            'confidence': overall_confidence,
            'voice_analysis': voice_analysis,
            'face_emotion': face_emotion
        }
    
    def generate_response(self, evaluation_result, topic_title):
        """Generate appropriate AI teacher response."""
        import random
        
        if evaluation_result['correct'] and evaluation_result['confidence'] > 0.6:
            return random.choice(self.responses['encouragement'])
        elif evaluation_result['correct'] and evaluation_result['confidence'] <= 0.6:
            return "That's correct! You seem a bit uncertain, but you got it right. Well done!"
        elif not evaluation_result['correct'] and evaluation_result['confidence'] > 0.6:
            return "You sound confident, but that's not quite right. Let me help you."
        else:
            return random.choice(self.responses['gentle_correction'])
    
    def generate_re_explanation(self, topic):
        """Generate a re-explanation of the topic."""
        base_explanation = topic.get('explanation', '')
        return f"Let me explain {topic['title']} again in a different way. {base_explanation} Do you understand this better now?"

class StudentProgressTracker:
    """Track student progress and performance."""
    
    def __init__(self):
        self.db_path = "student_progress.db"
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for progress tracking."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lesson_name TEXT,
                    topic_name TEXT,
                    question TEXT,
                    answer TEXT,
                    correct BOOLEAN,
                    confidence REAL,
                    face_emotion TEXT,
                    voice_emotion TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"⚠️ Database initialization error: {e}")
    
    def log_interaction(self, lesson_name, topic_name, question, answer, evaluation):
        """Log student interaction."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sessions (lesson_name, topic_name, question, answer, correct, confidence, face_emotion, voice_emotion)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                lesson_name,
                topic_name, 
                question,
                answer,
                evaluation['correct'],
                evaluation['confidence'],
                evaluation.get('face_emotion', ''),
                evaluation.get('voice_analysis', {}).get('emotion', '') if evaluation.get('voice_analysis') else ''
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ Database logging error: {e}")

class AITeacherGUI:
    """Main GUI for AI Teacher application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Virtual Teacher")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.emotion_detector = self.init_emotion_detector()
        self.speech_engine = SpeechEngine()
        self.lesson_manager = LessonManager()
        self.ai_model = AILanguageModel()
        self.progress_tracker = StudentProgressTracker()
        
        # State variables
        self.teaching_active = False
        self.current_face_emotion = "neutral"
        self.lesson_thread = None
        
        self.create_gui()
        self.start_emotion_detection()
        
    def init_emotion_detector(self):
        """Initialize emotion detection from existing code."""
        class SimpleEmotionDetector:
            def __init__(self):
                self.cap = None
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.current_emotion = "analyzing"
                
            def detect_emotion(self):
                if not self.cap or not self.cap.isOpened():
                    return "no_camera"
                    
                ret, frame = self.cap.read()
                if not ret:
                    return "no_frame"
                
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    
                    try:
                        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
                        if isinstance(result, list):
                            emotions = result[0]['emotion']
                        else:
                            emotions = result['emotion']
                        
                        dominant_emotion = max(emotions, key=emotions.get)
                        
                        # Map to exam emotions
                        exam_emotions = {
                            'happy': 'confident',
                            'neutral': 'focused', 
                            'sad': 'confused',
                            'angry': 'frustrated',
                            'fear': 'anxiety',
                            'surprise': 'curious',
                            'disgust': 'bored'
                        }
                        
                        self.current_emotion = exam_emotions.get(dominant_emotion, dominant_emotion)
                        return self.current_emotion
                        
                    except Exception as e:
                        return "analyzing"
                
                return "no_face"
            
            def start_camera(self):
                try:
                    self.cap = cv2.VideoCapture(0)
                    return self.cap.isOpened()
                except Exception as e:
                    print(f"Camera error: {e}")
                    return False
            
            def stop_camera(self):
                if self.cap:
                    self.cap.release()
        
        return SimpleEmotionDetector()
    
    def create_gui(self):
        """Create the main GUI interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🤖 AI Virtual Teacher", 
                               font=("Arial", 20, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Lesson selection
        ttk.Label(control_frame, text="Select Lesson:", font=("Arial", 12)).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.lesson_var = tk.StringVar()
        lesson_combo = ttk.Combobox(control_frame, textvariable=self.lesson_var, 
                                   values=self.lesson_manager.get_available_lessons(), 
                                   state="readonly", width=20)
        lesson_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        lesson_combo.set("mathematics")  # Default selection
        
        # Control buttons
        self.start_btn = ttk.Button(control_frame, text="🎓 Start Teaching", 
                                   command=self.start_teaching)
        self.start_btn.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop Teaching", 
                                  command=self.stop_teaching, state='disabled')
        self.stop_btn.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Status indicators
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding="10")
        status_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        
        ttk.Label(status_frame, text="Emotion:").grid(row=0, column=0, sticky=tk.W)
        self.emotion_label = ttk.Label(status_frame, text="analyzing...", 
                                      font=("Arial", 10, "bold"), foreground="blue")
        self.emotion_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(status_frame, text="Teaching:").grid(row=1, column=0, sticky=tk.W)
        self.teaching_status = ttk.Label(status_frame, text="Stopped", 
                                        font=("Arial", 10, "bold"), foreground="red")
        self.teaching_status.grid(row=1, column=1, sticky=tk.W)
        
        # Middle panel - Conversation
        conv_frame = ttk.LabelFrame(main_frame, text="AI Teacher Conversation", padding="10")
        conv_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        conv_frame.columnconfigure(0, weight=1)
        conv_frame.rowconfigure(0, weight=1)
        
        # Conversation display
        self.conversation_text = scrolledtext.ScrolledText(conv_frame, wrap=tk.WORD, 
                                                          height=25, width=60,
                                                          font=("Arial", 11))
        self.conversation_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input frame
        input_frame = ttk.Frame(conv_frame)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        input_frame.columnconfigure(0, weight=1)
        
        self.manual_input = ttk.Entry(input_frame, font=("Arial", 11))
        self.manual_input.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.manual_input.bind('<Return>', self.on_manual_input)
        
        ttk.Button(input_frame, text="Send", command=self.on_manual_input).grid(row=0, column=1)
        
        # Right panel - Progress
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Progress display
        self.progress_text = scrolledtext.ScrolledText(progress_frame, wrap=tk.WORD, 
                                                      height=25, width=40,
                                                      font=("Arial", 10))
        self.progress_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.add_to_conversation("🤖 AI Teacher", "Hello! I'm your AI teacher. Select a lesson and click 'Start Teaching' to begin!")
        self.add_to_progress("System ready. Waiting for lesson to start...")
    
    def add_to_conversation(self, speaker, message):
        """Add message to conversation display."""
        try:
            self.conversation_text.config(state='normal')
            
            # Color coding
            if speaker.startswith("🤖"):
                color = "blue"
            elif speaker.startswith("👤"):
                color = "green"
            else:
                color = "black"
            
            # Configure tags
            self.conversation_text.tag_configure(color, foreground=color, font=("Arial", 11, "bold"))
            
            current_time = datetime.now().strftime("%H:%M:%S")
            self.conversation_text.insert(tk.END, f"[{current_time}] {speaker}: ", color)
            self.conversation_text.insert(tk.END, f"{message}\n\n")
            
            self.conversation_text.config(state='disabled')
            self.conversation_text.see(tk.END)
        except Exception as e:
            print(f"GUI update error: {e}")
    
    def add_to_progress(self, message):
        """Add message to progress display."""
        try:
            self.progress_text.config(state='normal')
            current_time = datetime.now().strftime("%H:%M:%S")
            self.progress_text.insert(tk.END, f"[{current_time}] {message}\n")
            self.progress_text.config(state='disabled')
            self.progress_text.see(tk.END)
        except Exception as e:
            print(f"Progress update error: {e}")
    
    def start_emotion_detection(self):
        """Start emotion detection in background."""
        def detect_emotions():
            if self.emotion_detector.start_camera():
                while True:
                    try:
                        emotion = self.emotion_detector.detect_emotion()
                        self.current_face_emotion = emotion
                        
                        # Update GUI safely
                        try:
                            self.root.after(0, lambda: self.emotion_label.config(text=emotion))
                        except:
                            pass  # GUI might be closed
                        
                        time.sleep(0.5)  # Check twice per second
                    except Exception as e:
                        print(f"Emotion detection error: {e}")
                        time.sleep(1)
            else:
                try:
                    self.root.after(0, lambda: self.emotion_label.config(text="Camera Error"))
                except:
                    pass
        
        emotion_thread = threading.Thread(target=detect_emotions, daemon=True)
        emotion_thread.start()
    
    def start_teaching(self):
        """Start AI teaching session."""
        lesson_name = self.lesson_var.get()
        if not lesson_name:
            messagebox.showwarning("Warning", "Please select a lesson first!")
            return
        
        if not self.lesson_manager.start_lesson(lesson_name):
            messagebox.showerror("Error", "Failed to start lesson!")
            return
        
        self.teaching_active = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.teaching_status.config(text="Active", foreground="green")
        
        self.add_to_conversation("🤖 AI Teacher", f"Starting {lesson_name.title()} lesson!")
        self.add_to_progress(f"Lesson started: {lesson_name}")
        
        # Start teaching in separate thread
        self.lesson_thread = threading.Thread(target=self.teaching_loop, daemon=True)
        self.lesson_thread.start()
    
    def stop_teaching(self):
        """Stop AI teaching session."""
        self.teaching_active = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.teaching_status.config(text="Stopped", foreground="red")
        
        self.add_to_conversation("🤖 AI Teacher", "Teaching session ended. Great work today!")
        self.add_to_progress("Teaching session stopped")
    
    def teaching_loop(self):
        """Main teaching loop."""
        while self.teaching_active:
            topic = self.lesson_manager.get_current_topic()
            
            if not topic:
                # Lesson completed
                self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", 
                    "Congratulations! You've completed this lesson!"))
                self.root.after(0, self.stop_teaching)
                break
            
            # Teach current topic
            self.teach_topic(topic)
            
            if not self.teaching_active:
                break
        
    def teach_topic(self, topic):
        """Teach a specific topic."""
        try:
            # Explain the topic
            explanation = topic['explanation']
            self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", 
                f"Let's learn about {topic['title']}. {explanation}"))
            self.root.after(0, lambda: self.add_to_progress(f"Teaching topic: {topic['title']}"))
            
            # Speak the explanation
            self.speech_engine.speak_text(f"Let's learn about {topic['title']}. {explanation}")
            
            time.sleep(2)  # Give time for explanation to be processed
            
            # Ask question
            question_data = self.lesson_manager.get_random_question(topic)
            if not question_data:
                self.lesson_manager.move_to_next_topic()
                return
            
            question = question_data['question']
            self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", f"Question: {question}"))
            self.speech_engine.speak_text(question)
            
            # Wait for student response
            self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", "Please answer (I'm listening...)"))
            
            # Get speech input
            speech_result = self.speech_engine.listen_for_speech(timeout=10)
            
            if speech_result['success'] and speech_result['text']:
                student_answer = speech_result['text']
                self.root.after(0, lambda: self.add_to_conversation("👤 Student", student_answer))
                
                # Evaluate answer
                evaluation = self.ai_model.evaluate_answer(
                    question_data, 
                    student_answer, 
                    speech_result['voice_analysis'],
                    self.current_face_emotion
                )
                
                # Log interaction
                self.progress_tracker.log_interaction(
                    self.lesson_manager.current_lesson,
                    topic['title'],
                    question,
                    student_answer,
                    evaluation
                )
                
                # Generate response
                response = self.ai_model.generate_response(evaluation, topic['title'])
                self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", response))
                self.speech_engine.speak_text(response)
                
                # Decision making
                if evaluation['correct'] and evaluation['confidence'] > 0.5:
                    # Student understood, move to next topic
                    self.root.after(0, lambda: self.add_to_progress(f"✅ Topic mastered: {topic['title']}"))
                    self.lesson_manager.move_to_next_topic()
                else:
                    # Student needs help
                    self.lesson_manager.retry_current_topic()
                    
                    if self.lesson_manager.should_re_explain():
                        # Re-explain the topic
                        re_explanation = self.ai_model.generate_re_explanation(topic)
                        self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", re_explanation))
                        self.speech_engine.speak_text(re_explanation)
                        self.root.after(0, lambda: self.add_to_progress(f"🔄 Re-explaining: {topic['title']}"))
                        self.lesson_manager.move_to_next_topic()  # Move on after re-explanation
                    else:
                        self.root.after(0, lambda: self.add_to_progress(f"🤔 Retrying: {topic['title']}"))
            
            else:
                # No response or unclear
                error_msg = speech_result.get('error', 'unknown')
                if error_msg == 'timeout':
                    msg = "I didn't hear anything. Let me ask again."
                elif error_msg == 'unclear':
                    msg = "I didn't understand your answer clearly. Let me ask again."
                else:
                    msg = "I had trouble hearing you. Let me ask again."
                
                self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", msg))
                self.speech_engine.speak_text(msg)
                # Don't move topic, will retry same question
                
        except Exception as e:
            print(f"Teaching error: {e}")
            self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", 
                "I encountered an issue. Let's continue with the next topic."))
    
    def on_manual_input(self, event=None):
        """Handle manual text input."""
        text = self.manual_input.get().strip()
        if text:
            self.add_to_conversation("👤 Student (Manual)", text)
            self.manual_input.delete(0, tk.END)
    
    def run(self):
        """Run the application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Application interrupted")
        finally:
            self.emotion_detector.stop_camera()

def main():
    """Main function to run AI Teacher."""
    print("="*60)
    print("AI Virtual Teacher - Multi-Modal Education Platform")
    print("="*60)
    print("Features:")
    print("- Face emotion detection")
    print("- Voice analysis and speech recognition")
    print("- Adaptive teaching based on student understanding")
    print("- Progress tracking")
    print("- Interactive lessons")
    print("="*60)
    
    # Check core dependencies
    missing_deps = []
    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import deepface
        print("✅ DeepFace available")
    except ImportError:
        missing_deps.append("deepface")
    
    try:
        import speech_recognition
        print("✅ SpeechRecognition available")
    except ImportError:
        missing_deps.append("SpeechRecognition")
    
    try:
        import pyttsx3
        print("✅ pyttsx3 available")
    except ImportError:
        missing_deps.append("pyttsx3")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        print("Please install: pip install " + " ".join(missing_deps))
        return
    
    print("✅ All core dependencies available")
    
    # Optional dependencies
    if not WHISPER_AVAILABLE:
        print("💡 For better speech recognition, install: pip install openai-whisper")
    
    if not LIBROSA_AVAILABLE:
        print("💡 For advanced voice analysis, install: pip install librosa soundfile")
    
    if not PYAUDIO_AVAILABLE:
        print("💡 For better audio handling, install: pip install pyaudio")
    
    print("="*60)
    
    # Start application
    app = AITeacherGUI()
    app.run()

if __name__ == "__main__":
    main()