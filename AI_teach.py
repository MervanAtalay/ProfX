import cv2
import numpy as np
from deepface import DeepFace
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

# Re-enable pyttsx3 for text-to-speech
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
    print("✅ pyttsx3 available for text-to-speech")
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("⚠️ pyttsx3 not available. Install with: pip install pyttsx3")

# Google Gemini API
try:
    from google import genai
    GEMINI_AVAILABLE = True
    print("✅ Google Gemini available")
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Google Gemini not available. Install with: pip install google-genai")

# Language translations
TRANSLATIONS = {
    'en': {
        'title': '🤖 AI Virtual Teacher',
        'controls': 'Controls',
        'select_lesson': 'Select Lesson:',
        'select_language': 'Language:',
        'start_teaching': '🎓 Start Teaching',
        'stop_teaching': '⏹️ Stop Teaching',
        'status': 'Status',
        'emotion': 'Emotion:',
        'teaching': 'Teaching:',
        'conversation': 'AI Teacher Conversation',
        'progress': 'Progress',
        'send': 'Send',
        'analyzing': 'analyzing...',
        'stopped': 'Stopped',
        'active': 'Active',
        'welcome': "Hello! I'm your AI teacher. Let me get to know you first!",
        'system_ready': 'System ready. Please set up your profile to begin...',
        'warning': 'Warning',
        'select_lesson_first': 'Please select a lesson first!',
        'error': 'Error',
        'failed_start': 'Failed to start lesson!',
        'lesson_started': 'Lesson started',
        'starting_lesson': 'Starting {} lesson!',
        'session_ended': 'Teaching session ended. You can still chat with me about education!',
        'session_stopped': 'Teaching session stopped',
        'lesson_completed': "Congratulations! You've completed this lesson! Feel free to ask me questions.",
        'listening': "Please type your answer below:",  # Changed for text input
        'no_response': "Please provide an answer.",  # Changed for text input
        'unclear': "I didn't understand your answer clearly. Let me ask again.",
        'trouble_hearing': "I had trouble hearing you. Let me ask again.",
        'issue': "I encountered an issue. Let's continue with the next topic.",
        'student': 'Student',
        'manual': 'Manual',
        'teaching_topic': 'Teaching topic',
        'topic_mastered': 'Topic mastered',
        're_explaining': 'Re-explaining',
        'retrying': 'Retrying',
        'waiting_answer': 'Waiting for your answer...',
        'type_answer': 'Type your answer and press Enter or click Send',
        'student_profile': 'Student Profile',
        'name': 'Name:',
        'age': 'Age:',
        'grade': 'Grade:',
        'save_profile': 'Save Profile',
        'profile_saved': 'Profile saved!',
        'enter_name': 'Please enter your name',
        'enter_age': 'Please enter your age',
        'enter_grade': 'Please enter your grade',
        'chat_active': 'Chat with me about anything educational!',
        'off_topic': "I'm your education teacher! Let's keep our conversation focused on learning. What would you like to learn about?"
    },
    'tr': {
        'title': '🤖 Yapay Zeka Öğretmen',
        'controls': 'Kontroller',
        'select_lesson': 'Ders Seç:',
        'select_language': 'Dil:',
        'start_teaching': '🎓 Öğretimi Başlat',
        'stop_teaching': '⏹️ Öğretimi Durdur',
        'status': 'Durum',
        'emotion': 'Duygu:',
        'teaching': 'Öğretim:',
        'conversation': 'Yapay Zeka Öğretmen Konuşması',
        'progress': 'İlerleme',
        'send': 'Gönder',
        'analyzing': 'analiz ediliyor...',
        'stopped': 'Durduruldu',
        'active': 'Aktif',
        'welcome': "Merhaba! Ben senin yapay zeka öğretmeninim. Önce seni tanıyayım!",
        'system_ready': 'Sistem hazır. Lütfen profilini oluştur...',
        'warning': 'Uyarı',
        'select_lesson_first': 'Lütfen önce bir ders seçin!',
        'error': 'Hata',
        'failed_start': 'Ders başlatılamadı!',
        'lesson_started': 'Ders başladı',
        'starting_lesson': '{} dersi başlıyor!',
        'session_ended': 'Öğretim oturumu sona erdi. Benimle eğitim hakkında konuşmaya devam edebilirsin!',
        'session_stopped': 'Öğretim oturumu durduruldu',
        'lesson_completed': 'Tebrikler! Bu dersi tamamladın! Bana soru sorabilirsin.',
        'listening': 'Lütfen cevabını aşağıya yaz:',  # Changed for text input
        'no_response': 'Lütfen bir cevap ver.',  # Changed for text input
        'unclear': 'Cevabını net anlayamadım. Tekrar sorayım.',
        'trouble_hearing': 'Seni duymakta zorlandım. Tekrar sorayım.',
        'issue': 'Bir sorunla karşılaştım. Bir sonraki konuya geçelim.',
        'student': 'Öğrenci',
        'manual': 'Manuel',
        'teaching_topic': 'Konu öğretiliyor',
        'topic_mastered': 'Konu öğrenildi',
        're_explaining': 'Tekrar açıklanıyor',
        'retrying': 'Tekrar deneniyor',
        'waiting_answer': 'Cevabın bekleniyor...',
        'type_answer': 'Cevabını yaz ve Enter\'a bas veya Gönder\'e tıkla',
        'student_profile': 'Öğrenci Profili',
        'name': 'İsim:',
        'age': 'Yaş:',
        'grade': 'Sınıf:',
        'save_profile': 'Profili Kaydet',
        'profile_saved': 'Profil kaydedildi!',
        'enter_name': 'Lütfen adınızı girin',
        'enter_age': 'Lütfen yaşınızı girin',
        'enter_grade': 'Lütfen sınıfınızı girin',
        'chat_active': 'Benimle eğitimle ilgili her şey hakkında konuş!',
        'off_topic': "Ben senin eğitim öğretmeninim! Konuşmamızı öğrenmeye odaklayalım. Ne öğrenmek istersin?"
    }
}

# class VoiceAnalyzer:
#     """Analyze voice for emotional state and confidence."""
#     # ...existing code... (all commented out)

class SpeechEngine:
    """Handle text display with text-to-speech (voice input disabled)."""
    
    def __init__(self, language='en'):
        self.language = language
        self.tts_engine = None
        
        # Initialize TTS engine
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                self.set_voice_for_language(language)
                print("✅ Text-to-Speech initialized")
            except Exception as e:
                print(f"⚠️ TTS initialization error: {e}")
                self.tts_engine = None
        
        print("✅ Text-only input mode initialized (voice input disabled)")
    
    def set_language(self, language):
        """Set the language."""
        self.language = language
        if self.tts_engine:
            self.set_voice_for_language(language)
    
    def set_voice_for_language(self, language):
        """Set TTS voice based on language."""
        if not self.tts_engine:
            return
            
        try:
            voices = self.tts_engine.getProperty('voices')
            if not voices:
                return
            
            # Voice selection based on language
            if language == 'tr':
                # Try to find Turkish voice
                for voice in voices:
                    if 'turkish' in voice.name.lower() or 'tr' in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        print(f"✅ Turkish voice set: {voice.name}")
                        return
            
            # Default to English female voice
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    print(f"✅ Voice set: {voice.name}")
                    return
        except Exception as e:
            print(f"Voice setting error: {e}")
    
    def get_language_code(self):
        """Get language code."""
        lang_codes = {
            'en': 'en-US',
            'tr': 'tr-TR'
        }
        return lang_codes.get(self.language, 'en-US')
    
    def speak_text(self, text):
        """Convert text to speech."""
        try:
            print(f"🤖 AI Teacher: {text}")
            
            if self.tts_engine:
                self.tts_engine.stop()
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                time.sleep(0.3)
            else:
                print("[TTS not available]")
                
        except Exception as e:
            print(f"❌ TTS Error: {e}")

class LessonManager:
    """Manage lessons, topics, and questions with multi-language support."""
    
    def __init__(self, language='en'):
        self.language = language
        self.lessons_db = self.initialize_lessons()
        self.current_lesson = None
        self.current_topic_index = 0
        self.topic_attempt_count = 0
        self.max_attempts = 2
    
    def set_language(self, language):
        """Change the language."""
        self.language = language
        self.lessons_db = self.initialize_lessons()
        
    def initialize_lessons(self):
        """Initialize lesson database with translations."""
        if self.language == 'tr':
            return {
                "matematik": {
                    "title": "Temel Matematik",
                    "topics": [
                        {
                            "title": "Toplama",
                            "explanation": "Toplama, iki veya daha fazla sayıyı birleştirerek toplamlarını bulmaktır. Örneğin, 2 artı 3 eşittir 5.",
                            "questions": [
                                {"question": "5 artı 3 kaç eder?", "answer": "8", "alternatives": ["sekiz"]},
                                {"question": "4 elmanız varsa ve 2 tane daha alırsanız, kaç elmanız olur?", "answer": "6", "alternatives": ["altı"]}
                            ]
                        },
                        {
                            "title": "Çıkarma",
                            "explanation": "Çıkarma, bir sayıdan başka bir sayıyı çıkarmaktır. Örneğin, 10 eksi 3 eşittir 7.",
                            "questions": [
                                {"question": "10 eksi 4 kaç eder?", "answer": "6", "alternatives": ["altı"]},
                                {"question": "8 kurabiyeniz varsa ve 3 tanesini yerseniz, kaç kurabiye kalır?", "answer": "5", "alternatives": ["beş"]}
                            ]
                        }
                    ]
                },
                "fen": {
                    "title": "Temel Fen Bilgisi",
                    "topics": [
                        {
                            "title": "Su Döngüsü",
                            "explanation": "Su döngüsü, suyun Dünya'da nasıl hareket ettiğini gösterir. Su okyanuslardan buharlaşır, bulutları oluşturur ve yağmur olarak geri döner.",
                            "questions": [
                                {"question": "Su buharlaştığında ne olur?", "answer": "su buharı olur", "alternatives": ["buhar", "gaz", "buhar haline gelir"]},
                                {"question": "Bulutlardan düşen suya ne denir?", "answer": "yağmur", "alternatives": ["yağış"]}
                            ]
                        }
                    ]
                }
            }
        else:  # English
            return {
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
            return None
            
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

class StudentProfile:
    """Manage student profile information."""
    
    def __init__(self):
        self.profile_file = "student_profile.json"
        self.name = ""
        self.age = 0
        self.grade = ""
        self.load_profile()
    
    def load_profile(self):
        """Load student profile from file."""
        try:
            if os.path.exists(self.profile_file):
                with open(self.profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.name = data.get('name', '')
                    self.age = data.get('age', 0)
                    self.grade = data.get('grade', '')
                    print(f"✅ Profile loaded: {self.name}, Age {self.age}, Grade {self.grade}")
        except Exception as e:
            print(f"⚠️ Profile loading error: {e}")
    
    def save_profile(self, name, age, grade):
        """Save student profile to file."""
        try:
            self.name = name
            self.age = age
            self.grade = grade
            
            data = {
                'name': name,
                'age': age,
                'grade': grade
            }
            
            with open(self.profile_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Profile saved: {name}, Age {age}, Grade {grade}")
            return True
        except Exception as e:
            print(f"⚠️ Profile saving error: {e}")
            return False
    
    def is_complete(self):
        """Check if profile is complete."""
        return bool(self.name and self.age > 0 and self.grade)
    
    def get_profile_summary(self):
        """Get profile summary string."""
        return f"Name: {self.name}, Age: {self.age}, Grade: {self.grade}"

class AILanguageModel:
    """Handle AI responses and evaluation with multi-language support and Gemini integration."""
    
    def __init__(self, language='en'):
        self.language = language
        self.responses = self.get_responses()
        self.use_gemini = False
        self.gemini_client = None
        self.gemini_model_name = None
        self.student_profile = None  # Will be set by GUI
        
        # Initialize Gemini if available
        if GEMINI_AVAILABLE:
            self.initialize_gemini()
    
    def set_student_profile(self, profile):
        """Set student profile for personalized responses."""
        self.student_profile = profile
    
    def initialize_gemini(self):
        """Initialize Google Gemini API with new Client structure."""
        try:
            # Get API key from environment variable or config file
            api_key = os.environ.get('GEMINI_API_KEY')
            
            if not api_key:
                # Try to load from config file
                config_file = 'gemini_config.json'
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        api_key = config.get('api_key')
            
            if api_key:
                # Create Gemini client with API key
                self.gemini_client = genai.Client(api_key=api_key)
                
                # Try different model names in order of preference
                models_to_try = [
                    'gemini-2.0-flash-exp',
                    'gemini-1.5-flash',
                    'gemini-1.5-pro',
                    'gemini-pro'
                ]
                
                for model_name in models_to_try:
                    try:
                        # Test the model with a simple query
                        test_response = self.gemini_client.models.generate_content(
                            model=model_name,
                            contents="Hi"
                        )
                        self.gemini_model_name = model_name
                        self.use_gemini = True
                        print(f"✅ Gemini AI initialized successfully ({model_name})")
                        break
                    except Exception as e:
                        print(f"⚠️ {model_name} not available: {str(e)[:150]}")
                        continue
                
                if not self.use_gemini:
                    print("⚠️ No Gemini models available. Using rule-based responses.")
            else:
                print("⚠️ Gemini API key not found. Using rule-based responses.")
                print("   To use Gemini:")
                print("   1. Get free API key from: https://aistudio.google.com/app/apikey")
                print("   2. Set environment variable: GEMINI_API_KEY=your_key")
                print("   3. Or create gemini_config.json with: {\"api_key\": \"your_key\"}")
        except Exception as e:
            print(f"⚠️ Gemini initialization error: {e}")
            import traceback
            traceback.print_exc()
            self.use_gemini = False
    
    def set_language(self, language):
        """Change the language."""
        self.language = language
        self.responses = self.get_responses()
    
    def get_responses(self):
        """Get responses in current language."""
        if self.language == 'tr':
            return {
                'encouragement': [
                    "Harika iş! Çok iyi gidiyorsun!",
                    "Mükemmel cevap! Bunu çok iyi anladın!",
                    "Süper! Doğru cevap!"
                ],
                'gentle_correction': [
                    "İyi bir deneme! Sana daha iyi anlamanda yardımcı olayım.",
                    "Yaklaştın! Hadi birlikte düşünelim.",
                    "Tam değil, ama doğru yoldasın!"
                ],
                'confusion_detected': [
                    "Biraz kafan karışmış gibi görünüyor. Sorun değil!",
                    "Bunu farklı bir şekilde açıklayayım.",
                    "Merak etme, bunu adım adım inceleyelim."
                ],
                'correct_but_confused': "Cevabın doğru ama kafan karışmış görünüyor. Konuyu daha iyi anlamanı sağlamak için tekrar açıklayayım.",
                'correct_but_uncertain': "Doğru cevap verdin ama emin değil gibi görünüyorsun. Sana bu konuyu pekiştirmemiz için bir kez daha açıklayayım.",
                'correct_and_confident': "Mükemmel! Hem doğru cevap verdin hem de kendinden eminsin. Harika iş!",
                'wrong_but_confident': "Cevabın yanlış ama çok emin görünüyorsun. Hadi bu konuyu daha dikkatli inceleyelim."
            }
        else:  # English
            return {
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
                ],
                'correct_but_confused': "Your answer is correct, but you seem confused. Let me explain this again to make sure you truly understand.",
                'correct_but_uncertain': "That's the right answer, but you don't seem very confident. Let me reinforce this topic for you.",
                'correct_and_confident': "Perfect! You got it right and you seem confident. Excellent work!",
                'wrong_but_confident': "Your answer is incorrect, but you seem very confident. Let's carefully review this topic together."
            }
    
    def evaluate_answer(self, question_data, student_answer, voice_analysis, face_emotion):
        """Evaluate student's answer considering text and face emotion with detailed analysis."""
        if not question_data or not student_answer:
            return {'correct': False, 'confidence': 0, 'needs_re_explanation': True}
        
        correct_answer = question_data['answer'].lower()
        alternatives = [alt.lower() for alt in question_data.get('alternatives', [])]
        student_lower = student_answer.lower().strip()
        
        is_correct = (correct_answer in student_lower or 
                     any(alt in student_lower for alt in alternatives))
        
        # Base confidence on correctness
        text_confidence = 0.8 if is_correct else 0.3
        
        # Analyze face emotion for confidence and understanding
        emotion_confidence = 0.5
        needs_re_explanation = False
        
        if face_emotion:
            if face_emotion in ['confident', 'focused', 'happy']:
                emotion_confidence = 0.9
            elif face_emotion in ['confused', 'anxiety', 'frustrated', 'sad']:
                emotion_confidence = 0.2
                needs_re_explanation = True
            elif face_emotion in ['curious', 'neutral']:
                emotion_confidence = 0.6
            elif face_emotion in ['bored']:
                emotion_confidence = 0.4
            else:
                emotion_confidence = 0.5
        
        # Combined confidence score
        overall_confidence = (text_confidence * 0.6) + (emotion_confidence * 0.4)
        
        # Determine if re-explanation is needed based on emotion even if answer is correct
        if is_correct and emotion_confidence < 0.5:
            needs_re_explanation = True
        
        return {
            'correct': is_correct,
            'confidence': overall_confidence,
            'emotion_confidence': emotion_confidence,
            'face_emotion': face_emotion,
            'needs_re_explanation': needs_re_explanation,
            'voice_analysis': None  # Voice analysis disabled
        }
    
    def generate_response_with_gemini(self, evaluation_result, topic_title, question, student_answer):
        """Generate AI response using Gemini for more contextual feedback."""
        if not self.use_gemini or not self.gemini_client:
            return self.generate_response(evaluation_result, topic_title)
        
        try:
            is_correct = evaluation_result['correct']
            emotion = evaluation_result.get('face_emotion', 'neutral')
            emotion_conf = evaluation_result.get('emotion_confidence', 0.5)
            
            # Build prompt for Gemini
            lang_instruction = "Respond in Turkish" if self.language == 'tr' else "Respond in English"
            
            prompt = f"""You are a patient, encouraging AI teacher for elementary students.

Topic: {topic_title}
Question: {question}
Student's Answer: {student_answer}
Correct: {is_correct}
Student's Emotion: {emotion}
Confidence Level: {emotion_conf:.2f}

{lang_instruction}. Provide a brief, encouraging response (1-2 sentences) that:
- If correct and confident: praise the student
- If correct but {emotion} (low confidence): encourage and offer brief reinforcement
- If incorrect: gently correct with a hint, don't give away the full answer
- Keep it simple and age-appropriate for children
- Be warm and supportive

Response:"""

            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt
            )
            ai_response = response.text.strip()
            
            # Fallback if response is too long
            if len(ai_response) > 300:
                return self.generate_response(evaluation_result, topic_title)
            
            return ai_response
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Fallback to rule-based response
            return self.generate_response(evaluation_result, topic_title)
    
    def generate_response(self, evaluation_result, topic_title):
        """Generate appropriate AI teacher response based on correctness and emotion (rule-based)."""
        import random
        
        is_correct = evaluation_result['correct']
        emotion_conf = evaluation_result.get('emotion_confidence', 0.5)
        face_emotion = evaluation_result.get('face_emotion', 'neutral')
        
        # Correct answer cases
        if is_correct:
            if emotion_conf >= 0.7:
                # Correct and confident/focused
                return self.responses['correct_and_confident']
            elif emotion_conf < 0.5:
                # Correct but confused/uncertain
                if face_emotion in ['confused', 'anxiety']:
                    return self.responses['correct_but_confused']
                else:
                    return self.responses['correct_but_uncertain']
            else:
                # Correct with moderate confidence
                return random.choice(self.responses['encouragement'])
        
        # Incorrect answer cases
        else:
            if emotion_conf >= 0.7:
                # Wrong but confident
                return self.responses['wrong_but_confident']
            else:
                # Wrong and uncertain/confused
                return random.choice(self.responses['gentle_correction'])
    
    def generate_re_explanation_with_gemini(self, topic):
        """Generate a re-explanation using Gemini for more creative explanations."""
        if not self.use_gemini or not self.gemini_client:
            return self.generate_re_explanation(topic)
        
        try:
            lang_instruction = "Respond in Turkish" if self.language == 'tr' else "Respond in English"
            
            prompt = f"""You are a creative AI teacher for elementary students.

Topic: {topic['title']}
Original Explanation: {topic.get('explanation', '')}

{lang_instruction}. Provide a DIFFERENT, creative way to explain this topic to a child who didn't fully understand the first explanation. Use:
- Simple analogies or real-world examples
- Engaging language
- 2-3 sentences maximum
- Make it fun and relatable

Alternative Explanation:"""

            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt
            )
            ai_response = response.text.strip()
            
            # Fallback if response is too long
            if len(ai_response) > 400:
                return self.generate_re_explanation(topic)
            
            return ai_response
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self.generate_re_explanation(topic)
    
    def generate_chat_response(self, user_message, emotion):
        """Generate educational chat response using Gemini or rule-based."""
        if self.use_gemini and self.gemini_client:
            return self.generate_chat_with_gemini(user_message, emotion)
        else:
            return self.generate_chat_rule_based(user_message, emotion)
    
    def generate_chat_with_gemini(self, user_message, emotion):
        """Generate chat using Gemini (no topic restrictions for now)."""
        try:
            lang_instruction = "Respond in Turkish" if self.language == 'tr' else "Respond in English"
            
            # Build context with student profile
            profile_context = ""
            if self.student_profile and self.student_profile.is_complete():
                profile_context = f"""
Student Profile:
- Name: {self.student_profile.name}
- Age: {self.student_profile.age}
- Grade: {self.student_profile.grade}
"""
            
            prompt = f"""You are a patient, friendly AI assistant helping a student.

{profile_context}
Student's Current Emotion: {emotion}
Student's Message: {user_message}

IMPORTANT RULES:
1. Be helpful, friendly and supportive
2. Keep responses appropriate for the student's age: {self.student_profile.age if self.student_profile else 'young'} years old
3. {lang_instruction}
4. Keep response to 3-4 sentences maximum
5. If student seems {emotion}, adjust your tone accordingly
6. Provide clear, helpful explanations when asked
7. Use examples when explaining concepts

Response:"""

            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt
            )
            ai_response = response.text.strip()
            
            # Check if response is too long
            if len(ai_response) > 500:
                # Truncate but keep it coherent
                sentences = ai_response.split('.')
                ai_response = '. '.join(sentences[:4]) + '.'
            
            return ai_response
            
        except Exception as e:
            print(f"Gemini chat error: {e}")
            return self.generate_chat_rule_based(user_message, emotion)
    
    def generate_chat_rule_based(self, user_message, emotion):
        """Generate chat using rule-based system (no topic restrictions)."""
        # Simple friendly responses
        if self.language == 'tr':
            responses = [
                "Merhaba! Sana nasıl yardımcı olabilirim?",
                "Tabii! Bu konu hakkında sana yardımcı olabilirim. Ne bilmek istersin?",
                "İlginç bir soru! Sana yardımcı olmaktan mutluluk duyarım.",
                "Harika! Seninle konuşmak güzel. Bana daha fazla anlat.",
                "Anladım! Bu konuda sana yardımcı olayım."
            ]
        else:
            responses = [
                "Hello! How can I help you?",
                "Of course! I'd be happy to help you with this. What would you like to know?",
                "That's an interesting question! I'm happy to assist you.",
                "Great! It's nice talking with you. Tell me more.",
                "I understand! Let me help you with that."
            ]
        
        import random
        return random.choice(responses)

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
            
            # Check if table exists and get its columns
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
            table_exists = cursor.fetchone()
            
            if table_exists:
                # Check if language column exists
                cursor.execute("PRAGMA table_info(sessions)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'language' not in columns:
                    print("⚠️ Updating database schema - adding language column...")
                    cursor.execute("ALTER TABLE sessions ADD COLUMN language TEXT")
                    conn.commit()
                    print("✅ Database schema updated")
            else:
                # Create table with all columns including language
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
                        language TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                print("✅ Database created with language column")
            
            conn.close()
            print("✅ Database initialized")
        except Exception as e:
            print(f"⚠️ Database error: {e}")
            import traceback
            traceback.print_exc()
    
    def log_interaction(self, lesson_name, topic_name, question, answer, evaluation, language):
        """Log student interaction."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sessions (lesson_name, topic_name, question, answer, correct, confidence, face_emotion, voice_emotion, language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                lesson_name,
                topic_name, 
                question,
                answer,
                evaluation.get('correct', False),
                evaluation.get('confidence', 0.0),
                evaluation.get('face_emotion', ''),
                evaluation.get('voice_analysis', {}).get('emotion', '') if evaluation.get('voice_analysis') else '',
                language
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ Database logging error: {e}")
            import traceback
            traceback.print_exc()

class AITeacherGUI:
    """Main GUI for AI Teacher application with multi-language support."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.current_language = 'en'
        self.update_title()
        self.root.geometry("1400x800")  # Wider for profile panel
        
        # Initialize student profile
        self.student_profile = StudentProfile()
        
        # Initialize components
        self.emotion_detector = self.init_emotion_detector()
        self.speech_engine = SpeechEngine(self.current_language)
        self.lesson_manager = LessonManager(self.current_language)
        self.ai_model = AILanguageModel(self.current_language)
        self.ai_model.set_student_profile(self.student_profile)  # Set profile in AI model
        self.progress_tracker = StudentProgressTracker()
        
        # State variables
        self.teaching_active = False
        self.chat_mode = True  # Always allow chat
        self.current_face_emotion = "neutral"
        self.lesson_thread = None
        self.waiting_for_answer = False
        self.current_question_data = None
        self.current_topic = None
        self.answer_queue = queue.Queue()
        self.answer_event = threading.Event()
        
        self.create_gui()
        self.start_emotion_detection()
        
        # Show initial greeting
        self.show_initial_greeting()
    
    def show_initial_greeting(self):
        """Show personalized greeting based on profile."""
        t = TRANSLATIONS[self.current_language]
        
        if self.student_profile.is_complete():
            if self.current_language == 'en':
                greeting = f"Welcome back, {self.student_profile.name}! I'm excited to continue learning with you."
            else:
                greeting = f"Tekrar hoş geldin, {self.student_profile.name}! Seninle öğrenmeye devam etmek için heyecanlıyım."
            self.add_to_conversation("🤖 AI Teacher", greeting)
            self.add_to_progress(f"Student: {self.student_profile.get_profile_summary()}")
        else:
            self.add_to_conversation("🤖 AI Teacher", t['welcome'])
            self.add_to_progress(t['system_ready'])
        
        # Show Gemini status
        if self.ai_model.use_gemini:
            self.add_to_progress("✨ Gemini AI enabled for enhanced responses")
        else:
            self.add_to_progress("ℹ️ Using rule-based responses (Gemini not configured)")
    
    def update_title(self):
        """Update window title based on language."""
        self.root.title(TRANSLATIONS[self.current_language]['title'])
    
    def change_language(self, event=None):
        """Change application language."""
        new_lang = self.language_var.get()
        if new_lang == self.current_language:
            return
        
        self.current_language = new_lang
        self.update_title()
        
        # Update components
        self.speech_engine.set_language(new_lang)
        self.lesson_manager.set_language(new_lang)
        self.ai_model.set_language(new_lang)
        
        # Update GUI elements
        self.update_gui_language()
        
        # Update lesson list
        self.lesson_combo['values'] = self.lesson_manager.get_available_lessons()
        if self.lesson_manager.get_available_lessons():
            self.lesson_var.set(self.lesson_manager.get_available_lessons()[0])
    
    def update_gui_language(self):
        """Update all GUI text to current language."""
        t = TRANSLATIONS[self.current_language]
        
        # Update labels and buttons
        self.control_frame.config(text=t['controls'])
        self.lesson_label.config(text=t['select_lesson'])
        self.lang_label.config(text=t['select_language'])
        self.start_btn.config(text=t['start_teaching'])
        self.stop_btn.config(text=t['stop_teaching'])
        self.status_frame.config(text=t['status'])
        self.emotion_text_label.config(text=t['emotion'])
        self.teaching_text_label.config(text=t['teaching'])
        self.conv_frame.config(text=t['conversation'])
        self.progress_frame.config(text=t['progress'])
        self.send_btn.config(text=t['send'])
        
        # Update status labels if needed
        if not self.teaching_active:
            self.teaching_status.config(text=t['stopped'])
        else:
            self.teaching_status.config(text=t['active'])
    
    def init_emotion_detector(self):
        """Initialize emotion detection."""
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
        t = TRANSLATIONS[self.current_language]
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1)  # Middle column (conversation) expands
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text=t['title'], 
                               font=("Arial", 20, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))
        
        # Left panel - Student Profile
        self.profile_frame = ttk.LabelFrame(main_frame, text=t['student_profile'], padding="10")
        self.profile_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Name
        ttk.Label(self.profile_frame, text=t['name'], font=("Arial", 11)).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.name_entry = ttk.Entry(self.profile_frame, width=20, font=("Arial", 11))
        self.name_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        if self.student_profile.name:
            self.name_entry.insert(0, self.student_profile.name)
        
        # Age
        ttk.Label(self.profile_frame, text=t['age'], font=("Arial", 11)).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.age_entry = ttk.Entry(self.profile_frame, width=20, font=("Arial", 11))
        self.age_entry.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        if self.student_profile.age:
            self.age_entry.insert(0, str(self.student_profile.age))
        
        # Grade
        ttk.Label(self.profile_frame, text=t['grade'], font=("Arial", 11)).grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        self.grade_entry = ttk.Entry(self.profile_frame, width=20, font=("Arial", 11))
        self.grade_entry.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        if self.student_profile.grade:
            self.grade_entry.insert(0, self.student_profile.grade)
        
        # Save profile button
        ttk.Button(self.profile_frame, text=t['save_profile'], 
                  command=self.save_student_profile).grid(row=6, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Separator
        ttk.Separator(self.profile_frame, orient='horizontal').grid(row=7, column=0, sticky=(tk.W, tk.E), pady=20)
        
        # Controls section within profile frame
        ttk.Label(self.profile_frame, text=t['controls'], font=("Arial", 12, "bold")).grid(row=8, column=0, sticky=tk.W, pady=(0, 10))
        
        # Language selection
        self.lang_label = ttk.Label(self.profile_frame, text=t['select_language'], font=("Arial", 11))
        self.lang_label.grid(row=9, column=0, sticky=tk.W, pady=(0, 5))
        
        self.language_var = tk.StringVar(value='en')
        language_combo = ttk.Combobox(self.profile_frame, textvariable=self.language_var,
                                     values=['en', 'tr'], state="readonly", width=17)
        language_combo.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        language_combo.bind('<<ComboboxSelected>>', self.change_language)
        
        # Lesson selection
        self.lesson_label = ttk.Label(self.profile_frame, text=t['select_lesson'], font=("Arial", 11))
        self.lesson_label.grid(row=11, column=0, sticky=tk.W, pady=(0, 5))
        
        self.lesson_var = tk.StringVar()
        self.lesson_combo = ttk.Combobox(self.profile_frame, textvariable=self.lesson_var, 
                                   values=self.lesson_manager.get_available_lessons(), 
                                   state="readonly", width=17)
        self.lesson_combo.grid(row=12, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        if self.lesson_manager.get_available_lessons():
            self.lesson_combo.set(self.lesson_manager.get_available_lessons()[0])
        
        # Control buttons
        self.start_btn = ttk.Button(self.profile_frame, text=t['start_teaching'], 
                                   command=self.start_teaching)
        self.start_btn.grid(row=13, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.stop_btn = ttk.Button(self.profile_frame, text=t['stop_teaching'], 
                                  command=self.stop_teaching, state='disabled')
        self.stop_btn.grid(row=14, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Status indicators
        self.status_frame = ttk.LabelFrame(self.profile_frame, text=t['status'], padding="10")
        self.status_frame.grid(row=15, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        
        self.emotion_text_label = ttk.Label(self.status_frame, text=t['emotion'])
        self.emotion_text_label.grid(row=0, column=0, sticky=tk.W)
        self.emotion_label = ttk.Label(self.status_frame, text=t['analyzing'], 
                                      font=("Arial", 10, "bold"), foreground="blue")
        self.emotion_label.grid(row=0, column=1, sticky=tk.W)
        
        self.teaching_text_label = ttk.Label(self.status_frame, text=t['teaching'])
        self.teaching_text_label.grid(row=1, column=0, sticky=tk.W)
        self.teaching_status = ttk.Label(self.status_frame, text=t['stopped'], 
                                        font=("Arial", 10, "bold"), foreground="red")
        self.teaching_status.grid(row=1, column=1, sticky=tk.W)
        
        # Middle panel - Conversation (spans 2 columns for more width)
        self.conv_frame = ttk.LabelFrame(main_frame, text=t['conversation'], padding="10")
        self.conv_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        self.conv_frame.columnconfigure(0, weight=1)
        self.conv_frame.rowconfigure(0, weight=1)
        
        self.conversation_text = scrolledtext.ScrolledText(self.conv_frame, wrap=tk.WORD, 
                                                          height=25, width=70,
                                                          font=("Arial", 11))
        self.conversation_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input frame
        input_frame = ttk.Frame(self.conv_frame)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        input_frame.columnconfigure(0, weight=1)
        
        self.manual_input = ttk.Entry(input_frame, font=("Arial", 11))
        self.manual_input.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.manual_input.bind('<Return>', self.on_manual_input)
        
        self.send_btn = ttk.Button(input_frame, text=t['send'], command=self.on_manual_input)
        self.send_btn.grid(row=0, column=1)
        
        # Right panel - Progress
        self.progress_frame = ttk.LabelFrame(main_frame, text=t['progress'], padding="10")
        self.progress_frame.grid(row=1, column=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        self.progress_text = scrolledtext.ScrolledText(self.progress_frame, wrap=tk.WORD, 
                                                      height=25, width=35,
                                                      font=("Arial", 10))
        self.progress_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def save_student_profile(self):
        """Save student profile information."""
        t = TRANSLATIONS[self.current_language]
        
        name = self.name_entry.get().strip()
        age_str = self.age_entry.get().strip()
        grade = self.grade_entry.get().strip()
        
        # Validation
        if not name:
            messagebox.showwarning(t['warning'], t['enter_name'])
            return
        
        if not age_str or not age_str.isdigit():
            messagebox.showwarning(t['warning'], t['enter_age'])
            return
        
        if not grade:
            messagebox.showwarning(t['warning'], t['enter_grade'])
            return
        
        age = int(age_str)
        
        if self.student_profile.save_profile(name, age, grade):
            self.add_to_progress(f"✅ {t['profile_saved']}")
            self.add_to_progress(f"Student: {self.student_profile.get_profile_summary()}")
            
            # Personalized greeting
            if self.current_language == 'en':
                greeting = f"Nice to meet you, {name}! I'm looking forward to teaching you. You can ask me anything about your studies!"
            else:
                greeting = f"Tanıştığımıza memnun oldum, {name}! Sana ders anlatmayı dört gözle bekliyorum. Bana derslerinle ilgili her şeyi sorabilirsin!"
            
            self.add_to_conversation("🤖 AI Teacher", greeting)
            self.speech_engine.speak_text(greeting)
    
    def add_to_conversation(self, speaker, message):
        """Add message to conversation display."""
        try:
            self.conversation_text.config(state='normal')
            
            if speaker.startswith("🤖"):
                color = "blue"
            elif speaker.startswith("👤"):
                color = "green"
            else:
                color = "black"
            
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
                        
                        try:
                            self.root.after(0, lambda: self.emotion_label.config(text=emotion))
                        except:
                            pass
                        
                        time.sleep(0.5)
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
        t = TRANSLATIONS[self.current_language]
        
        lesson_name = self.lesson_var.get()
        if not lesson_name:
            messagebox.showwarning(t['warning'], t['select_lesson_first'])
            return
        
        if not self.lesson_manager.start_lesson(lesson_name):
            messagebox.showerror(t['error'], t['failed_start'])
            return
        
        self.teaching_active = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.teaching_status.config(text=t['active'], foreground="green")
        
        self.add_to_conversation("🤖 AI Teacher", t['starting_lesson'].format(lesson_name.title()))
        self.add_to_progress(f"{t['lesson_started']}: {lesson_name}")
        
        self.lesson_thread = threading.Thread(target=self.teaching_loop, daemon=True)
        self.lesson_thread.start()
    
    def stop_teaching(self):
        """Stop AI teaching session but keep chat active."""
        t = TRANSLATIONS[self.current_language]
        
        self.teaching_active = False
        self.chat_mode = True  # Keep chat active
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.teaching_status.config(text=t['stopped'], foreground="red")
        
        self.add_to_conversation("🤖 AI Teacher", t['session_ended'])
        self.add_to_progress(t['session_stopped'])
        self.add_to_progress(f"💬 {t['chat_active']}")
        
        self.speech_engine.speak_text(t['session_ended'])
    
    def teaching_loop(self):
        """Main teaching loop."""
        t = TRANSLATIONS[self.current_language]
        
        while self.teaching_active:
            topic = self.lesson_manager.get_current_topic()
            
            if not topic:
                self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", t['lesson_completed']))
                self.root.after(0, self.stop_teaching)
                break
            
            self.teach_topic(topic)
            
            if not self.teaching_active:
                break
    
    def teach_topic(self, topic):
        """Teach a specific topic using text input with proper synchronization."""
        t = TRANSLATIONS[self.current_language]
        
        try:
            explanation = topic['explanation']
            msg = f"{topic['title']}. {explanation}" if self.current_language == 'tr' else f"Let's learn about {topic['title']}. {explanation}"
            
            self.root.after(0, lambda m=msg: self.add_to_conversation("🤖 AI Teacher", m))
            self.root.after(0, lambda: self.add_to_progress(f"{t['teaching_topic']}: {topic['title']}"))
            
            # Speak the explanation
            self.speech_engine.speak_text(msg)
            time.sleep(1)  # Short pause after explanation
            
            question_data = self.lesson_manager.get_random_question(topic)
            if not question_data:
                self.lesson_manager.move_to_next_topic()
                return
            
            question = question_data['question']
            self.root.after(0, lambda q=question: self.add_to_conversation("🤖 AI Teacher", q))
            self.speech_engine.speak_text(question)
            
            # Prompt for text input
            prompt_msg = t['type_answer']
            self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", prompt_msg))
            self.root.after(0, lambda: self.add_to_progress("⌨️ " + t['waiting_answer']))
            
            # Set up for waiting for manual input
            self.waiting_for_answer = True
            self.current_question_data = question_data
            self.current_topic = topic
            self.answer_event.clear()
            
            # Clear any old answers from queue
            while not self.answer_queue.empty():
                try:
                    self.answer_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Wait for answer using event
            timeout = 120
            answer_received = self.answer_event.wait(timeout=timeout)
            
            if not answer_received or not self.teaching_active:
                if self.teaching_active:
                    self.waiting_for_answer = False
                    msg = t['no_response']
                    self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", msg))
                    self.root.after(0, lambda: self.add_to_progress("⏰ Timeout - no answer provided"))
                    self.speech_engine.speak_text(msg)
                return
            
            # Get the answer from queue
            try:
                student_answer = self.answer_queue.get_nowait()
            except queue.Empty:
                return
            
            self.waiting_for_answer = False
            self.root.after(0, lambda: self.add_to_progress("✍️ Processing answer..."))
            
            # Evaluate the answer with emotion analysis
            evaluation = self.ai_model.evaluate_answer(
                question_data, 
                student_answer, 
                None,
                self.current_face_emotion
            )
            
            self.progress_tracker.log_interaction(
                self.lesson_manager.current_lesson,
                topic['title'],
                question,
                student_answer,
                evaluation,
                self.current_language
            )
            
            # Generate response - use Gemini if available
            if self.ai_model.use_gemini:
                self.root.after(0, lambda: self.add_to_progress("✨ Generating AI response..."))
                response = self.ai_model.generate_response_with_gemini(
                    evaluation, topic['title'], question, student_answer
                )
            else:
                response = self.ai_model.generate_response(evaluation, topic['title'])
            
            self.root.after(0, lambda r=response: self.add_to_conversation("🤖 AI Teacher", r))
            self.speech_engine.speak_text(response)
            
            time.sleep(1)
            
            # Decide next action based on evaluation
            if evaluation['correct'] and not evaluation['needs_re_explanation']:
                self.root.after(0, lambda: self.add_to_progress(f"✅ {t['topic_mastered']}: {topic['title']}"))
                self.lesson_manager.move_to_next_topic()
                time.sleep(1)
            elif evaluation['correct'] and evaluation['needs_re_explanation']:
                # Use Gemini for re-explanation if available
                if self.ai_model.use_gemini:
                    re_explanation = self.ai_model.generate_re_explanation_with_gemini(topic)
                else:
                    re_explanation = self.ai_model.generate_re_explanation(topic)
                
                self.root.after(0, lambda re=re_explanation: self.add_to_conversation("🤖 AI Teacher", re))
                self.speech_engine.speak_text(re_explanation)
                self.root.after(0, lambda: self.add_to_progress(f"🔄 {t['re_explaining']}: {topic['title']} (emotion-based)"))
                self.lesson_manager.move_to_next_topic()
                time.sleep(2)
            else:
                self.lesson_manager.retry_current_topic()
                
                if self.lesson_manager.should_re_explain():
                    # Use Gemini for re-explanation if available
                    if self.ai_model.use_gemini:
                        re_explanation = self.ai_model.generate_re_explanation_with_gemini(topic)
                    else:
                        re_explanation = self.ai_model.generate_re_explanation(topic)
                    
                    self.root.after(0, lambda re=re_explanation: self.add_to_conversation("🤖 AI Teacher", re))
                    self.speech_engine.speak_text(re_explanation)
                    self.root.after(0, lambda: self.add_to_progress(f"🔄 {t['re_explaining']}: {topic['title']}"))
                    self.lesson_manager.move_to_next_topic()
                    time.sleep(2)
                else:
                    self.root.after(0, lambda: self.add_to_progress(f"🤔 {t['retrying']}: {topic['title']}"))
                    time.sleep(1)
                
        except Exception as e:
            print(f"Teaching error: {e}")
            import traceback
            traceback.print_exc()
            self.waiting_for_answer = False
            msg = t['issue']
            self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", msg))
            self.speech_engine.speak_text(msg)
    
    def on_manual_input(self, event=None):
        """Handle manual text input - for answering questions or chatting."""
        t = TRANSLATIONS[self.current_language]
        text = self.manual_input.get().strip()
        if not text:
            return
        
        self.add_to_conversation(f"👤 {t['student']}", text)
        self.manual_input.delete(0, tk.END)
        
        # If waiting for an answer in teaching mode
        if self.waiting_for_answer and self.teaching_active:
            self.answer_queue.put(text)
            self.answer_event.set()
        
        # Otherwise, treat as general educational chat
        elif self.chat_mode:
            # Process in separate thread to avoid blocking GUI
            threading.Thread(target=self.process_chat_message, args=(text,), daemon=True).start()
    
    def process_chat_message(self, message):
        """Process general educational chat message."""
        try:
            self.root.after(0, lambda: self.add_to_progress("💬 Processing chat..."))
            
            # Generate AI response
            response = self.ai_model.generate_chat_response(message, self.current_face_emotion)
            
            # Display response
            self.root.after(0, lambda r=response: self.add_to_conversation("🤖 AI Teacher", r))
            self.speech_engine.speak_text(response)
            
            # Log chat interaction with proper structure
            chat_evaluation = {
                'correct': True,
                'confidence': 0.5,
                'face_emotion': self.current_face_emotion,
                'voice_analysis': None
            }
            
            self.progress_tracker.log_interaction(
                "chat",
                "general_chat",
                message,
                response,
                chat_evaluation,
                self.current_language
            )
            
        except Exception as e:
            print(f"Chat processing error: {e}")
            import traceback
            traceback.print_exc()
            error_msg = "Sorry, I encountered an error. Please try again." if self.current_language == 'en' else "Üzgünüm, bir hata oluştu. Lütfen tekrar dene."
            self.root.after(0, lambda: self.add_to_conversation("🤖 AI Teacher", error_msg))
    
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
    print("AI Virtual Teacher - Text Input with Speech Output")
    print("Multi-Language Support: English & Turkish")
    print("Enhanced with Google Gemini AI (Optional)")
    print("="*60)
    print("Features:")
    print("- Face emotion detection")
    print("- Text-based input (type your answers)")
    print("- Speech output (AI teacher speaks)")
    print("- Emotion-aware evaluation")
    print("- Adaptive teaching based on student understanding")
    print("- Progress tracking")
    print("- Interactive lessons")
    print("- Multi-language support (EN/TR)")
    print("- Google Gemini AI for intelligent responses (if configured)")
    print("="*60)
    
    # Check dependencies
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
    
    if not PYTTSX3_AVAILABLE:
        print("⚠️ pyttsx3 not available. AI teacher won't speak.")
        print("   Install with: pip install pyttsx3")
    
    if not GEMINI_AVAILABLE:
        print("⚠️ Google Gemini not available (optional).")
        print("   For enhanced AI responses, install: pip install google-genai")
        print("   Get free API key: https://aistudio.google.com/app/apikey")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        print("Please install: pip install " + " ".join(missing_deps))
        return
    
    print("✅ All core dependencies available")
    
    print("="*60)
    
    app = AITeacherGUI()
    app.run()

if __name__ == "__main__":
    main()