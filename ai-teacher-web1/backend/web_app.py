from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import numpy as np
import base64
import json
import os
from datetime import datetime
import sqlite3
from typing import Dict, List, Optional
import asyncio

# Import DeepFace with error handling
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully")
except Exception as e:
    print(f"⚠️ DeepFace not available: {e}")
    DEEPFACE_AVAILABLE = False

# Import Gemini
try:
    from google import genai
    GEMINI_AVAILABLE = True
    print("✅ Gemini library available")
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini library not available")

app = FastAPI(title="AI Virtual Teacher API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_sessions: Dict[str, dict] = {}

# Initialize Gemini
gemini_client = None
gemini_model_name = None

def init_gemini():
    global gemini_client, gemini_model_name
    
    if not GEMINI_AVAILABLE:
        print("⚠️ Gemini not available - AI features disabled")
        return False
    
    # Try to get API key
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gemini_config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get('api_key')
        except Exception as e:
            print(f"⚠️ Could not load gemini_config.json: {e}")
    
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        print("⚠️ Gemini API key not configured")
        return False
    
    try:
        gemini_client = genai.Client(api_key=api_key)
        
        # Try models in order of preference
        for model_name in ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro']:
            try:
                test_response = gemini_client.models.generate_content(
                    model=model_name,
                    contents="Hi"
                )
                gemini_model_name = model_name
                print(f"✅ Gemini initialized: {model_name}")
                return True
            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {e}")
                continue
        
        print("❌ No Gemini models available")
        return False
        
    except Exception as e:
        print(f"❌ Gemini initialization failed: {e}")
        return False

# Initialize Gemini on startup
GEMINI_READY = init_gemini()

# Database setup
DB_PATH = "student_progress.db"

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            lesson_name TEXT,
            topic_name TEXT,
            question TEXT,
            answer TEXT,
            correct BOOLEAN,
            confidence REAL,
            face_emotion TEXT,
            language TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            grade TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lesson_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            lesson_name TEXT,
            topic_index INTEGER,
            question_index INTEGER,
            completed BOOLEAN DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

init_database()

# Comprehensive lessons database
LESSONS_EN = {
    "mathematics": {
        "title": "Basic Mathematics",
        "topics": [
            {
                "title": "Addition",
                "explanation": "Addition is combining two or more numbers to get their sum. When we add, we count all items together. For example, if you have 2 apples and get 3 more apples, you now have 5 apples total. We write this as 2 + 3 = 5.",
                "questions": [
                    {"question": "What is 5 plus 3?", "answer": "8", "alternatives": ["eight"]},
                    {"question": "If you have 4 apples and get 2 more, how many do you have?", "answer": "6", "alternatives": ["six"]},
                    {"question": "What is 7 + 2?", "answer": "9", "alternatives": ["nine"]},
                ]
            },
            {
                "title": "Subtraction",
                "explanation": "Subtraction means taking away. When we subtract, we remove items from a group. For example, if you have 5 cookies and eat 2, you have 3 cookies left. We write this as 5 - 2 = 3.",
                "questions": [
                    {"question": "What is 10 minus 3?", "answer": "7", "alternatives": ["seven"]},
                    {"question": "If you have 8 toys and give away 3, how many are left?", "answer": "5", "alternatives": ["five"]},
                    {"question": "What is 6 - 4?", "answer": "2", "alternatives": ["two"]},
                ]
            }
        ]
    },
    "english": {
        "title": "Basic English",
        "topics": [
            {
                "title": "Colors",
                "explanation": "Colors help us describe things around us. The sky is blue, grass is green, and the sun is yellow. Learning colors helps us communicate better!",
                "questions": [
                    {"question": "What color is the sky?", "answer": "blue", "alternatives": []},
                    {"question": "What color is grass?", "answer": "green", "alternatives": []},
                    {"question": "What color is the sun?", "answer": "yellow", "alternatives": []},
                ]
            }
        ]
    }
}

LESSONS_TR = {
    "matematik": {
        "title": "Temel Matematik",
        "topics": [
            {
                "title": "Toplama",
                "explanation": "Toplama, iki veya daha fazla sayıyı birleştirerek toplamlarını bulmaktır. Topladığımızda, tüm öğeleri birlikte sayarız. Örneğin, 2 elmanız varsa ve 3 elma daha alırsanız, toplam 5 elmanız olur. Bunu 2 + 3 = 5 şeklinde yazarız.",
                "questions": [
                    {"question": "5 artı 3 kaç eder?", "answer": "8", "alternatives": ["sekiz"]},
                    {"question": "4 elmanız varsa ve 2 tane daha alırsanız, kaç elmanız olur?", "answer": "6", "alternatives": ["altı"]},
                    {"question": "7 + 2 kaç eder?", "answer": "9", "alternatives": ["dokuz"]},
                ]
            },
            {
                "title": "Çıkarma",
                "explanation": "Çıkarma, bir gruptan öğeleri almak demektir. Çıkardığımızda sayıları azaltırız. Örneğin, 5 kurabiyeniz varsa ve 2 tanesini yerseniz, 3 kurabiye kalır. Bunu 5 - 2 = 3 şeklinde yazarız.",
                "questions": [
                    {"question": "10 eksi 3 kaç eder?", "answer": "7", "alternatives": ["yedi"]},
                    {"question": "8 oyuncağınız varsa ve 3 tanesini verirseniz, kaç tane kalır?", "answer": "5", "alternatives": ["beş"]},
                    {"question": "6 - 4 kaç eder?", "answer": "2", "alternatives": ["iki"]},
                ]
            }
        ]
    }
}

@app.get("/")
async def root():
    """Serve the main HTML page"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base_dir, "frontend", "index.html")
    
    try:
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(
            content=f"""
            <html>
                <head><title>Error</title></head>
                <body>
                    <h1>❌ Frontend Not Found</h1>
                    <p>Looking for: {html_path}</p>
                    <p>Create the frontend/index.html file</p>
                </body>
            </html>
            """,
            status_code=500
        )

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "deepface": DEEPFACE_AVAILABLE,
        "gemini": GEMINI_READY,
        "gemini_model": gemini_model_name,
        "active_sessions": len(active_sessions)
    }

@app.get("/api/lessons")
async def get_lessons(language: str = "en"):
    """Get available lessons"""
    lessons = LESSONS_TR if language == "tr" else LESSONS_EN
    return {
        "lessons": {
            key: {
                "title": value["title"],
                "topic_count": len(value["topics"])
            }
            for key, value in lessons.items()
        }
    }

@app.get("/api/lesson/{lesson_name}")
async def get_lesson(lesson_name: str, language: str = "en"):
    """Get specific lesson details"""
    lessons = LESSONS_TR if language == "tr" else LESSONS_EN
    
    if lesson_name not in lessons:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    return lessons[lesson_name]

@app.post("/api/profile")
async def save_profile(data: dict):
    """Save student profile"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO profiles (name, age, grade)
            VALUES (?, ?, ?)
        ''', (data.get('name'), data.get('age'), data.get('grade')))
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Profile saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lesson/start")
async def start_lesson(data: dict):
    """Start a new lesson"""
    session_id = data.get('session_id')
    lesson_name = data.get('lesson_name')
    language = data.get('language', 'en')
    
    lessons = LESSONS_TR if language == "tr" else LESSONS_EN
    
    if lesson_name not in lessons:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Initialize session progress
    if session_id in active_sessions:
        active_sessions[session_id]['lesson'] = {
            'name': lesson_name,
            'language': language,
            'topic_index': 0,
            'question_index': 0,
            'data': lessons[lesson_name]
        }
    
    # Get first topic
    first_topic = lessons[lesson_name]['topics'][0]
    
    return {
        "status": "started",
        "lesson_name": lesson_name,
        "lesson_title": lessons[lesson_name]['title'],
        "topic": first_topic['title'],
        "explanation": first_topic['explanation']
    }

@app.post("/api/lesson/next_question")
async def get_next_question(data: dict):
    """Get next question in current lesson"""
    session_id = data.get('session_id')
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    if 'lesson' not in session:
        raise HTTPException(status_code=400, detail="No active lesson")
    
    lesson_data = session['lesson']['data']
    topic_index = session['lesson']['topic_index']
    question_index = session['lesson']['question_index']
    
    topics = lesson_data['topics']
    
    if topic_index >= len(topics):
        return {
            "completed": True,
            "message": "Congratulations! You completed the lesson!" if session['lesson']['language'] == 'en' 
                      else "Tebrikler! Dersi tamamladınız!"
        }
    
    current_topic = topics[topic_index]
    questions = current_topic['questions']
    
    if question_index >= len(questions):
        # Move to next topic
        session['lesson']['topic_index'] += 1
        session['lesson']['question_index'] = 0
        
        if session['lesson']['topic_index'] >= len(topics):
            return {
                "completed": True,
                "message": "Congratulations! You completed the lesson!" if session['lesson']['language'] == 'en'
                          else "Tebrikler! Dersi tamamladınız!"
            }
        
        # Get next topic
        next_topic = topics[session['lesson']['topic_index']]
        return {
            "new_topic": True,
            "topic": next_topic['title'],
            "explanation": next_topic['explanation']
        }
    
    question = questions[question_index]
    session['lesson']['question_index'] += 1
    
    return {
        "question": question['question'],
        "question_data": question,
        "topic": current_topic['title'],
        "progress": {
            "topic": topic_index + 1,
            "total_topics": len(topics),
            "question": question_index + 1,
            "total_questions": len(questions)
        }
    }

@app.websocket("/ws/emotion/{session_id}")
async def websocket_emotion(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time emotion detection"""
    await websocket.accept()
    print(f"✅ WebSocket connected: {session_id}")
    
    active_sessions[session_id] = {
        'websocket': websocket,
        'last_emotion': 'neutral',
        'emotion_history': [],
        'connected_at': datetime.now()
    }
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                # Decode image
                img_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({'emotion': 'no_frame', 'error': 'Invalid frame'})
                    continue
                
                if not DEEPFACE_AVAILABLE:
                    await websocket.send_json({
                        'emotion': 'neutral',
                        'confidence': 0.5,
                        'all_emotions': {'neutral': 100.0},
                        'note': 'DeepFace not available'
                    })
                    continue
                
                # Detect emotion using DeepFace
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
                dominant = max(emotions, key=emotions.get)
                
                # Map to teaching-relevant emotions
                emotion_map = {
                    'happy': 'confident',
                    'neutral': 'focused',
                    'sad': 'confused',
                    'angry': 'frustrated',
                    'fear': 'anxiety',
                    'surprise': 'curious',
                    'disgust': 'bored'
                }
                
                teaching_emotion = emotion_map.get(dominant, dominant)
                
                # Update session
                active_sessions[session_id]['last_emotion'] = teaching_emotion
                active_sessions[session_id]['emotion_history'].append({
                    'emotion': teaching_emotion,
                    'confidence': emotions[dominant],
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only last 10 emotions
                if len(active_sessions[session_id]['emotion_history']) > 10:
                    active_sessions[session_id]['emotion_history'].pop(0)
                
                await websocket.send_json({
                    'emotion': teaching_emotion,
                    'confidence': float(emotions[dominant]),
                    'all_emotions': {k: float(v) for k, v in emotions.items()},
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"Emotion detection error: {e}")
                await websocket.send_json({
                    'emotion': 'analyzing',
                    'error': str(e)
                })
            
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        print(f"❌ WebSocket disconnected: {session_id}")
        if session_id in active_sessions:
            del active_sessions[session_id]
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.post("/api/evaluate_answer")
async def evaluate_answer(data: dict):
    """Evaluate student's answer with AI"""
    try:
        question_data = data.get('question_data')
        student_answer = data.get('answer', '').strip().lower()
        session_id = data.get('session_id')
        language = data.get('language', 'en')
        
        if not question_data or not student_answer:
            return {
                "correct": False,
                "response": "Please provide an answer." if language == 'en' else "Lütfen bir cevap verin."
            }
        
        # Get current emotion
        emotion = 'neutral'
        if session_id in active_sessions:
            emotion = active_sessions[session_id]['last_emotion']
        
        # Check answer
        correct_answer = str(question_data['answer']).lower()
        alternatives = [str(alt).lower() for alt in question_data.get('alternatives', [])]
        
        is_correct = (correct_answer == student_answer or 
                     correct_answer in student_answer or
                     any(alt == student_answer or alt in student_answer for alt in alternatives))
        
        # Calculate confidence
        text_confidence = 0.9 if is_correct else 0.3
        
        emotion_confidence = 0.5
        if emotion in ['confident', 'focused', 'curious']:
            emotion_confidence = 0.9
        elif emotion in ['confused', 'anxiety', 'frustrated']:
            emotion_confidence = 0.3
        
        overall_confidence = (text_confidence * 0.7) + (emotion_confidence * 0.3)
        
        # Generate AI response with Gemini
        if GEMINI_READY and gemini_client:
            try:
                lang_instr = "Yanıtını Türkçe ver" if language == "tr" else "Respond in English"
                
                prompt = f"""You are an encouraging AI teacher for elementary students (ages 6-12).

Question: {question_data['question']}
Student's Answer: {student_answer}
Correct Answer: {correct_answer}
Is Correct: {is_correct}
Student's Current Emotion: {emotion}

{lang_instr}. Provide a warm, encouraging response (2-3 sentences):

If CORRECT:
- Praise enthusiastically
- Acknowledge their effort
- If they seem anxious/confused despite correct answer, reassure them

If INCORRECT:
- Stay very positive and encouraging
- Give a gentle hint without revealing full answer
- Encourage them to try again
- If frustrated/anxious, provide extra comfort

Keep it simple, warm, and age-appropriate.

Response:"""
                
                response_obj = gemini_client.models.generate_content(
                    model=gemini_model_name,
                    contents=prompt
                )
                ai_response = response_obj.text.strip()
                
            except Exception as e:
                print(f"Gemini error: {e}")
                ai_response = "Great job!" if is_correct else "Good try! Think about it again."
        else:
            # Fallback responses
            if language == "tr":
                ai_response = "Harika! Doğru cevap!" if is_correct else "İyi deneme! Tekrar düşün."
            else:
                ai_response = "Excellent! That's correct!" if is_correct else "Good try! Think about it again."
        
        # Log to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions 
            (session_id, lesson_name, topic_name, question, answer, correct, confidence, face_emotion, language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            data.get('lesson_name', ''),
            data.get('topic_name', ''),
            question_data['question'],
            student_answer,
            is_correct,
            overall_confidence,
            emotion,
            language
        ))
        conn.commit()
        conn.close()
        
        return {
            "correct": is_correct,
            "confidence": overall_confidence,
            "emotion": emotion,
            "response": ai_response,
            "needs_encouragement": emotion in ['confused', 'frustrated', 'anxiety']
        }
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(data: dict):
    """Handle general educational chat with AI"""
    try:
        message = data.get('message', '')
        session_id = data.get('session_id')
        language = data.get('language', 'en')
        
        emotion = 'neutral'
        if session_id in active_sessions:
            emotion = active_sessions[session_id]['last_emotion']
        
        if not GEMINI_READY or not gemini_client:
            return {
                "response": "I'm here to help! (AI not available)" if language == 'en' 
                           else "Yardımcı olmak için buradayım! (AI mevcut değil)"
            }
        
        try:
            lang_instr = "Türkçe yanıt ver" if language == "tr" else "Respond in English"
            
            prompt = f"""You are a friendly, patient AI teacher assistant for elementary students.

Student's Message: {message}
Student's Emotion: {emotion}

{lang_instr}. Provide a helpful, warm, age-appropriate response (2-3 sentences).
- Be encouraging and supportive
- Keep explanations simple
- If student seems confused/frustrated, be extra patient
- Make learning fun!

Response:"""
            
            response_obj = gemini_client.models.generate_content(
                model=gemini_model_name,
                contents=prompt
            )
            ai_response = response_obj.text.strip()
            
        except Exception as e:
            print(f"Gemini chat error: {e}")
            ai_response = "I'm here to help you learn!" if language == 'en' else "Sana yardım etmek için buradayım!"
        
        return {"response": ai_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/progress/{session_id}")
async def get_progress(session_id: str):
    """Get student progress"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as total, 
                   SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct,
                   AVG(confidence) as avg_confidence
            FROM sessions 
            WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            "total_questions": result[0],
            "correct_answers": result[1] or 0,
            "accuracy": (result[1] / result[0] * 100) if result[0] > 0 else 0,
            "avg_confidence": result[2] or 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("🚀 AI Virtual Teacher - Web API")
    print(f"Server: http://localhost:8000")
    print(f"DeepFace (Emotion): {'✅ Enabled' if DEEPFACE_AVAILABLE else '❌ Disabled'}")
    print(f"Gemini AI: {'✅ Enabled (' + gemini_model_name + ')' if GEMINI_READY else '❌ Disabled'}")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")