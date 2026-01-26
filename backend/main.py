from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import detect_emotion_from_image, map_emotion_to_mood
from recommendations import get_recommendations, RECOMMENDATIONS_POOL
import uvicorn

app = FastAPI(title="AI Emotion Wellness API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "AI Emotion Wellness System API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "emotion-detection"}

from pydantic import BaseModel
from typing import Optional

class EmotionRequest(BaseModel):
    emotion: str

@app.post("/detect")
async def detect_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        emotion, confidence = detect_emotion_from_image(contents)
        
        if emotion is None:
            raise HTTPException(status_code=400, detail="Could not detect face or emotion")
        
        mood = map_emotion_to_mood(emotion)
        recommendations = get_recommendations(mood)
        
        return {
            "emotion": emotion,
            "confidence": float(confidence),
            "mood": mood,
            "recommendations": recommendations,
            "disclaimer": "This system is not a medical diagnostic tool. It provides emotional wellness support only."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manual-update")
async def manual_update(request: EmotionRequest):
    try:
        emotion = request.emotion
        mood = map_emotion_to_mood(emotion)
        recommendations = get_recommendations(mood)
        
        return {
            "emotion": emotion,
            "confidence": 1.0,
            "mood": mood,
            "recommendations": recommendations,
            "disclaimer": "This system is not a medical diagnostic tool. It provides emotional wellness support only."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import re
from textblob import TextBlob

class TextRequest(BaseModel):
    text: str
    entry_id: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[list[ChatMessage]] = []
    current_mood: Optional[str] = None
    nickname: Optional[str] = None
    bio: Optional[str] = None

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Simulate AI logic based on message and mood
        message = request.message.lower().strip()
        mood = request.current_mood or "stable and calm state"
        name = request.nickname or "friend"
        bio = request.bio or ""
        
        # Map mood to recommendation key
        mood_map = {
            "positive state": "positive state",
            "low": "low",
            "frustrated and high stressed": "frustrated and high stressed",
            "anxious": "anxious",
            "strong dislike": "strong dislike",
            "shocking and unexpected wonders": "shocking and unexpected wonders",
            "stable and calm state": "stable and calm state",
            "sad": "low",
            "happy": "positive state",
            "angry": "frustrated and high stressed",
            "fear": "anxious",
            "neutral": "stable and calm state"
        }
        rec_key = mood_map.get(mood.lower(), "stable and calm state")
        
        # Simple sentiment analysis for the current message
        analysis = TextBlob(message)
        sentiment = analysis.sentiment.polarity
        
        response = ""
        
        # Helper to check for whole words
        def has_word(word, text):
            return bool(re.search(rf"\b{word}\b", text))
        
        # Helper to get recommendations as formatted text
        def get_suggestions_text(limit=3):
            sample_recs = get_recommendations(rec_key)
            suggestions_list = []
            if sample_recs:
                # Prioritize quote, music, activity
                for rec_type in ["quote", "music", "activity", "movie"]:
                    type_recs = [r for r in sample_recs if r["type"] == rec_type]
                    if type_recs and len(suggestions_list) < limit:
                        rec = type_recs[0]
                        suggestions_list.append(f"• {rec['content']} - {rec['reason']}")
            return "\n".join(suggestions_list) if suggestions_list else "I have some personalized recommendations ready for you!"
        
        # Check for common greetings
        if has_word("hello", message) or has_word("hi", message) or has_word("hey", message) or has_word("greetings", message):
            suggestions = get_suggestions_text(2)
            greetings = [
                f"Hello {name}! I'm Enso Buddy, your wellness companion. I'm here to support you in whatever you're feeling today.\n\nI noticed you've been feeling {mood} lately. Here are a couple of things that might help:\n\n{suggestions}\n\nWhat can I do for you right now? You can ask me for wellness tips, tell me about your day, or we can just chat.",
                f"Hi {name}! I'm Enso Buddy. I see you've been feeling {mood} recently. Based on that, I'd suggest:\n\n{suggestions}\n\nHow are you doing today? I'm here to listen if you want to share anything.",
                f"Hey {name}! Great to see you. Since your recent mood has been {mood}, here's something that might help:\n\n{suggestions}\n\nI can help with mindfulness, provide personalized recommendations, or just be a supportive ear. How's your day going?"
            ]
            response = greetings[hash(message) % len(greetings)]
        
        # Check for capabilities / how it works
        elif (has_word("what", message) and (has_word("do", message) or has_word("can", message))) or \
             (has_word("how", message) and (has_word("work", message) or has_word("help", message))):
            response = (
                f"I'm Enso Buddy, your AI wellness companion, {name}! I'm designed to help you maintain emotional balance. Here's exactly how I can help:\n\n"
                "1. **Analyze Your Mood**: Go to the Home page and capture a photo. I'll use AI to detect your emotion.\n"
                "2. **Personalized Recommendations**: Based on your mood ({mood}), I suggest specific music, quotes, and wellness activities.\n"
                "3. **Mood Journaling**: You can write notes about your day, and I'll analyze the sentiment to give you insights.\n"
                "4. **Wellness Challenges**: I track your progress on mindfulness and positivity challenges in your Dashboard.\n"
                "5. **Mindfulness Exercises**: Ask me for a 'breathing exercise' or 'grounding technique' right here!\n\n"
                "What would you like to start with? I can give you a tip for your current {mood} state right now."
            ).format(mood=mood)
        
        # Check for specific "do something" or "help me with X" requests
        elif has_word("recommend", message) or has_word("suggest", message) or has_word("tips", message) or has_word("advice", message):
            suggestions = get_suggestions_text(4)
            response = (
                f"I've got some personalized recommendations for you, {name}! Since you've been in a {mood} state, "
                f"these might be particularly helpful:\n\n{suggestions}\n\n"
                "Would you like more options for a different category, like just music or just activities?"
            )
        
        # Check for gratitude
        elif has_word("thank", message) or has_word("thanks", message) or has_word("appreciate", message):
            gratitude_responses = [
                f"You're very welcome, {name}! I'm always here for you.",
                f"You're so welcome, {name}! It means a lot that you're taking care of yourself. Keep it up!",
                f"Anytime, {name}! I'm glad I could help. Remember, I'm here whenever you need me."
            ]
            response = gratitude_responses[hash(message) % len(gratitude_responses)]
        
        # Check for questions about feelings/emotions
        elif has_word("feel", message) or has_word("feeling", message) or has_word("emotion", message) or has_word("mood", message):
            suggestions = get_suggestions_text(3)
            feeling_responses = [
                f"I understand, {name}. It sounds like you're processing some emotions. Since you've been feeling {mood} lately, this is completely valid. Here are some suggestions:\n\n{suggestions}\n\nWould you like to talk about what's contributing to that?",
                f"Feelings can be complex, {name}, and your {mood} state tells me you're going through something. Based on your mood, I'd suggest:\n\n{suggestions}\n\nI'm here to listen. How does your current mood relate to what you're experiencing right now?",
                f"Emotions are valid, {name}, whatever they are. Your {mood} mood is important. Here's what might help:\n\n{suggestions}\n\nWould you like to explore what you're feeling right now? I can help you process it."
            ]
            response = feeling_responses[hash(message) % len(feeling_responses)]
        
        # Check for stress/anxiety keywords
        elif has_word("stress", message) or has_word("stressed", message) or has_word("anxious", message) or has_word("anxiety", message) or has_word("worried", message) or has_word("worry", message) or has_word("tension", message):
            suggestions = get_suggestions_text(3)
            stress_responses = [
                f"I hear you, {name}. Stress and anxiety can be overwhelming. Since you're feeling {mood}, here are some suggestions:\n\n{suggestions}\n\nHave you tried any breathing exercises? I can guide you through one if you'd like.",
                f"It's completely normal to feel stressed or anxious sometimes, {name}. Based on your {mood} mood, I'd suggest:\n\n{suggestions}\n\nWhat's been causing you the most worry lately?",
                f"Stress is tough, {name}. Remember, it's okay to take breaks and practice self-care. Here's what might help with your {mood} state:\n\n{suggestions}\n\nWhat's one thing that usually helps you feel calmer?"
            ]
            response = stress_responses[hash(message) % len(stress_responses)]
        
        # Check for sadness/depression keywords
        elif has_word("sad", message) or has_word("depressed", message) or has_word("down", message) or has_word("lonely", message) or has_word("hopeless", message) or has_word("upset", message):
            suggestions = get_suggestions_text(3)
            sad_responses = [
                f"I'm sorry you're going through this, {name}. It's okay to feel sad. You're not alone. Since you're feeling {mood}, here are some suggestions:\n\n{suggestions}\n\nWould you like to talk about what's making you feel this way?",
                f"I'm here with you, {name}. Sadness is a valid emotion, and it's okay to not be okay. Based on your {mood} mood, I'd suggest:\n\n{suggestions}\n\nWhat's been weighing on you?",
                f"I can sense you're going through a tough time, {name}. Remember, this feeling won't last forever. Here's what might help:\n\n{suggestions}\n\nWhat's one small thing that might help you feel a bit better today?"
            ]
            response = sad_responses[hash(message) % len(sad_responses)]
        
        # Check for anger/frustration keywords
        elif has_word("angry", message) or has_word("mad", message) or has_word("frustrated", message) or has_word("frustration", message) or has_word("annoyed", message) or has_word("irritated", message):
            suggestions = get_suggestions_text(3)
            anger_responses = [
                f"I understand that anger can be intense, {name}. It's okay to feel this way. Since you're feeling {mood}, here are some suggestions:\n\n{suggestions}\n\nWhat's been triggering these feelings?",
                f"Anger is a natural emotion, {name}. Sometimes it helps to express it safely. Based on your {mood} mood, I'd suggest:\n\n{suggestions}\n\nWould you like to talk about what's making you feel this way?",
                f"I hear your frustration, {name}. It's valid. Here's what might help with your {mood} state:\n\n{suggestions}\n\nWhat would help you process these feelings right now?"
            ]
            response = anger_responses[hash(message) % len(anger_responses)]
        
        # Check for positive keywords
        elif has_word("happy", message) or has_word("good", message) or has_word("great", message) or has_word("excited", message) or has_word("joy", message) or has_word("grateful", message):
            positive_responses = [
                f"That's wonderful to hear, {name}! Your positive energy is contagious. What's making you feel this way?",
                f"I love hearing that, {name}! It's great that you're in a good space. How can we keep this momentum going?",
                f"That's fantastic, {name}! Positive moments are worth celebrating. What's contributing to your good mood today?"
            ]
            response = positive_responses[hash(message) % len(positive_responses)]
        
        # Check for breathing/mindfulness requests
        elif has_word("breath", message) or has_word("breathe", message) or has_word("meditation", message) or has_word("mindful", message) or has_word("calm", message):
            mindfulness_responses = [
                f"Great idea, {name}! Breathing exercises can really help. Try this: inhale for 4 counts, hold for 4, exhale for 4. Repeat a few times. How does that feel?",
                f"Mindfulness is a powerful tool, {name}. Would you like to try a quick grounding exercise? Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
                f"Taking time to breathe and be present is so important, {name}. What kind of mindfulness practice would you like to explore?"
            ]
            response = mindfulness_responses[hash(message) % len(mindfulness_responses)]
        
        # Negative sentiment handling
        elif sentiment < -0.3:
            if mood in ["sad", "angry", "fear", "low"]:
                response = f"I can feel that things are tough right now, {name}, especially since you've been feeling {mood}. It's okay to let it out. What's on your mind? What would help you feel supported right now?"
            else:
                response = f"I can sense some heavy emotions in your words, {name}. Do you want to talk more about what's bothering you? I'm here to listen without judgment."
        
        # Positive sentiment handling
        elif sentiment > 0.3:
            positive_sentiment_responses = [
                f"That's wonderful to hear, {name}! Your positive energy is contagious. What's making you feel this way?",
                f"I love your positivity, {name}! It's great to see you in such a good space. How can we keep this momentum going?",
                f"That's fantastic, {name}! It sounds like things are going well. What's contributing to your good mood?"
            ]
            response = positive_sentiment_responses[hash(message) % len(positive_sentiment_responses)]
        
        # Check for bio-related personalization
        elif bio and any(word in message for word in bio.lower().split()):
            response = f"I noticed you mentioned something related to what you shared in your bio: '{bio}'. How does that tie into how you're feeling today, {name}?"
        
        # Default responses based on mood with recommendations - ALWAYS provide suggestions
        else:
            suggestions = get_suggestions_text(3)
            
            if mood == "sad" or mood == "low" or "low" in mood.lower():
                default_sad = [
                    f"I'm here with you, {name}. It's okay to not be okay. Since you've been feeling {mood}, remember this feeling is temporary. Here are some suggestions:\n\n{suggestions}\n\nWhat's on your mind?",
                    f"I'm listening, {name}. I understand you're going through a tough time. Your {mood} mood tells me you might need extra support. Based on your mood, I'd suggest:\n\n{suggestions}\n\nWould you like to talk about it?",
                    f"You're not alone, {name}. I can see you've been feeling {mood} lately. Here's what might help:\n\n{suggestions}\n\nWhat would feel helpful - talking, a mindfulness exercise, or something else?"
                ]
                response = default_sad[hash(message) % len(default_sad)]
            elif mood == "happy" or mood == "positive state" or "positive" in mood.lower():
                default_happy = [
                    f"It's great to see you in such a good mood, {name}! Your {mood} energy is wonderful. To keep this momentum going, I suggest:\n\n{suggestions}\n\nHow can we keep this positive energy flowing?",
                    f"I love your positive energy, {name}! Since you're feeling {mood}, this is perfect for trying something new. Here are my recommendations:\n\n{suggestions}\n\nWhat's been making you feel so good?",
                    f"That's wonderful, {name}! It's great when we're in a positive space like {mood}. Here's what might enhance your mood:\n\n{suggestions}\n\nWhat would you like to explore or talk about?"
                ]
                response = default_happy[hash(message) % len(default_happy)]
            elif "frustrated" in mood.lower() or "stressed" in mood.lower() or mood == "anxious" or "anxious" in mood.lower():
                default_stress = [
                    f"I can sense tension, {name}. Your {mood} state suggests you need release. Here are some suggestions:\n\n{suggestions}\n\nWhat's been on your mind? Sometimes talking helps.",
                    f"It sounds overwhelming, {name}. Since you're feeling {mood}, let's find ways to ground yourself. Based on your mood:\n\n{suggestions}\n\nWhat would help you feel more centered?",
                    f"I'm here for you, {name}. Your {mood} mood tells me you're dealing with a lot. Here's what might help:\n\n{suggestions}\n\nWhat's one thing causing stress? We can work through it."
                ]
                response = default_stress[hash(message) % len(default_stress)]
            else:
                default_responses = [
                    f"I'm listening, {name}. I notice you've been feeling {mood} lately. Here are some suggestions based on your mood:\n\n{suggestions}\n\nTell me more. How are you processing everything?",
                    f"I'm here for you, {name}. Your current {mood} state is valid. Based on your mood, I'd suggest:\n\n{suggestions}\n\nWhat's on your mind? Feel free to share.",
                    f"I'm listening, {name}. Since you're in a {mood} state, I want to understand. Here's what might help:\n\n{suggestions}\n\nWhat would you like to talk about?",
                    f"Tell me more, {name}. I'm here to listen and help. Based on your {mood} mood:\n\n{suggestions}\n\nYour mood matters, and I want to support you."
                ]
                response = default_responses[hash(message) % len(default_responses)]
        
        # Ensure we always have a response
        if not response:
            response = f"I'm here for you, {name}. I notice you've been feeling {mood} lately. Can you tell me more? I'm listening."

        return {"response": response}
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-text")
async def analyze_text(request: TextRequest):
    try:
        analysis = TextBlob(request.text)
        score = analysis.sentiment.polarity # -1 to 1
        
        label = "neutral"
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
            
        return {
            "score": float(score),
            "label": label,
            "insight": get_text_insight(label, score)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_text_insight(label: str, score: float) -> str:
    if score > 0.6:
        return "Your words radiate such positivity! This is a wonderful state of mind."
    elif score > 0.1:
        return "I can sense a positive tone in your writing. It's good to focus on the bright side."
    elif score < -0.6:
        return "Your writing seems quite heavy today. Remember that it's okay to feel this way, and taking it out on paper is a great first step."
    elif score < -0.1:
        return "There's a bit of a low or stressed tone in your journal. How can we make today a little lighter?"
    else:
        return "Your journal entry is very balanced and reflective."

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
