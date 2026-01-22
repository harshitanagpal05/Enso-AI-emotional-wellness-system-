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

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Simulate AI logic based on message and mood
        message = request.message.lower().strip()
        mood = request.current_mood or "stable and calm state"
        
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
                f"Hello! I'm Enso Buddy. I noticed you've been feeling {mood} lately. Here are some suggestions that might help:\n\n{suggestions}\n\nHow can I support you today?",
                f"Hi there! I'm here for you. I see you've been feeling {mood} recently. Based on your mood, I'd suggest:\n\n{suggestions}\n\nWhat's on your mind?",
                f"Hey! Great to see you. Since you're feeling {mood}, here's what might help:\n\n{suggestions}\n\nHow are you doing today? I'm here to listen and help."
            ]
            response = greetings[hash(message) % len(greetings)]
        
        # Check for help requests - EXPANDED to catch more variations
        elif (has_word("help", message) or has_word("support", message) or has_word("advice", message) or 
              has_word("suggest", message) or has_word("recommend", message) or has_word("what", message) or
              has_word("how", message) or has_word("can you", message) or has_word("should", message) or
              has_word("need", message) or has_word("want", message) or has_word("give", message)):
            suggestions = get_suggestions_text(4)
            help_responses = [
                f"I'm here to listen and help! Since you've been feeling {mood}, here are personalized suggestions:\n\n{suggestions}\n\nWould you like to talk about your day, or need more specific wellness tips?",
                f"I'm here for you! Based on your {mood} mood, here are my recommendations:\n\n{suggestions}\n\nYou can share what's on your mind, ask for more tips, or we can just chat. What would help you most right now?",
                f"I'm here to support you! Given your {mood} state, here's what might help:\n\n{suggestions}\n\nFeel free to share what's bothering you, or ask me about mindfulness techniques, breathing exercises, or anything else on your mind."
            ]
            response = help_responses[hash(message) % len(help_responses)]
        
        # Check for gratitude
        elif has_word("thank", message) or has_word("thanks", message) or has_word("appreciate", message):
            gratitude_responses = [
                "You're very welcome! I'm always here for you.",
                "You're so welcome! It means a lot that you're taking care of yourself. Keep it up!",
                "Anytime! I'm glad I could help. Remember, I'm here whenever you need me."
            ]
            response = gratitude_responses[hash(message) % len(gratitude_responses)]
        
        # Check for questions about feelings/emotions
        elif has_word("feel", message) or has_word("feeling", message) or has_word("emotion", message) or has_word("mood", message):
            suggestions = get_suggestions_text(3)
            feeling_responses = [
                f"I understand. It sounds like you're processing some emotions. Since you've been feeling {mood} lately, this is completely valid. Here are some suggestions:\n\n{suggestions}\n\nWould you like to talk about what's contributing to that?",
                f"Feelings can be complex, and your {mood} state tells me you're going through something. Based on your mood, I'd suggest:\n\n{suggestions}\n\nI'm here to listen. How does your current mood relate to what you're experiencing right now?",
                f"Emotions are valid, whatever they are. Your {mood} mood is important. Here's what might help:\n\n{suggestions}\n\nWould you like to explore what you're feeling right now? I can help you process it."
            ]
            response = feeling_responses[hash(message) % len(feeling_responses)]
        
        # Check for stress/anxiety keywords
        elif has_word("stress", message) or has_word("stressed", message) or has_word("anxious", message) or has_word("anxiety", message) or has_word("worried", message) or has_word("worry", message) or has_word("tension", message):
            suggestions = get_suggestions_text(3)
            stress_responses = [
                f"I hear you. Stress and anxiety can be overwhelming. Since you're feeling {mood}, here are some suggestions:\n\n{suggestions}\n\nHave you tried any breathing exercises? I can guide you through one if you'd like.",
                f"It's completely normal to feel stressed or anxious sometimes. Based on your {mood} mood, I'd suggest:\n\n{suggestions}\n\nWhat's been causing you the most worry lately?",
                f"Stress is tough. Remember, it's okay to take breaks and practice self-care. Here's what might help with your {mood} state:\n\n{suggestions}\n\nWhat's one thing that usually helps you feel calmer?"
            ]
            response = stress_responses[hash(message) % len(stress_responses)]
        
        # Check for sadness/depression keywords
        elif has_word("sad", message) or has_word("depressed", message) or has_word("down", message) or has_word("lonely", message) or has_word("hopeless", message) or has_word("upset", message):
            suggestions = get_suggestions_text(3)
            sad_responses = [
                f"I'm sorry you're going through this. It's okay to feel sad. You're not alone. Since you're feeling {mood}, here are some suggestions:\n\n{suggestions}\n\nWould you like to talk about what's making you feel this way?",
                f"I'm here with you. Sadness is a valid emotion, and it's okay to not be okay. Based on your {mood} mood, I'd suggest:\n\n{suggestions}\n\nWhat's been weighing on you?",
                f"I can sense you're going through a tough time. Remember, this feeling won't last forever. Here's what might help:\n\n{suggestions}\n\nWhat's one small thing that might help you feel a bit better today?"
            ]
            response = sad_responses[hash(message) % len(sad_responses)]
        
        # Check for anger/frustration keywords
        elif has_word("angry", message) or has_word("mad", message) or has_word("frustrated", message) or has_word("frustration", message) or has_word("annoyed", message) or has_word("irritated", message):
            suggestions = get_suggestions_text(3)
            anger_responses = [
                f"I understand that anger can be intense. It's okay to feel this way. Since you're feeling {mood}, here are some suggestions:\n\n{suggestions}\n\nWhat's been triggering these feelings?",
                f"Anger is a natural emotion. Sometimes it helps to express it safely. Based on your {mood} mood, I'd suggest:\n\n{suggestions}\n\nWould you like to talk about what's making you feel this way?",
                f"I hear your frustration. It's valid. Here's what might help with your {mood} state:\n\n{suggestions}\n\nWhat would help you process these feelings right now?"
            ]
            response = anger_responses[hash(message) % len(anger_responses)]
        
        # Check for positive keywords
        elif has_word("happy", message) or has_word("good", message) or has_word("great", message) or has_word("excited", message) or has_word("joy", message) or has_word("grateful", message):
            positive_responses = [
                "That's wonderful to hear! Your positive energy is contagious. What's making you feel this way?",
                "I love hearing that! It's great that you're in a good space. How can we keep this momentum going?",
                "That's fantastic! Positive moments are worth celebrating. What's contributing to your good mood today?"
            ]
            response = positive_responses[hash(message) % len(positive_responses)]
        
        # Check for breathing/mindfulness requests
        elif has_word("breath", message) or has_word("breathe", message) or has_word("meditation", message) or has_word("mindful", message) or has_word("calm", message):
            mindfulness_responses = [
                "Great idea! Breathing exercises can really help. Try this: inhale for 4 counts, hold for 4, exhale for 4. Repeat a few times. How does that feel?",
                "Mindfulness is a powerful tool. Would you like to try a quick grounding exercise? Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
                "Taking time to breathe and be present is so important. What kind of mindfulness practice would you like to explore?"
            ]
            response = mindfulness_responses[hash(message) % len(mindfulness_responses)]
        
        # Negative sentiment handling
        elif sentiment < -0.3:
            if mood in ["sad", "angry", "fear", "low"]:
                response = f"I can feel that things are tough right now, especially since you've been feeling {mood}. It's okay to let it out. What's on your mind? What would help you feel supported right now?"
            else:
                response = "I can sense some heavy emotions in your words. Do you want to talk more about what's bothering you? I'm here to listen without judgment."
        
        # Positive sentiment handling
        elif sentiment > 0.3:
            positive_sentiment_responses = [
                "That's wonderful to hear! Your positive energy is contagious. What's making you feel this way?",
                "I love your positivity! It's great to see you in such a good space. How can we keep this momentum going?",
                "That's fantastic! It sounds like things are going well. What's contributing to your good mood?"
            ]
            response = positive_sentiment_responses[hash(message) % len(positive_sentiment_responses)]
        
        # Default responses based on mood with recommendations - ALWAYS provide suggestions
        else:
            suggestions = get_suggestions_text(3)
            
            if mood == "sad" or mood == "low" or "low" in mood.lower():
                default_sad = [
                    f"I'm here with you. It's okay to not be okay. Since you've been feeling {mood}, remember this feeling is temporary. Here are some suggestions:\n\n{suggestions}\n\nWhat's on your mind?",
                    f"I'm listening. I understand you're going through a tough time. Your {mood} mood tells me you might need extra support. Based on your mood, I'd suggest:\n\n{suggestions}\n\nWould you like to talk about it?",
                    f"You're not alone. I can see you've been feeling {mood} lately. Here's what might help:\n\n{suggestions}\n\nWhat would feel helpful - talking, a mindfulness exercise, or something else?"
                ]
                response = default_sad[hash(message) % len(default_sad)]
            elif mood == "happy" or mood == "positive state" or "positive" in mood.lower():
                default_happy = [
                    f"It's great to see you in such a good mood! Your {mood} energy is wonderful. To keep this momentum going, I suggest:\n\n{suggestions}\n\nHow can we keep this positive energy flowing?",
                    f"I love your positive energy! Since you're feeling {mood}, this is perfect for trying something new. Here are my recommendations:\n\n{suggestions}\n\nWhat's been making you feel so good?",
                    f"That's wonderful! It's great when we're in a positive space like {mood}. Here's what might enhance your mood:\n\n{suggestions}\n\nWhat would you like to explore or talk about?"
                ]
                response = default_happy[hash(message) % len(default_happy)]
            elif "frustrated" in mood.lower() or "stressed" in mood.lower() or mood == "anxious" or "anxious" in mood.lower():
                default_stress = [
                    f"I can sense tension. Your {mood} state suggests you need release. Here are some suggestions:\n\n{suggestions}\n\nWhat's been on your mind? Sometimes talking helps.",
                    f"It sounds overwhelming. Since you're feeling {mood}, let's find ways to ground yourself. Based on your mood:\n\n{suggestions}\n\nWhat would help you feel more centered?",
                    f"I'm here for you. Your {mood} mood tells me you're dealing with a lot. Here's what might help:\n\n{suggestions}\n\nWhat's one thing causing stress? We can work through it."
                ]
                response = default_stress[hash(message) % len(default_stress)]
            else:
                default_responses = [
                    f"I'm listening. I notice you've been feeling {mood} lately. Here are some suggestions based on your mood:\n\n{suggestions}\n\nTell me more. How are you processing everything?",
                    f"I'm here for you. Your current {mood} state is valid. Based on your mood, I'd suggest:\n\n{suggestions}\n\nWhat's on your mind? Feel free to share.",
                    f"I'm listening. Since you're in a {mood} state, I want to understand. Here's what might help:\n\n{suggestions}\n\nWhat would you like to talk about?",
                    f"Tell me more. I'm here to listen and help. Based on your {mood} mood:\n\n{suggestions}\n\nYour mood matters, and I want to support you."
                ]
                response = default_responses[hash(message) % len(default_responses)]
        
        # Ensure we always have a response
        if not response:
            response = f"I'm here for you. I notice you've been feeling {mood} lately. Can you tell me more? I'm listening."

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
