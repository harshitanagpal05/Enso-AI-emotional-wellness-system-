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
        message = request.message.strip()
        message_lower = message.lower()
        mood = request.current_mood or "neutral"
        name = request.nickname or "friend"
        bio = request.bio or ""
        history = request.history or []
        
        # Build conversation context from history
        conversation_context = ""
        if history:
            conversation_context = "\n".join([
                f"{'User' if h.role == 'user' else 'Enso Buddy'}: {h.content}" 
                for h in history[-5:]
            ])
        
        # Get mood-based recommendations
        mood_map = {
            "positive state": "positive state", "low": "low",
            "frustrated and high stressed": "frustrated and high stressed",
            "anxious": "anxious", "strong dislike": "strong dislike",
            "shocking and unexpected wonders": "shocking and unexpected wonders",
            "stable and calm state": "stable and calm state",
            "sad": "low", "happy": "positive state",
            "angry": "frustrated and high stressed", "fear": "anxious",
            "neutral": "stable and calm state"
        }
        rec_key = mood_map.get(mood.lower(), "stable and calm state")
        sample_recs = get_recommendations(rec_key)
        
        # Analyze user intent with improved detection
        analysis = TextBlob(message)
        sentiment = analysis.sentiment.polarity
        
        # Detect question types
        is_question = any(message_lower.startswith(q) for q in ["what", "how", "why", "when", "where", "who", "can", "could", "would", "should", "is", "are", "do", "does", "will"])
        is_question = is_question or "?" in message
        
        # Specific topic detection
        topics = {
            "wellness": ["wellness", "health", "self-care", "self care", "wellbeing", "well-being"],
            "breathing": ["breath", "breathe", "breathing", "inhale", "exhale"],
            "meditation": ["meditat", "mindful", "calm", "relax", "peace"],
            "exercise": ["exercise", "workout", "physical", "fitness", "yoga", "stretch"],
            "sleep": ["sleep", "insomnia", "tired", "rest", "fatigue", "exhausted"],
            "anxiety": ["anxious", "anxiety", "panic", "nervous", "worry", "worried"],
            "stress": ["stress", "stressed", "overwhelm", "pressure", "tension"],
            "sadness": ["sad", "depress", "down", "lonely", "hopeless", "grief", "loss"],
            "anger": ["angry", "mad", "frustrated", "irritat", "annoy", "furious"],
            "happiness": ["happy", "joy", "excited", "grateful", "thankful", "great", "good", "awesome"],
            "relationship": ["friend", "family", "partner", "relationship", "colleague", "parent", "sibling"],
            "work": ["work", "job", "career", "boss", "office", "deadline", "project"],
            "motivation": ["motivat", "inspir", "goal", "purpose", "meaning", "drive"],
            "confidence": ["confiden", "self-esteem", "believe", "doubt", "insecur"],
            "recommendation": ["recommend", "suggest", "movie", "song", "music", "watch", "listen", "activity"]
        }
        
        detected_topics = []
        for topic, keywords in topics.items():
            if any(kw in message_lower for kw in keywords):
                detected_topics.append(topic)
        
        response = ""
        
        # DIRECT QUESTION ANSWERING - Respond to what user actually asks
        
        # Greeting
        if any(message_lower.startswith(g) for g in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]):
            response = f"Hello {name}! I'm Enso Buddy, your wellness companion. I can see your current mood is {mood}. How can I help you today? Feel free to ask me anything about wellness, mindfulness, or just share what's on your mind."
        
        # What can you do / Help
        elif is_question and any(w in message_lower for w in ["what can you do", "what do you do", "help me", "how can you help", "what are you"]):
            response = f"""I'm Enso Buddy, your AI wellness companion! Here's what I can do for you, {name}:

**Wellness Support:**
- Guide you through breathing exercises and meditation
- Provide stress management and relaxation techniques
- Offer advice on sleep, exercise, and self-care

**Emotional Support:**
- Listen when you need to talk about your feelings
- Help you process emotions like stress, anxiety, or sadness
- Celebrate your positive moments with you

**Personalized Recommendations:**
- Suggest calming music, uplifting movies, or activities based on your mood
- Your current mood ({mood}) helps me tailor my suggestions

**General Questions:**
- Answer questions about mental health topics
- Provide tips and techniques for various wellness areas

What would you like to explore?"""
        
        # Breathing exercises
        elif "breathing" in detected_topics:
            response = f"""Here's a simple breathing exercise for you, {name}:

**4-7-8 Breathing Technique:**
1. Breathe in quietly through your nose for **4 seconds**
2. Hold your breath for **7 seconds**
3. Exhale completely through your mouth for **8 seconds**
4. Repeat this cycle 3-4 times

This technique activates your parasympathetic nervous system and helps reduce anxiety. Would you like me to guide you through it step by step, or would you prefer a different technique like box breathing?"""
        
        # Meditation/Mindfulness
        elif "meditation" in detected_topics:
            response = f"""Here's a quick mindfulness exercise for you, {name}:

**5-4-3-2-1 Grounding Technique:**
Take a moment to notice:
- **5 things** you can SEE
- **4 things** you can TOUCH
- **3 things** you can HEAR
- **2 things** you can SMELL
- **1 thing** you can TASTE

This brings you back to the present moment and helps calm racing thoughts. Would you like a guided body scan meditation or a longer mindfulness practice instead?"""
        
        # Sleep issues
        elif "sleep" in detected_topics:
            response = f"""Sleep is so important for emotional wellbeing, {name}. Here are some evidence-based tips:

**Better Sleep Habits:**
1. **Consistent schedule** - Go to bed and wake up at the same time daily
2. **Wind-down routine** - Start relaxing 1 hour before bed (dim lights, no screens)
3. **Cool, dark room** - Optimal temperature is 65-68°F (18-20°C)
4. **Avoid stimulants** - No caffeine after 2pm, limit alcohol

**If you can't sleep:**
- Try the 4-7-8 breathing technique
- Don't lie in bed awake for more than 20 minutes - get up and do something calm
- Avoid checking the clock

What specific sleep challenge are you facing? I can give more targeted advice."""
        
        # Anxiety
        elif "anxiety" in detected_topics:
            response = f"""I understand anxiety can be overwhelming, {name}. Here are some immediate techniques:

**Quick Anxiety Relief:**
1. **Ground yourself** - Feel your feet on the floor, notice 5 things around you
2. **Slow your breathing** - Breathe out longer than you breathe in
3. **Challenge the thought** - Ask "Is this thought true? What's the evidence?"
4. **Name it to tame it** - Say "I notice I'm feeling anxious" to create distance

**Longer-term strategies:**
- Regular exercise (even 10-minute walks help)
- Limit caffeine and alcohol
- Practice mindfulness daily
- Talk to someone you trust

What's making you feel anxious right now? I'm here to listen."""
        
        # Stress
        elif "stress" in detected_topics:
            suggestions = "\n".join([f"- {r['content']}" for r in sample_recs[:3] if r['type'] in ['activity', 'music']]) if sample_recs else ""
            response = f"""Stress management is crucial, {name}. Here are effective strategies:

**Immediate Stress Relief:**
1. **Take 3 deep breaths** - Activates your relaxation response
2. **Progressive muscle relaxation** - Tense and release each muscle group
3. **Step outside** - Even 5 minutes of fresh air helps
4. **Move your body** - Stretch, walk, or shake it out

**Based on your mood, I'd also suggest:**
{suggestions}

What's causing your stress? Sometimes talking it through helps."""
        
        # Sadness
        elif "sadness" in detected_topics:
            response = f"""I'm sorry you're feeling down, {name}. Your feelings are valid. Here's what might help:

**When You're Feeling Sad:**
1. **Allow yourself to feel** - It's okay to cry or feel low
2. **Reach out** - Talk to a friend, family member, or professional
3. **Be gentle with yourself** - Do one small kind thing for yourself today
4. **Move a little** - Even a short walk can shift your mood
5. **Avoid isolation** - Being around people (even quietly) can help

**Remember:** Sadness is temporary, even when it doesn't feel that way. If these feelings persist for more than two weeks, please consider talking to a mental health professional.

Would you like to talk about what's making you feel this way? I'm here to listen without judgment."""
        
        # Anger
        elif "anger" in detected_topics:
            response = f"""Anger is a natural emotion, {name}. Here's how to work with it constructively:

**Immediate Anger Management:**
1. **Pause before reacting** - Count to 10 slowly
2. **Walk away temporarily** - Give yourself space
3. **Physical release** - Exercise, punch a pillow, or do jumping jacks
4. **Cool down literally** - Splash cold water on your face

**Process the Anger:**
1. What triggered this feeling?
2. What need isn't being met?
3. What's in your control to change?

**Express it healthily:**
- Use "I feel..." statements instead of blaming
- Write it out in a journal
- Talk to someone you trust

What's making you feel angry? Sometimes naming it helps."""
        
        # Happiness/Positive
        elif "happiness" in detected_topics:
            response = f"""That's wonderful to hear, {name}! Let's build on that positive energy.

**Ways to Sustain Your Good Mood:**
1. **Savor it** - Take a moment to really notice how good you feel
2. **Write it down** - Journaling about positive moments strengthens them
3. **Share it** - Tell someone about what's making you happy
4. **Do more of what's working** - Identify what contributed to this feeling

**Spread the positivity:**
- Send a kind message to someone
- Help someone out
- Express gratitude

What's bringing you joy today? I'd love to hear about it!"""
        
        # Recommendations request
        elif "recommendation" in detected_topics:
            movie_recs = [r for r in sample_recs if r['type'] == 'movie'][:2]
            music_recs = [r for r in sample_recs if r['type'] == 'music'][:2]
            activity_recs = [r for r in sample_recs if r['type'] == 'activity'][:2]
            
            response = f"""Based on your {mood} mood, here are my personalized recommendations for you, {name}:

**Movies to Watch:**
{chr(10).join([f"- {r['content']} - {r['reason']}" for r in movie_recs]) if movie_recs else "- Check out something that matches your mood!"}

**Music to Listen:**
{chr(10).join([f"- {r['content']} - {r['reason']}" for r in music_recs]) if music_recs else "- Try some calming instrumentals"}

**Activities to Try:**
{chr(10).join([f"- {r['content']} - {r['reason']}" for r in activity_recs]) if activity_recs else "- Take a mindful walk"}

Would you like more specific recommendations for any category?"""
        
        # Work/Career stress
        elif "work" in detected_topics:
            response = f"""Work challenges can be draining, {name}. Here are some strategies:

**Managing Work Stress:**
1. **Prioritize** - Focus on 3 most important tasks each day
2. **Take breaks** - Short breaks every 90 minutes improve focus
3. **Set boundaries** - Define when work ends for the day
4. **Communicate** - Talk to your manager if workload is unsustainable

**After Work Recovery:**
- Have a transition ritual (change clothes, take a walk)
- Avoid checking emails in evening
- Do something you enjoy daily

What's happening at work? I'm here to help you think through it."""
        
        # Relationship issues
        elif "relationship" in detected_topics:
            response = f"""Relationships are complex, {name}. Here's some guidance:

**Healthy Communication:**
1. **Listen actively** - Focus on understanding, not just responding
2. **Use "I" statements** - "I feel..." instead of "You always..."
3. **Pick your timing** - Important conversations need calm moments
4. **Validate feelings** - Even if you disagree, acknowledge their perspective

**Self-reflection questions:**
- What do I need from this relationship?
- Am I communicating my needs clearly?
- What's my part in this situation?

Would you like to talk about a specific relationship situation?"""
        
        # Motivation
        elif "motivation" in detected_topics:
            response = f"""Motivation can ebb and flow, {name}. Here's how to build it:

**Finding Motivation:**
1. **Start tiny** - Commit to just 2 minutes of any task
2. **Connect to your "why"** - Why does this matter to you?
3. **Remove friction** - Make the first step as easy as possible
4. **Track progress** - Small wins build momentum
5. **Reward yourself** - Celebrate completed tasks

**When motivation is low:**
- Rely on discipline, not just motivation
- Change your environment
- Do the task you're avoiding first

What are you trying to find motivation for? I can help you break it down."""
        
        # Confidence
        elif "confidence" in detected_topics:
            response = f"""Building confidence is a journey, {name}. Here are some strategies:

**Building Self-Confidence:**
1. **Challenge negative self-talk** - Would you say that to a friend?
2. **Recall past successes** - You've overcome challenges before
3. **Prepare thoroughly** - Competence builds confidence
4. **Power poses** - Stand tall for 2 minutes before challenging situations
5. **Accept imperfection** - Confidence isn't about being perfect

**Daily practices:**
- List 3 things you did well today
- Accept compliments gracefully
- Set and achieve small goals

What area would you like to feel more confident in?"""
        
        # General question - provide helpful response
        elif is_question:
            response = f"""That's a great question, {name}. Let me help you with that.

Based on your {mood} mood and what you're asking, here's my perspective:

{message} is something many people wonder about. While I'm a wellness companion (not a search engine), I can offer support in areas like:
- Emotional wellbeing and mental health
- Stress management and relaxation
- Sleep, exercise, and self-care
- Relationships and communication
- Motivation and goal-setting

Could you tell me more about what you're looking for? I want to give you the most helpful response."""
        
        # General sharing/venting - be supportive
        elif sentiment < -0.2:
            response = f"""I hear you, {name}. It sounds like you're going through something difficult. 

Thank you for sharing that with me. Whatever you're feeling is valid. You don't have to have it all figured out.

Sometimes it helps to:
- Put feelings into words (which you're doing!)
- Take things one moment at a time
- Be as kind to yourself as you'd be to a friend

What would feel most supportive right now - talking more about it, or would you like some calming techniques?"""
        
        # Positive sharing
        elif sentiment > 0.2:
            response = f"""That's great to hear, {name}! Your positive energy comes through.

It's wonderful when things are going well. Taking time to notice and appreciate good moments actually strengthens their impact on your wellbeing.

What's contributing to this positive feeling? I'd love to hear more!"""
        
        # Default - acknowledge and ask clarifying question
        else:
            response = f"""Thanks for sharing, {name}. I want to make sure I understand and give you the most helpful response.

Your current mood is {mood}, and I'm here to support you however I can. 

Could you tell me more about:
- What's on your mind right now?
- Is there something specific you'd like help with?
- Or would you just like to chat?

I can help with wellness tips, emotional support, recommendations, or just listen."""
        
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
