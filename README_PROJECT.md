# AI Emotion & Mood Detection + Personalized Recommendation System

## Overview
This system is a comprehensive emotional wellness support platform that uses real-time facial expression recognition to detect user emotions and provide personalized wellness recommendations (music, movies, activities, and quotes).

## Core Features
- **Real-time Emotion Detection**: Uses a pre-trained CNN model (via the `FER` library) to analyze webcam captures or uploaded images.
- **Mood Mapping**: Transparent rule-based logic that maps complex emotions (Happy, Sad, Angry, etc.) to mental states (Positive, Low Mood, High Stress, Stable).
- **Personalized Recommendations**: A hybrid engine that suggests content based on the current mood, with a feedback loop to track effectiveness.
- **Wellness Dashboard**: Visual analytics using Recharts to track mood trends, emotion distribution, and recommendation success rates.
- **Secure Authentication**: Integrated with Supabase Auth for user management and data privacy.

## Tech Stack
- **Frontend**: Next.js 15, TypeScript, Tailwind CSS, Shadcn/UI, Recharts.
- **Backend**: FastAPI (Python), FER (Facial Expression Recognition), OpenCV.
- **Database**: Supabase (PostgreSQL) for storing mood history, recommendations, and feedback.

## Why Pre-trained Models?
- **Efficiency**: Leveraging state-of-the-art pre-trained models (FER2013-based CNNs) ensures high accuracy without the need for massive datasets or extensive training resources.
- **Ethical Compliance**: Avoids the bias issues often associated with training small-scale custom models for emotion recognition.
- **Production Readiness**: Pre-trained models are optimized for inference, making the system responsive and reliable for real-time use.

## Ethical Considerations & Limitations
- **Wellness Only**: This system is **not a medical diagnostic tool**. It is designed for emotional support and self-reflection.
- **Privacy**: All captured images are processed in-memory and not stored on the server. Only the metadata (detected emotion, confidence) is saved.
- **Bias**: While pre-trained models are robust, they may still exhibit performance variations across different lighting conditions and demographics.

## Future Scope
- **Long-term Personalization**: Implementing more advanced collaborative filtering based on historical user preferences.
- **Multimodal Analysis**: Integrating voice tone analysis and text sentiment for a more holistic mood assessment.
- **Wearable Integration**: Syncing with heart rate and sleep data for better context awareness.

## Setup Instructions
1. **Backend**:
   - `cd backend`
   - `pip install -r requirements.txt`
   - `python main.py` (Runs on port 8000)
2. **Frontend**:
   - `npm install`
   - `npm run dev` (Runs on port 3000)
3. **Environment Variables**: Ensure `.env.local` contains valid Supabase credentials and the backend URL.
