# Enso AI - Emotion Wellness System

An AI-powered emotional wellness application that detects facial emotions using deep learning and provides personalized wellness recommendations, mindfulness tools, and an AI chat companion.

## Features

### Core Functionality
- **Real-time Emotion Detection** - Capture emotions via webcam or upload images using a custom-trained ResNet/EfficientNet model
- **Mood Journal** - Log thoughts with AI-powered sentiment analysis
- **Personalized Recommendations** - Get curated music, movies, activities, and quotes based on detected emotions
- **Enso Buddy** - AI chat companion that provides context-aware emotional support
- **Wellness Dashboard** - Track mood history, streaks, challenges, and view analytics
- **Mindfulness Hub** - Guided breathing exercises, grounding techniques, and daily affirmations

### Emotion Categories
Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral

## Tech Stack

### Frontend
- **Next.js 15** (App Router, React 19)
- **TypeScript**
- **Tailwind CSS**
- **Radix UI** (Component primitives)
- **Recharts** (Data visualization)
- **Framer Motion** (Animations)
- **react-webcam** (Camera capture)

### Backend
- **FastAPI** (Python)
- **PyTorch** (Deep learning - ResNet emotion model)
- **OpenCV** (Face detection)
- **TextBlob** (Sentiment analysis)

### Database & Auth
- **Supabase** (PostgreSQL database + Auth)

## Project Structure

```
/
├── src/
│   ├── app/
│   │   ├── page.tsx          # Home - Emotion capture
│   │   ├── auth/             # Authentication
│   │   ├── buddy/            # AI Chat companion
│   │   ├── dashboard/        # Analytics & history
│   │   ├── mindfulness/      # Breathing exercises
│   │   └── profile/          # User profile
│   ├── components/           # Reusable UI components
│   ├── hooks/                # Custom React hooks
│   └── lib/                  # Utilities & Supabase client
├── backend/
│   ├── main.py               # FastAPI server
│   ├── utils.py              # Emotion detection logic
│   ├── recommendations.py    # Recommendation engine
│   └── models/               # TensorFlow/Keras models
├── saved_model/              # PyTorch models
└── public/                   # Static assets
```

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.9+
- Supabase account

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/harshitanagpal05/Enso-AI.git
   cd Enso-AI
   ```

2. **Install frontend dependencies**
   ```bash
   npm install
   # or
   bun install
   ```

3. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   # or create a virtual environment first
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install fastapi uvicorn opencv-python torch torchvision textblob fer pillow
   ```

4. **Set up environment variables**
   
   Create a `.env.local` file in the root directory:
   ```env
   NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
   NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
   ```

5. **Set up Supabase database**
   
   Create the following tables in your Supabase project:
   - `mood_entries` - Stores emotion detection results
   - `recommendations` - Stores personalized recommendations
   - `chat_messages` - Stores Enso Buddy conversations
   - `wellness_challenges` - Tracks user challenges

### Running the Application

1. **Start the backend server**
   ```bash
   cd backend
   python main.py
   # or
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the frontend development server**
   ```bash
   npm run dev
   # or
   bun dev
   ```

3. **For camera access** (if not on localhost):
   ```bash
   npm run dev:camera
   # Runs on port 3001 for secure camera access
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Database Schema

### mood_entries
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| user_id | uuid | User reference |
| emotion | text | Detected emotion |
| mood | text | Mapped mood state |
| confidence | float | Detection confidence |
| note | text | Journal entry |
| sentiment_score | float | Sentiment analysis score |
| sentiment_label | text | Positive/Negative/Neutral |
| context_data | jsonb | Additional metadata |
| created_at | timestamp | Entry timestamp |

### recommendations
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| entry_id | uuid | Mood entry reference |
| category | text | music/movie/activity/quote |
| content | text | Recommendation content |
| reason | text | Why recommended |
| link | text | External link |
| feedback_score | int | User feedback |

### chat_messages
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| user_id | uuid | User reference |
| role | text | user/assistant |
| content | text | Message content |
| created_at | timestamp | Message timestamp |

### wellness_challenges
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| user_id | uuid | User reference |
| title | text | Challenge name |
| description | text | Challenge details |
| category | text | mood/consistency/mindfulness |
| target_count | int | Goal count |
| current_count | int | Current progress |
| status | text | in_progress/completed |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect` | POST | Detect emotion from uploaded image |
| `/manual-update` | POST | Update emotion manually |
| `/chat` | POST | Chat with Enso Buddy |
| `/analyze-text` | POST | Analyze text sentiment |
| `/health` | GET | Health check |

## Custom Model Integration

The system supports custom PyTorch emotion models:

1. Place your `.pth` model file in `saved_model/personal_face_recognition_model.pth`
2. The system auto-detects ResNet-34/ResNet-50 architectures
3. Falls back to FER library if no custom model is available

## License

MIT License

## Acknowledgments

- FER library for fallback emotion detection
- Supabase for backend infrastructure
- Radix UI for accessible components
