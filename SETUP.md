# Setup Guide for EmotiX

## Quick Start

### 1. Backend Setup

The backend is already configured and should be running on `http://localhost:8000`.

To start manually:
```bash
cd backend
python main.py
```

### 2. Frontend Setup

The frontend should be running on `http://localhost:3000` using Bun.

To start manually:
```bash
bun dev
```

### 3. Environment Variables (IMPORTANT)

Create a `.env.local` file in the root directory with your Supabase credentials:

```env
# Supabase Configuration
# Get these from your Supabase project: https://app.supabase.com/project/_/settings/api
NEXT_PUBLIC_SUPABASE_URL=your_supabase_project_url_here
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key_here

# Backend API URL (default is correct for localhost)
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

**How to get Supabase credentials:**
1. Go to https://app.supabase.com
2. Create a new project or select an existing one
3. Go to Settings → API
4. Copy the "Project URL" → `NEXT_PUBLIC_SUPABASE_URL`
5. Copy the "anon public" key → `NEXT_PUBLIC_SUPABASE_ANON_KEY`

### 4. Database Setup

Make sure your Supabase database has these tables:
- `mood_entries`
- `recommendations`
- `chat_messages`
- `wellness_challenges`

Check the `backfill.sql` file for table schemas if needed.

## Troubleshooting

### Backend not responding
- Check if Python dependencies are installed: `pip install -r backend/requirements.txt`
- Verify backend is running: Visit `http://localhost:8000/health`
- Check for port conflicts (another app using port 8000)

### Frontend not connecting to backend
- Verify `NEXT_PUBLIC_BACKEND_URL` in `.env.local` is `http://localhost:8000`
- Check browser console for CORS errors
- Make sure backend CORS is enabled (it should be by default)

### Enso AI not responding
- The chat endpoint has been improved with better response logic
- Check browser console for errors
- Verify backend is running and accessible
- Test the endpoint directly: `POST http://localhost:8000/chat` with body `{"message": "hello", "current_mood": "happy"}`

### Supabase connection errors
- Verify `.env.local` exists and has correct credentials
- Check Supabase project is active and not paused
- Ensure database tables are created
- Check browser console for specific error messages

## Testing

1. **Backend Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test Chat Endpoint:**
   ```bash
   curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\":\"hello\",\"current_mood\":\"happy\"}"
   ```

3. **Frontend:**
   - Open http://localhost:3000
   - Sign up/Sign in
   - Try capturing an emotion
   - Test Enso Buddy chat

## Current Status

✅ Backend: Running on port 8000
✅ Frontend: Running on port 3000 (Bun)
✅ Chat AI: Improved with better response logic
⚠️ Supabase: Requires `.env.local` configuration


