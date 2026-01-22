# Testing Your Personal PyTorch Model Integration

## Localhost Links

### Frontend (Camera-Friendly)
```
http://localhost:3001
```
**Use this URL** - Camera access works best on this port.

### Backend API
```
http://localhost:8000
```
- Health Check: `http://localhost:8000/health`
- API Documentation: `http://localhost:8000/docs` (Interactive Swagger UI)

---

## How to Test Your Model Integration

### Step 1: Check Backend Logs

When the backend starts, look for these messages in the console:

**✅ Success Messages:**
```
[OK] Model found at: [path]
✅ Loaded personal PyTorch model from [path]
   Device: cpu, Image size: 224
```

**❌ If you see errors:**
- Check if PyTorch is installed: `pip install torch torchvision`
- Verify model file exists at: `saved_model/personal_face_recognition_model.pth`

### Step 2: Test via Frontend

1. **Open** `http://localhost:3001` in your browser
2. **Sign in/Sign up** (if needed)
3. **Capture an image** or **upload a photo**
4. **Check backend console** - You should see:
   ```
   DEBUG: Using personal PyTorch face recognition model
   DEBUG: Top 3 predictions (Personal PyTorch Model):
     happy: 0.856
     neutral: 0.120
     sad: 0.024
   ```

### Step 3: Test via API Directly

**Using PowerShell:**
```powershell
# Test health endpoint
Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing

# Test emotion detection (requires image file)
$imagePath = "path/to/your/test/image.jpg"
$form = @{
    file = Get-Item $imagePath
}
Invoke-RestMethod -Uri "http://localhost:8000/detect" -Method Post -Form $form
```

**Using curl (if available):**
```bash
# Health check
curl http://localhost:8000/health

# Emotion detection
curl -X POST http://localhost:8000/detect \
  -F "file=@path/to/image.jpg"
```

**Using Python:**
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Emotion detection
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/detect", files=files)
    print(response.json())
```

### Step 4: Verify Model is Being Used

**Check the response** - When you detect an emotion, the backend should:
1. Use your PyTorch model (check console logs)
2. Return emotion predictions with confidence scores
3. Provide recommendations based on detected emotion

**Expected Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.856,
  "mood": "positive state",
  "recommendations": [...],
  "disclaimer": "..."
}
```

---

## Troubleshooting

### Model Not Loading?

1. **Install PyTorch:**
   ```bash
   cd backend
   pip install torch torchvision pillow
   ```

2. **Check model file:**
   ```bash
   # Should exist at:
   saved_model/personal_face_recognition_model.pth
   ```

3. **Test model loading:**
   ```bash
   python backend/load_personal_model.py
   ```

### Backend Not Starting?

1. **Check for errors** in console
2. **Verify dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Check port 8000** is not in use:
   ```powershell
   netstat -ano | findstr :8000
   ```

### Frontend Not Connecting?

1. **Check backend URL** in frontend code
2. **Verify CORS** settings in backend
3. **Check browser console** for errors

---

## Quick Test Checklist

- [ ] Backend running on port 8000
- [ ] Frontend running on port 3001
- [ ] Backend logs show "Loaded personal PyTorch model"
- [ ] Can access http://localhost:3001
- [ ] Can capture/upload image
- [ ] Emotion detection works
- [ ] Backend console shows "Using personal PyTorch face recognition model"

---

## Expected Behavior

When your model is working correctly:

1. **Backend startup:** Shows model loading message
2. **Image detection:** Uses your PyTorch model (not FER or TensorFlow)
3. **Predictions:** Returns emotions with confidence scores
4. **Performance:** Should be fast (first inference may be slower)

If you see "Using FER library fallback" in logs, your PyTorch model didn't load. Check the troubleshooting section above.



