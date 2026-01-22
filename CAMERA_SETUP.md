# Camera Access Setup Guide

## Quick Solution

The camera requires a secure context (HTTPS or localhost). I've set up a dedicated port for camera access.

### Option 1: Use the Camera-Friendly URL (Recommended)

**Open this URL in your browser:**
```
http://localhost:3001
```

This port is specifically configured for camera access and should work immediately.

### Option 2: Start the Camera-Friendly Server

If port 3001 is not running, start it with:

```bash
bun run dev:camera
```

Or with npm:
```bash
npm run dev:camera
```

This will start the frontend on port 3001, which is optimized for camera access.

## Why Port 3001?

- Port 3000 might have security restrictions in some browsers
- Port 3001 is explicitly configured as a secure localhost context
- The app automatically detects and enables camera access on this port

## Troubleshooting

### Camera Still Not Working?

1. **Check Browser Permissions:**
   - Make sure you've granted camera permissions to localhost
   - Go to browser settings → Site permissions → Camera
   - Ensure localhost is allowed

2. **Try Different Browser:**
   - Chrome/Edge usually work best
   - Firefox might need additional permissions
   - Safari on Mac requires explicit permission

3. **Check if Port 3001 is Running:**
   ```bash
   # Windows PowerShell
   netstat -ano | findstr :3001
   
   # Should show a listening process
   ```

4. **Manual Start:**
   ```bash
   # Stop any existing Next.js processes
   # Then run:
   bun run dev:camera
   ```

5. **Alternative: Use Image Upload**
   - If camera still doesn't work, use the "Upload Image" tab
   - This doesn't require camera permissions

## Current Setup

- **Main Frontend:** `http://localhost:3000` (general use)
- **Camera-Friendly Frontend:** `http://localhost:3001` (camera access)
- **Backend:** `http://localhost:8000` (API)

Both frontend ports connect to the same backend, so your data is synced.

## Browser Console

If you see errors, check the browser console (F12) for:
- Camera permission errors
- CORS errors
- Network errors

The app will show helpful error messages if camera access is blocked.


