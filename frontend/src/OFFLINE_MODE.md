# Offline Mode Explained

## What You're Seeing

When you see the **"Backend not available: TypeError: Failed to fetch"** message in the browser console, **this is completely normal and expected!** 

The Echoes frontend is designed to work in two modes:

## 🟢 Backend Connected Mode

When the FastAPI backend is running:
- Green "Backend Connected" indicator shows in top-right
- Real embeddings and semantic analysis from backend
- Can analyze any concept with available data
- Dynamic timeline generation from real data

## 🟡 Offline Mode (Current)

When the backend is **not** running (like now):
- Amber "Offline Mode" indicator shows in top-right
- Automatically falls back to rich demo data
- Pre-loaded concepts work perfectly:
  - ✅ **Freedom** - National independence → Digital autonomy
  - ✅ **Circle** - Cosmic perfection → UI design
  - ✅ **AI** - Automation → Creative partnership
- Any other concept shows generic fallback data

## This is NOT an Error!

The frontend is **working correctly**. The "Failed to fetch" message is just the app checking if the backend is available, then gracefully falling back to offline mode when it's not.

### Console Messages You'll See (Normal!)

```
Backend not available: TypeError: Failed to fetch
  → This is the health check failing (expected)

📡 Backend unavailable - using offline mode for "Freedom"
  → The app successfully loaded demo data instead
```

## How to Enable Backend Mode

If you want to connect to the real backend:

### 1. Start the Backend Server

```bash
# In your backend directory
cd echoes-backend

# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Mac/Linux

# Start the server
uvicorn api.main:app --reload --port 8000
```

### 2. Refresh the Frontend

Once the backend starts, refresh your browser. You should see:
- 🟢 **Green "Backend Connected"** indicator
- Real-time semantic analysis
- Dynamic embeddings

## Try It Now!

**In Offline Mode**, search for these concepts:

1. **"Freedom"** - See evolution from 1940s national independence to 2020s digital privacy
2. **"Circle"** - Watch transformation from ancient sun symbols to modern UI elements  
3. **"AI"** - Trace journey from 1936 Turing machines to 2020s ChatGPT

All work perfectly in offline mode! The visualizations, timeline slider, and force-directed graphs all function with the demo data.

## Why This Design?

This offline-first approach means:
- ✅ Frontend works immediately without setup
- ✅ Perfect for demos and presentations
- ✅ No need to set up backend for UI development
- ✅ Graceful degradation if backend goes down

## Summary

| Indicator | Status | What It Means |
|-----------|--------|---------------|
| 🟢 Backend Connected | Backend running | Real data from API |
| 🟡 Offline Mode | Backend not running | Demo data (works great!) |

**Current Status:** 🟡 Offline Mode - Everything is working as designed!

---

**Next Steps:**
- ✅ Try the demo concepts (Freedom, Circle, AI)
- 📖 See [QUICKSTART.md](./QUICKSTART.md) to start the backend
- 🔧 See [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) for API details
