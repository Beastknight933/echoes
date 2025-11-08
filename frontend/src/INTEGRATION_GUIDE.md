# Backend Integration Guide

Complete guide for integrating the Echoes frontend with the FastAPI backend.

## 🔗 Connection Overview

The frontend connects to the backend through a REST API client (`/lib/api.ts`) that handles all HTTP requests. The app gracefully falls back to mock data if the backend is unavailable.

## 📡 API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Purpose:** Check if backend is running

**Response:**
```json
{
  "status": "ok"
}
```

**Frontend Usage:**
```typescript
import { api } from './lib/api';

const health = await api.health();
// { status: "ok" }
```

---

### 2. Get Embeddings

**Endpoint:** `POST /embed`

**Purpose:** Get sentence embeddings for text

**Request Body:**
```json
{
  "text": "freedom and liberty"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

**Frontend Usage:**
```typescript
const result = await api.embed("freedom and liberty");
// { embedding: [0.123, -0.456, ...] }
```

---

### 3. Get Timeline Data

**Endpoint:** `GET /timeline?concept=freedom&top_n=10`

**Purpose:** Get concept evolution across eras

**Query Parameters:**
- `concept` (required) - The concept to analyze (e.g., "freedom")
- `top_n` (optional) - Number of top items per era (default: 5)

**Response:**
```json
{
  "concept": "freedom",
  "eras": [
    {
      "era": "1900s",
      "items": [
        {
          "text": "liberty and independence",
          "similarity": 0.89,
          "metadata": {}
        }
      ],
      "centroid": [0.1, 0.2, ...]
    },
    {
      "era": "2020s",
      "items": [
        {
          "text": "digital privacy and autonomy",
          "similarity": 0.92,
          "metadata": {}
        }
      ],
      "centroid": [0.3, 0.4, ...]
    }
  ],
  "semantic_shift": 35.5,
  "primary_association": {
    "from": "national independence",
    "to": "digital autonomy"
  }
}
```

**Frontend Usage:**
```typescript
const timeline = await api.getTimeline("freedom", 10);
// Returns timeline data with eras, items, and semantic shift
```

**Data Transformation:**

The frontend transforms this to:
1. **Graph Nodes** - Each item becomes a node connected to the central concept
2. **Narrative Summary** - Uses `semantic_shift` and `primary_association`
3. **Timeline Years** - Maps eras to display years (e.g., "1900s" → 1940)

---

### 4. Get Era-Specific Data

**Endpoint:** `GET /era?concept=freedom&era=1900s&top_n=20`

**Purpose:** Get detailed data for a specific era

**Query Parameters:**
- `concept` (required) - The concept
- `era` (required) - The era (e.g., "1900s", "2020s")
- `top_n` (optional) - Number of items (default: 10)

**Response:**
```json
{
  "concept": "freedom",
  "era": "1900s",
  "items": [
    {
      "text": "liberty and independence",
      "similarity": 0.89,
      "metadata": {}
    }
  ]
}
```

**Frontend Usage:**
```typescript
const eraData = await api.getEraData("freedom", "1900s", 20);
// Returns items for specific era
```

---

### 5. Get Symbol Pairs

**Endpoint:** `GET /symbol-pairs?symbol=freedom`

**Purpose:** Get ancient/modern visual symbol pairs

**Query Parameters:**
- `symbol` (required) - The symbol/concept name

**Response:**
```json
{
  "symbol": "freedom",
  "pairs": {
    "1900s": {
      "ancient": {
        "path": "/assets/symbols/freedom/1900s_ancient.jpg",
        "title": "Liberty Bell",
        "description": "Symbol of American independence",
        "era": "1753 CE"
      },
      "modern": {
        "path": "/assets/symbols/freedom/1900s_modern.jpg",
        "title": "Victory Sign",
        "description": "WWII-era peace symbol",
        "era": "1940s"
      }
    },
    "2020s": {
      "ancient": {
        "path": "/assets/symbols/freedom/2020s_ancient.jpg",
        "title": "Open Padlock",
        "description": "Traditional symbol of liberation",
        "era": "Ancient Symbol"
      },
      "modern": {
        "path": "/assets/symbols/freedom/2020s_modern.jpg",
        "title": "Encryption Key",
        "description": "Digital freedom through privacy",
        "era": "2020s"
      }
    }
  }
}
```

**Frontend Usage:**
```typescript
const symbols = await api.getSymbolPairs("freedom");
// Returns image paths and metadata for visual comparisons
```

---

## 🔄 Data Transformation Flow

### Backend Response → Frontend Visualization

```typescript
// 1. Fetch data from backend
const timelineData = await api.getTimeline(concept);
const symbolData = await api.getSymbolPairs(concept);

// 2. Transform to frontend format
import { transformTimelineToConceptData } from './lib/dataTransform';

const conceptData = transformTimelineToConceptData(
  concept,
  timelineData,
  symbolData
);

// 3. Use in components
<MemeticEvolution data={conceptData} />
<PatternRebirth data={conceptData} />
<NarrativeSummary data={conceptData} />
```

### Era String Conversion

Backend uses strings like `"1900s"`, `"2020s"`. Frontend converts to years:

```typescript
"1900s" → 1940  // Mid-point of decade
"2020s" → 2020  // Start of decade
"1950s" → 1950
```

---

## 🛠️ Backend Setup for Frontend Integration

### Required Backend Structure

```
echoes-backend/
├── data/
│   ├── 1900s_freedom.csv      # Historical data
│   └── 2020s_freedom.csv      # Modern data
├── embeddings/
│   └── freedom/
│       ├── 1900s.json         # Precomputed embeddings
│       └── 2020s.json
├── assets/
│   └── symbols/
│       └── freedom/           # Visual assets
│           ├── 1900s_ancient.jpg
│           ├── 1900s_modern.jpg
│           ├── 2020s_ancient.jpg
│           └── 2020s_modern.jpg
└── api/
    └── main.py                # FastAPI endpoints
```

### Generate Embeddings for New Concepts

```bash
# In backend directory
python scripts/build_embeddings.py --concept democracy --eras 1900s,1950s,2020s
```

This creates:
- `embeddings/democracy/1900s.json`
- `embeddings/democracy/1950s.json`
- `embeddings/democracy/2020s.json`

### Start Backend Server

```bash
cd echoes-backend
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
uvicorn api.main:app --reload --port 8000
```

Backend will be available at `http://localhost:8000`

---

## 🎯 Frontend Configuration

### Environment Variables

Create `.env` file:

```env
VITE_API_URL=http://localhost:8000
```

For production:

```env
VITE_API_URL=https://your-backend-api.com
```

### CORS Configuration

Ensure your backend allows requests from frontend origin:

```python
# api/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Development
        "https://your-frontend.vercel.app"  # Production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🔍 Testing the Integration

### 1. Test Backend Health

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

### 2. Test Timeline Endpoint

```bash
curl "http://localhost:8000/timeline?concept=freedom&top_n=5"
```

Expected: JSON with eras, items, and semantic_shift

### 3. Test Frontend Connection

1. Start backend: `uvicorn api.main:app --reload --port 8000`
2. Start frontend: `npm run dev`
3. Open `http://localhost:5173`
4. Look for green "Backend Connected" indicator
5. Search for "freedom"
6. Should see data from backend

---

## 🐛 Troubleshooting

### Backend Not Connected

**Symptoms:**
- Amber "Offline Mode" indicator
- Using mock data instead of backend data

**Solutions:**

1. **Check backend is running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check CORS settings:**
   - Ensure frontend origin is allowed in backend CORS config

3. **Check API URL:**
   - Verify `VITE_API_URL` in `.env` is correct
   - Restart frontend after changing `.env`

4. **Check network:**
   - Open browser DevTools → Network tab
   - Look for failed requests to backend
   - Check error messages

### Empty or Missing Data

**Symptoms:**
- Backend connected but no data showing
- Error: "Timeline fetch failed"

**Solutions:**

1. **Generate embeddings:**
   ```bash
   python scripts/build_embeddings.py --concept freedom --eras 1900s,2020s
   ```

2. **Check data files exist:**
   ```bash
   ls embeddings/freedom/
   # Should show: 1900s.json, 2020s.json
   ```

3. **Test endpoint directly:**
   ```bash
   curl "http://localhost:8000/timeline?concept=freedom&top_n=5"
   ```

### CORS Errors

**Symptoms:**
- Browser console: "CORS policy: No 'Access-Control-Allow-Origin' header"

**Solutions:**

Add CORS middleware to backend:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 Data Requirements

### Minimum Data for Concept

To add a new concept, you need:

1. **Data files:** `data/1900s_concept.csv`, `data/2020s_concept.csv`
2. **CSV format:**
   ```csv
   text,year,source
   "related text about concept",1920,"source name"
   ```

3. **Generate embeddings:** Run `build_embeddings.py`

4. **(Optional) Symbol pairs:** Add images to `assets/symbols/concept/`

### Example: Adding "Democracy"

```bash
# 1. Add data files
echo "text,year,source" > data/1900s_democracy.csv
echo "representative government,1920,historical" >> data/1900s_democracy.csv

echo "text,year,source" > data/2020s_democracy.csv
echo "digital voting systems,2022,news" >> data/2020s_democracy.csv

# 2. Generate embeddings
python scripts/build_embeddings.py --concept democracy --eras 1900s,2020s

# 3. Test in frontend
# Search for "democracy" - should work!
```

---

## 🚀 Production Deployment

### Backend Deployment

Deploy to Railway, Render, or AWS:

```bash
# Example: Railway
railway up
```

Note your backend URL: `https://echoes-backend.railway.app`

### Frontend Deployment

1. **Update environment variable:**
   ```env
   VITE_API_URL=https://echoes-backend.railway.app
   ```

2. **Build and deploy:**
   ```bash
   npm run build
   vercel --prod
   ```

3. **Update backend CORS:**
   ```python
   allow_origins=["https://echoes-frontend.vercel.app"]
   ```

---

## 📝 API Client Customization

### Custom Base URL

```typescript
import EchoesAPI from './lib/api';

const api = new EchoesAPI('https://custom-backend.com');
const data = await api.getTimeline('freedom');
```

### Error Handling

```typescript
try {
  const data = await api.getTimeline('freedom');
} catch (error) {
  console.error('API error:', error);
  // Fallback to mock data
  const mockData = getMockData('freedom');
}
```

### Request Timeout

Modify `/lib/api.ts`:

```typescript
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 5000);

const response = await fetch(url, {
  signal: controller.signal
});
```

---

## 🎉 Success Checklist

- [ ] Backend running at `http://localhost:8000`
- [ ] Frontend running at `http://localhost:5173`
- [ ] Green "Backend Connected" indicator visible
- [ ] Can search for "freedom" and see data
- [ ] Timeline slider works and updates visualizations
- [ ] Force-directed graph animates smoothly
- [ ] Images load in Pattern Rebirth panel
- [ ] No CORS errors in browser console

---

**Ready to build!** Your frontend is now fully integrated with the backend. 🚀
