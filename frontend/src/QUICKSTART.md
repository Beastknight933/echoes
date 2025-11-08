# 🚀 Echoes - Quick Start Guide

Get up and running with Echoes in 5 minutes!

## Option 1: Frontend Only (Offline Mode)

Perfect for UI testing and development without backend.

```bash
# 1. Install dependencies
npm install

# 2. Start dev server
npm run dev
```

✅ **That's it!** Visit `http://localhost:5173`

The app will run in offline mode with pre-loaded mock data for:
- Freedom
- Circle  
- AI

---

## Option 2: Full Stack (Frontend + Backend)

For the complete experience with real embeddings and AI analysis.

### Terminal 1: Backend

```bash
# Navigate to backend directory
cd echoes-backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Generate embeddings for demo concept
python scripts/build_embeddings.py --concept freedom --eras 1900s,2020s

# Start backend server
uvicorn api.main:app --reload --port 8000
```

Backend running at: `http://localhost:8000` ✅

### Terminal 2: Frontend

```bash
# Navigate to frontend directory
cd echoes-frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Start frontend dev server
npm run dev
```

Frontend running at: `http://localhost:5173` ✅

---

## ✅ Verify Installation

### Check Backend Health

```bash
curl http://localhost:8000/health
```

Expected response: `{"status":"ok"}`

### Check Frontend

1. Open `http://localhost:5173` in browser
2. Look for **green "Backend Connected"** indicator in top-right
3. Search for **"freedom"** 
4. You should see data visualization with timeline and graphs!

---

## 🎯 Demo Flow

Try this demo sequence to showcase all features:

1. **Search "Freedom"**
   - See the AI narrative about 35% semantic shift
   - Notice the "national independence" → "digital autonomy" evolution

2. **Use Timeline Slider**
   - Drag slider between 1940, 1980, and 2020
   - Watch the graph nodes morph and reposition
   - Observe how related concepts change over time

3. **Explore Memetic Evolution Graph**
   - Hover over nodes to see them highlight
   - Notice the connections (links) between concepts
   - Central node is the main concept, connected nodes are related terms

4. **Compare Visual Patterns**
   - Look at the Pattern Rebirth panel
   - See ancient symbols vs modern representations
   - Timeline slider syncs both visualizations

5. **Try Other Concepts**
   - Search "AI" - see computation → creative partnership evolution
   - Search "Circle" - see cosmic symbol → UI design evolution
   - Search any word - see generic fallback data

---

## 🐛 Troubleshooting

### Backend Not Starting

**Error: "ModuleNotFoundError: No module named 'fastapi'"**

```bash
# Make sure you activated the virtual environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Mac/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

**Error: "torch" installation fails**

```bash
# Install PyTorch CPU version explicitly
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### Frontend Shows "Offline Mode"

**Amber indicator instead of green:**

1. Check backend is running: `curl http://localhost:8000/health`
2. Check `.env` file has correct URL: `VITE_API_URL=http://localhost:8000`
3. Restart frontend after changing `.env`
4. Check browser console for CORS errors

---

### No Data Showing

**"Timeline fetch failed" error:**

```bash
# Generate embeddings
cd echoes-backend
python scripts/build_embeddings.py --concept freedom --eras 1900s,2020s

# Verify files created
ls embeddings/freedom/
# Should show: 1900s.json, 2020s.json
```

---

## 📦 Project Structure

```
echoes/
├── echoes-backend/          # FastAPI backend
│   ├── api/                 # API endpoints
│   ├── data/                # CSV data files
│   ├── embeddings/          # Precomputed embeddings (generated)
│   ├── scripts/             # Build scripts
│   └── requirements.txt
│
└── echoes-frontend/         # React frontend (YOU ARE HERE)
    ├── components/          # UI components
    ├── lib/                 # API client & utilities
    ├── hooks/               # React hooks
    └── styles/              # CSS styles
```

---

## 🎨 Adding New Concepts

Want to add your own concepts? Here's how:

### Step 1: Create Data Files (Backend)

```bash
cd echoes-backend

# Create data file for historical era
cat > data/1900s_democracy.csv << EOF
text,year,source
"representative government and voting rights",1920,"historical records"
"parliamentary systems and constitutions",1915,"academic"
EOF

# Create data file for modern era  
cat > data/2020s_democracy.csv << EOF
text,year,source
"digital voting and blockchain governance",2022,"tech news"
"social media and political discourse",2021,"research"
EOF
```

### Step 2: Generate Embeddings

```bash
python scripts/build_embeddings.py --concept democracy --eras 1900s,2020s
```

### Step 3: Test in Frontend

Search for "democracy" - it should now work with real backend data!

---

## 🚀 Next Steps

- **Customize UI**: Edit components in `/components/`
- **Add more eras**: Include 1950s, 1980s, etc.
- **Deploy**: See [README.md](./README.md) for deployment guides
- **Explore API**: Read [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)

---

## 💡 Pro Tips

### Hot Reloading

Both servers support hot reloading:
- Backend: Edit Python files, server reloads automatically
- Frontend: Edit React files, browser updates instantly

### Browser DevTools

- **Network Tab**: See API requests to backend
- **Console Tab**: Check for errors or warnings
- **React DevTools**: Inspect component state

### Performance

- Force-directed graph uses Canvas for smooth 60fps animation
- Images lazy-load on demand
- API responses cached by React hooks

---

## 🆘 Need Help?

1. Check [README.md](./README.md) for detailed documentation
2. Read [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) for API details
3. Look at browser console for error messages
4. Verify backend logs for API errors

---

## 🎉 You're Ready!

Your Echoes installation is complete. Start exploring how human ideas evolve through time! 🧠✨

**Happy Hacking!** 🚀
