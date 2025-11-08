# Echoes - Frontend

**Decoding the Evolution of Human Thought**

A visually stunning, interactive web application that visualizes the evolution of human ideas across time, from ancient civilizations to the digital age. Built for an 18-hour hackathon MVP.

## 🎨 Features

- **🔍 Concept Search** - Search any concept to see its evolution through history
- **🧠 Memetic Evolution** - Interactive force-directed graph showing semantic relationships over time
- **🖼️ Pattern Rebirth** - Side-by-side visual comparisons of ancient and modern representations
- **📊 AI Narrative** - AI-generated insights about semantic shifts and cultural transformations
- **⏱️ Timeline Control** - Synchronized timeline slider across all visualizations
- **🌐 Backend Integration** - Connects to FastAPI backend for real embeddings and data
- **💾 Offline Mode** - Falls back to mock data when backend is unavailable

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Backend API running (optional - app works in offline mode)

### Installation

1. **Clone and install dependencies:**

```bash
npm install
```

2. **Configure environment variables:**

```bash
cp .env.example .env
```

Edit `.env` and set your backend URL:

```env
VITE_API_URL=http://localhost:8000
```

3. **Start the development server:**

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## 🔧 Backend Integration

### Connecting to the Backend

The frontend automatically connects to the backend API specified in `VITE_API_URL`. 

**Backend endpoints used:**

- `GET /health` - Health check
- `POST /embed` - Get embeddings for text
- `GET /timeline?concept=...&top_n=...` - Get timeline data for a concept
- `GET /era?concept=...&era=...&top_n=...` - Get data for specific era
- `GET /symbol-pairs?symbol=...` - Get visual symbol pairs

### Backend Setup

1. **Navigate to your backend directory:**

```bash
cd ../echoes-backend
```

2. **Follow backend setup instructions:**

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Generate embeddings for demo concepts
python scripts/build_embeddings.py --concept freedom --eras 1900s,2020s

# Start the server
uvicorn api.main:app --reload --port 8000
```

3. **Verify connection:**

The frontend will show a "Backend Connected" indicator (green wifi icon) in the top-right when successfully connected.

### Offline Mode

If the backend is unavailable, the app automatically falls back to mock data and displays an "Offline Mode" indicator (amber wifi icon). You can still explore the UI with pre-loaded concepts: **Freedom**, **Circle**, and **AI**.

## 🎨 Tech Stack

- **React** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Motion** (Framer Motion) - Animations
- **Vite** - Build tool
- **shadcn/ui** - UI components
- **Lucide React** - Icons
- **Canvas API** - Force-directed graph rendering

## 📁 Project Structure

```
/
├── App.tsx                      # Main app component
├── components/
│   ├── SearchInterface.tsx      # Landing page with search
│   ├── VisualizationView.tsx    # Main visualization container
│   ├── MemeticEvolution.tsx     # Force-directed graph
│   ├── PatternRebirth.tsx       # Image comparison gallery
│   ├── NarrativeSummary.tsx     # AI insights display
│   └── ui/                      # shadcn/ui components
├── lib/
│   ├── api.ts                   # Backend API client
│   ├── dataTransform.ts         # Transform backend data to frontend format
│   └── mockData.ts              # Fallback mock data
├── hooks/
│   └── useConceptData.ts        # Data fetching hook
└── styles/
    └── globals.css              # Global styles and theme
```

## 🎯 Usage

1. **Search for a concept** - Enter any word or idea (e.g., "Freedom", "AI", "Democracy")
2. **Explore the evolution** - Use the timeline slider to see how the concept changed over time
3. **Examine relationships** - Hover over nodes in the Memetic Evolution graph
4. **Compare visuals** - View ancient and modern visual representations in Pattern Rebirth

### Pre-loaded Concepts

For the best experience with rich data:
- **Freedom** - From national independence to digital autonomy
- **Circle** - From cosmic perfection to UI design
- **AI** - From theoretical computing to creative partnership

## 🛠️ Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run linter

### Adding New Concepts

To add backend support for new concepts:

1. Add data files to backend: `data/1900s_<concept>.csv`, `data/2020s_<concept>.csv`
2. Generate embeddings: `python scripts/build_embeddings.py --concept <concept> --eras 1900s,2020s`
3. Add symbol pairs (optional): `assets/symbols/<concept>/`
4. Search for the concept in the frontend!

### Customizing the UI

- **Theme**: Edit `/styles/globals.css` for colors and design tokens
- **Components**: Modify individual components in `/components/`
- **Animations**: Adjust Motion animations in component files

## 🚢 Deployment

### Frontend Deployment

**Vercel (Recommended):**

```bash
npm run build
vercel --prod
```

**Netlify:**

```bash
npm run build
netlify deploy --prod --dir=dist
```

### Environment Variables

Set `VITE_API_URL` to your production backend URL in your deployment platform's environment settings.

## 📊 Data Flow

1. User searches for a concept
2. Frontend checks backend health
3. If backend available:
   - Fetch timeline data from `/timeline` endpoint
   - Fetch symbol pairs from `/symbol-pairs` endpoint
   - Transform backend data to frontend format
4. If backend unavailable:
   - Load mock data as fallback
   - Display offline mode indicator
5. Render visualizations with data

## 🎨 Design Philosophy

- **Dark Futuristic Aesthetic** - Deep slate backgrounds with cyan/blue neon accents
- **Data-Driven** - All visualizations based on real embeddings and similarity scores
- **Smooth Animations** - Motion-based transitions for a polished experience
- **Responsive** - Works on desktop, tablet, and mobile
- **Accessible** - Keyboard navigation and screen reader support

## 🤝 Contributing

This is a hackathon MVP! Feel free to:
- Add more visualization types
- Improve the force-directed layout algorithm
- Add more pre-loaded concepts
- Enhance the UI/UX
- Optimize performance

## 📝 License

MIT License - Built for the Echoes hackathon project

## 🔗 Related

- [Backend Repository](../echoes-backend/) - FastAPI backend with embeddings
- [Demo Video](#) - Watch the demo presentation

---

**Built with 💙 for the Echoes Hackathon** - Decoding how human ideas evolve through time
