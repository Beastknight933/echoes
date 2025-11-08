# Echoes - Backend (v1.0.0)

Minimal backend for the Echoes hackathon project (18 hr MVP).
Provides endpoints to serve precomputed sentence embeddings per concept-era and simple similarity/timeline results.

## Repo structure
echoes-backend/
├─ data/
│ ├─ 1900s_freedom.csv
│ └─ 2020s_freedom.csv
├─ embeddings/
│ └─ freedom/
│ ├─ 1900s.json
│ └─ 2020s.json
├─ assets/
│ └─ symbols/
├─ api/
│ ├─ main.py
│ ├─ utils.py
│ └─ models.py
├─ scripts/
│ └─ build_embeddings.py
├─ demo_query.py
├─ requirements.txt
└─ README.md


## Quickstart (local)
1. Create & activate a virtualenv:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   pip install --upgrade pip setuptools wheel

   pip install -r requirements.txt

   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu



2. Generate embeddings for demo concept freedom:

python scripts/build_embeddings.py --concept freedom --eras 1900s,2020s

> This will create embeddings/freedom/1900s.json and embeddings/freedom/2020s.json.




3. Start the server:

uvicorn api.main:app --reload --port 8000


4. Run the demo query:

python demo_query.py



Endpoints

GET /health — health check

POST /embed — body param text (string): returns embedding

GET /timeline?concept=...&top_n=... — returns era-by-era top similar items and centroid shift

GET /era?concept=...&era=1900s&top_n=... — returns top items for era

GET /symbol-pairs?symbol=... — returns static asset pairs under assets/symbols/<symbol>/


Notes & tips

For speed during demo, precompute and commit the embeddings/ JSON files.

If the model download is slow, run build_embeddings.py on a machine with an internet connection first.

Extend utils to compute TF-IDF labels, sentiment, or to add caching (Redis / disk cache).