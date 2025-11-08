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

## try this only if the requirement file is giving error
1. Upgrade pip / setuptools / wheel / build (this is the most important step):



# inside .venv
python -m pip install --upgrade pip setuptools wheel build

2. (Optional but helpful) Clear pip cache so pip re-downloads wheels:



python -m pip cache purge

3. Install NumPy first using binary-only preference so pip will not attempt to build from source:



# prefer binary, but allow fallback; if it still tries to build, switch to only-binary
pip install numpy==1.25.2 --prefer-binary

If the above still tries to build (error persists), force only-binary (this will fail if no wheel exists for your Python):

pip install numpy==1.25.2 --only-binary=:all:

If --only-binary fails (no wheel for your Python), try a different NumPy wheel that is more likely to exist on PyPI for modern Pythons:

pip install numpy==1.26.4 --prefer-binary

4. Install PyTorch (CPU wheel) using the PyTorch index:



# Install a CPU wheel that exists on the PyTorch index. Pick the one that was shown as available earlier:
pip install --trusted-host download.pytorch.org --index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu

If you want a different available torch (the index you showed had many +cpu versions), pick one from that list and replace 2.3.1+cpu.

5. Install everything else from the requirements file (final requirements.txt given below):



pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

6. Quick verification:



python -c "import numpy, torch, sklearn, fastapi, sentence_transformers; print('numpy', numpy.__version__, 'torch', getattr(torch,'__version__','n/a'), 'sklearn', __import__('sklearn').__version__)"


##



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