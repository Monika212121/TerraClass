# TerraClass — Offline Satellite Tile Classifier

A small, offline service that classifies satellite tiles by land-use type (Forest, River, Residential, Industrial, AnnualCrop, SeaLake, Highway) using
a locally-run MobileNetV3-Small model, stores results in SQLite, and exposes them for querying via a FastAPI endpoint.

Built for the GalaxEye Backend Engineer, ML Systems take-home assignment.



## Submission documents

- Design note (Part 1): `Part1_Design_Note.txt`
- Working slice (Part 2): this repository — see "Running the service" and "API" sections below
- Problem-solving answers (Part 3): `Part3_Problem_Solving.txt`


## 1. Architecture

```
TerraClass/
├── app/ # FastAPI service (routes, orchestration, DB access)
│ ├── main.py # App instance, lifespan startup (loads model + repository once)
│ ├── api/
│ │ └── endpoints.py # POST /predict, GET /predictions
│ ├── services/
│ │ └── classification_service.py # Orchestrates ml.predict() + persistence
│ ├── db/
│ │ └── database.py # SQLite schema + PredictionRepository (save/get_all)
│ └── core/
│ └── config.py # DB path, allowed content types
├── ml/ # Model definition, preprocessing, inference, training
│ ├── config.py # Class mapping, image size, normalization, model version
│ ├── model.py # MobileNetV3-Small architecture
│ ├── preprocess.py # Eval-time image transform
│ ├── predict.py # ImageClassifier: loads weights, runs inference
│ └── train.py # Training script (not imported by the service)
├── artifacts/ # Trained model weights + config (checked in — service runs offline out of the box)
├── data/ # candidate_tiles/, eval_set/, eval_labels.csv (see Dataset below)
├── storage/ # Runtime SQLite DB (created automatically, gitignored)
└── tests/
└── test_api.py # API tests: happy path, 3 failure modes, DB round-trip
```



## 2. Setup

```bash
python -m gxenv gxenv
gxenv\Scripts\activate        # Windows
pip install -r requirements.txt
```


## 3. Dataset
Place the dataset under `data/`:


## 4. Running the service

```bash
python -m uvicorn app.main:app --reload
```

On first run, if trained weights don't already exist under `artifacts/`, the service trains a fresh model automatically before starting up (a few minutes).
Trained weights are committed to this repo under `artifacts/`, so a normal run skips training and starts immediately — the service does not require internet access to run.

Once running:
- Swagger UI: http://127.0.0.1:8000/docs
- Health check: `GET /health`



## 5. API

### `POST /predict` (THIS IS THE MAIN ENDPOINT ASKED IN REQUIREMENT)

Upload a tile image, get back a classification, and have it stored.

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "tile=@data/eval_set/tile_001.png"
```

Response:
```json
{
  "prediction_id": 1,
  "tile_name": "tile_001.png",
  "tile_sha256": "a52a2df4...",
  "predicted_label": "Forest",
  "confidence": 0.9757,
  "model_version": "mobilenetv3-small-v1",
  "inference_time_ms": 16.57,
  "created_at": "2026-09-27T07:29:29.423496+00:00",
  "class_probs": { "Forest": 0.9757, "River": 0.01, "...": "..." }
}
```

### `GET /predictions`

Returns stored predictions, most recent first.

```bash
curl "http://127.0.0.1:8000/predictions"
curl "http://127.0.0.1:8000/predictions?limit=10"
```



## 6. Running tests

```bash
python -m pytest tests/ -v
```

5 tests: valid tile classification, rejection of non-image files, rejection of corrupt image bytes, rejection of empty uploads, and 
A write/read round-trip confirming a stored prediction is retrievable via `GET /predictions`.



## 7. What's implemented vs. deliberately stubbed

This is a thin slice per the assignment's instructions ("stub or skip the rest, and say so") — the following are explicit scope decisions, not oversights:

| Feature | Status | Reasoning |
|---|---|---|
| Single-tile classify + store | Implemented | Core requirement (Part 2) |
| Query stored results | Implemented (unfiltered list, `limit` param) | Simplest useful version of "querying"; see DESIGN.md |
| `needs_review` confidence flag | Analyzed, not wired into API | Threshold selection done via confidence-distribution sweep on the validation set (see DESIGN.md); deliberately left unimplemented in the API response as a natural next step |
| Filtering (`GET /predictions?label=&min_confidence=`) | Stubbed | Cheap follow-up; not required for the core path |
| Authentication | Stubbed | Not relevant to a single-analyst offline tool; would add API-key or mTLS before multi-user deployment |
| Batch/async classify endpoint | Stubbed | Brief specifies one endpoint, one tile; real throughput needs would push toward a queue-backed batch endpoint |
| Monitoring/alerting | Stubbed | Addressed in prose in `PART3.md` (Q2) |
| Model retraining pipeline | Stubbed | Auto-train-on-missing-artifacts covers the demo/first-run case; see DESIGN.md for the offline-deployment caveat |



## 8. Model

- **Architecture**: MobileNetV3-Small, ImageNet-pretrained backbone (frozen),
  fine-tuned classifier head only — chosen over a from-scratch CNN for
  better accuracy on a small training set, and over heavier backbones
  (ResNet18, EfficientNet) for CPU inference speed at scale. See
  `DESIGN.md` for the full comparison.

- **Input resolution**: 128×128 (upsampled from native 64×64 tiles) — chosen
  over the more common 224×224 since the source imagery carries no extra
  detail past 128px; keeps CPU inference cost lower for high-volume batch
  processing without a meaningful accuracy trade-off at this task's
  resolution.

- **Training data**: `candidate_tiles/` (150 images/class, 7 classes), stratified 80/20 train/val split.

- **Final held-out accuracy** (`eval_set/` + `eval_labels.csv`, never used
  during training or threshold selection): **[FILL IN — your printed final_acc]**

- **Confidence threshold** (for the `needs_review` flag, analyzed but not
  yet wired into the API — see table above): **[FILL IN — threshold value from your sweep table]**,
  which on the validation set gave **[FILL IN — % auto-accepted / accuracy-among-accepted at that threshold]**

- **Notable confusion pairs** (from validation confusion matrix): **[FILL IN — e.g. "River↔Highway, AnnualCrop↔Forest"]**




## 9. Inspecting the database

The SQLite file lives at `storage/terraclass.db`, created automatically on
first run. Three ways to look at its contents:

**Option A: Quick check via Python (no extra install needed):**
```bash
python -c "
import sqlite3
conn = sqlite3.connect('storage/terraclass.db')
conn.row_factory = sqlite3.Row
for row in conn.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 10'):
    print(dict(row))
"
```

**Option B: Via the `sqlite3` CLI**, if installed:
```bash
sqlite3 storage/terraclass.db
.headers on
.mode column
SELECT * FROM predictions ORDER BY id DESC LIMIT 10;
.quit
```

**Option C: Via a GUI** — [DB Browser for SQLite](https://sqlitebrowser.org/dl/)
(free): open the app → File → Open Database → select
`storage/terraclass.db` → "Browse Data" tab → `predictions` table.

Note: `storage/terraclass.db` is committed with a small number of real predictions from local testing, left in intentionally as evidence the
service was actually run end-to-end, not just written and never executed.


## 10. Screenshots & Evidence

- `docs/screenshots/` — Swagger UI responses for both endpoints, the `predictions` table viewed in DB Browser for SQLite, and the training log across all 8 epochs.

- `docs/swagger_output/prediction_list_from_swagger.json` — raw JSON response from `GET /predictions`, exported directly from Swagger UI.