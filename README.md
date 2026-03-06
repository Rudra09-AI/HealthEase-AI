# 🫁 PneumoScan AI — Complete Setup Guide
## Step-by-Step: Colab Training → PostgreSQL → FastAPI → Frontend

---

## 📁 PROJECT STRUCTURE
```
pneumoscan/
├── colab/
│   └── STEP1_Train_Model.ipynb     ← Upload to Google Colab
├── backend/
│   ├── main.py                     ← FastAPI server
│   ├── requirements.txt
│   └── models/                     ← Put trained model here
│       ├── pneumoscan_model.h5
│       └── class_names.json
├── frontend/
│   └── index.html                  ← Open in browser
└── pgadmin_setup/
    └── create_tables.sql           ← Run in pgAdmin
```

---

## ═══════════════════════════════════════
## PART A — GOOGLE COLAB (Model Training)
## ═══════════════════════════════════════

### A1. Open Google Colab
Go to: https://colab.research.google.com
Click: File → Upload Notebook → select STEP1_Train_Model.ipynb

### A2. Enable GPU
Runtime → Change runtime type → Hardware accelerator → GPU → Save

### A3. Upload Your 2GB Dataset
Option 1 (Recommended): Upload to Google Drive first
  - Go to drive.google.com
  - Upload your chest_xray.zip
  - In notebook Cell 1, set DATASET_PATH to your drive path

Option 2: Upload directly in Colab
  - Uncomment the files.upload() lines in Cell 1
  - Run and pick your .zip file (will take a few minutes)

### A4. Run All Cells in Order
Cell 1:  Mount Drive & locate dataset
Cell 2:  Set dataset path
Cell 3:  Install libraries
Cell 4:  Import & check GPU
Cell 5:  Data preprocessing + augmentation
Cell 6:  Visualize sample images
Cell 7:  Build DenseNet-121 model
Cell 8:  Train Phase 1 (10 epochs, ~20 min with GPU)
Cell 9:  Fine-tune Phase 2 (20 epochs, ~40 min)
Cell 10: Plot training curves
Cell 11: Evaluate on test set
Cell 12: Generate Grad-CAM visualizations
Cell 13: Save model files
Cell 14: Download files to your computer ← IMPORTANT

### A5. Files You Get After Training
- pneumoscan_model.h5     ← The trained AI model (~80MB)
- class_names.json        ← ["NORMAL", "PNEUMONIA"]
- model_info.json         ← accuracy, AUC scores
- training_curves.png     ← accuracy/loss graphs
- confusion_matrix.png    ← model performance
- gradcam_results.png     ← heatmap samples

---

## ═══════════════════════════════════════
## PART B — PGADMIN DATABASE SETUP
## ═══════════════════════════════════════

### B1. Install PostgreSQL
Download from: https://www.postgresql.org/download/
- Install with default settings
- Remember the password you set for 'postgres' user
- pgAdmin 4 is included automatically

### B2. Create Database
1. Open pgAdmin 4
2. Left panel: expand Servers → PostgreSQL
3. Right-click "Databases" → Create → Database
4. Name it: pneumoscan_db
5. Click Save

### B3. Run SQL Setup Script
1. Click on pneumoscan_db to select it
2. Click the SQL icon (query tool) at the top
3. Open file: pgadmin_setup/create_tables.sql
4. Press F5 or click the Run button (▶)
5. You should see: "Database setup complete! ✅"

### B4. Verify Tables Were Created
In the Query Tool, run:
  SELECT table_name FROM information_schema.tables WHERE table_schema='public';

You should see: patients, xray_scans, diagnosis_results, model_versions

---

## ═══════════════════════════════════════
## PART C — FASTAPI BACKEND
## ═══════════════════════════════════════

### C1. Install Python Dependencies
Open terminal/command prompt in the backend/ folder:

  pip install -r requirements.txt

### C2. Place Your Trained Model
Create a models/ folder inside backend/:
  backend/
    models/
      pneumoscan_model.h5     ← from Colab download
      class_names.json        ← from Colab download

### C3. Set Your Database Password
Open backend/main.py and find:
  "password": "YOUR_PGADMIN_PASSWORD"
Replace with the password you set in pgAdmin.

### C4. Start the Backend Server
In terminal, from the backend/ folder:
  uvicorn main:app --reload --port 8000

You should see:
  INFO:     Uvicorn running on http://127.0.0.1:8000
  Loading ML model...
  ✅ Model loaded! Classes: ['NORMAL', 'PNEUMONIA']

### C5. Test the API
Open browser: http://localhost:8000
You should see: {"message": "PneumoScan AI API is running ✅"}

Test endpoints:
  http://localhost:8000/health    ← model status
  http://localhost:8000/stats     ← database stats
  http://localhost:8000/records   ← all patient records
  http://localhost:8000/docs      ← Swagger auto-docs (BONUS!)

---

## ═══════════════════════════════════════
## PART D — FRONTEND (Connected UI)
## ═══════════════════════════════════════

### D1. Open the Frontend
Simply open frontend/index.html in any browser
(Chrome / Firefox / Edge)

### D2. Connect to Backend
The top of the page has a URL bar showing: http://localhost:8000
Click "Test Connection" — it should turn GREEN if backend is running.

### D3. Run a Full Analysis
1. Fill in patient details (name, age, gender, symptoms)
2. Click "Drop X-Ray Image Here" or drag your image
3. Click "Analyze with AI"
4. Watch the 5-step pipeline animate
5. See results: diagnosis + confidence bars + heatmap
6. Check "Records" tab — data is now in PostgreSQL!

---

## ═══════════════════════════════════════
## DATA FLOW EXPLAINED
## ═══════════════════════════════════════

  YOU (browser)
      │
      │ Upload X-Ray + Patient Info
      ▼
  FastAPI Backend (main.py)
      │
      ├─→ Save image to disk (uploads/ folder)
      │
      ├─→ Load DenseNet-121 model
      │       │
      │       ├─ Preprocess image (resize to 224×224, normalize)
      │       ├─ Run forward pass → get confidence scores
      │       └─ Generate Grad-CAM heatmap
      │
      ├─→ Save to PostgreSQL via psycopg2:
      │       ├─ INSERT into patients table
      │       ├─ INSERT into xray_scans table
      │       └─ INSERT into diagnosis_results table
      │
      └─→ Return JSON response to browser
              │
              ▼
          Frontend renders:
          - Disease confidence bars
          - Grad-CAM heatmap overlay
          - Clinical text report
          - Record saved confirmation

---

## TROUBLESHOOTING

Problem: "Cannot connect to backend"
Fix: Make sure uvicorn is running. Run: uvicorn main:app --reload

Problem: "Model file not found"
Fix: Place pneumoscan_model.h5 in backend/models/ folder

Problem: "Database connection failed"
Fix: Check DB_CONFIG password in main.py matches pgAdmin password

Problem: Colab disconnects during training
Fix: Use Google Colab Pro, or save checkpoints to Drive every 5 epochs

---

## MODEL PERFORMANCE (Expected after training)
Accuracy:  ~94%
AUC Score: ~97%
Precision: ~92%
Recall:    ~96%
