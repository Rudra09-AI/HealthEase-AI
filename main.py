"""
PneumoScan AI — FastAPI Backend
================================
Run: uvicorn main:app --reload --port 8000

Receives X-ray images → runs ML model → stores in PostgreSQL → returns results
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2, os, json, datetime, uuid, shutil
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
import tensorflow as tf

# ── APP SETUP ─────────────────────────────────────────────────────────────────
app = FastAPI(title="PneumoScan AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production: replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
UPLOAD_DIR   = BASE_DIR / "uploads"
HEATMAP_DIR  = BASE_DIR / "heatmaps"
MODEL_PATH   = BASE_DIR / "models" / "pneumoscan_model.h5"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"

UPLOAD_DIR.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(exist_ok=True)

app.mount("/uploads",  StaticFiles(directory=UPLOAD_DIR),  name="uploads")
app.mount("/heatmaps", StaticFiles(directory=HEATMAP_DIR), name="heatmaps")

# ── DATABASE CONNECTION ────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "pneumoscan_db",
    "user":     "postgres",
    "password": "YOUR_PGADMIN_PASSWORD"   # ← Change this!
}

def get_db():
    """Get a database connection"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

# ── LOAD ML MODEL ─────────────────────────────────────────────────────────────
model = None
CLASS_NAMES = []

def load_model():
    global model, CLASS_NAMES
    if MODEL_PATH.exists():
        print("Loading ML model...")
        model = tf.keras.models.load_model(str(MODEL_PATH))
        with open(CLASSES_PATH) as f:
            CLASS_NAMES = json.load(f)
        print(f"✅ Model loaded! Classes: {CLASS_NAMES}")
    else:
        print("⚠️  Model file not found. Using dummy predictions.")
        CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

load_model()

# ── GRAD-CAM FUNCTION ─────────────────────────────────────────────────────────
def generate_gradcam(img_array, model, layer_name="conv5_block16_concat"):
    """Generate Grad-CAM heatmap and save it"""
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            pred_idx = tf.argmax(preds[0])
            class_score = preds[:, pred_idx]

        grads = tape.gradient(class_score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return np.zeros((7, 7))

def save_heatmap_overlay(original_img, heatmap, filename):
    """Overlay heatmap on original image and save"""
    img_resized = cv2.resize(original_img, (224, 224))
    hm_resized  = cv2.resize(heatmap, (224, 224))
    hm_colored  = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
    overlay     = cv2.addWeighted(img_resized, 0.6, hm_colored, 0.4, 0)
    path        = HEATMAP_DIR / filename
    cv2.imwrite(str(path), overlay)
    return str(path)

# ── PREDICTION FUNCTION ───────────────────────────────────────────────────────
def run_prediction(image_path: str):
    """Run the ML model on an image and return all disease confidences"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)

    if model is None:
        # Dummy prediction if model not loaded
        confidences = {name: float(np.random.uniform(5, 30)) for name in CLASS_NAMES}
        top_class = CLASS_NAMES[0]
        confidences[top_class] = float(np.random.uniform(70, 95))
        return confidences, top_class, max(confidences.values()), None

    preds = model.predict(img_array, verbose=0)[0]
    heatmap = generate_gradcam(img_array, model)

    # Map predictions to disease names
    confidences = {CLASS_NAMES[i]: float(preds[i] * 100) for i in range(len(CLASS_NAMES))}
    top_class = CLASS_NAMES[np.argmax(preds)]
    top_conf  = float(np.max(preds) * 100)

    # Save heatmap
    heatmap_filename = f"heatmap_{uuid.uuid4().hex[:8]}.jpg"
    heatmap_path = save_heatmap_overlay(img, heatmap, heatmap_filename)

    return confidences, top_class, top_conf, heatmap_path

def save_to_database(patient_data, scan_data, result_data):
    """Save everything to PostgreSQL"""
    conn = get_db()
    cur  = conn.cursor()
    try:
        # 1. Insert patient
        cur.execute("""
            INSERT INTO patients (name, age, gender, symptoms)
            VALUES (%(name)s, %(age)s, %(gender)s, %(symptoms)s)
            RETURNING id
        """, patient_data)
        patient_id = cur.fetchone()["id"]

        # 2. Insert scan
        cur.execute("""
            INSERT INTO xray_scans (patient_id, image_filename, image_path, image_size_kb)
            VALUES (%(patient_id)s, %(image_filename)s, %(image_path)s, %(image_size_kb)s)
            RETURNING id
        """, {**scan_data, "patient_id": patient_id})
        scan_id = cur.fetchone()["id"]

        # 3. Insert diagnosis result
        cur.execute("""
            INSERT INTO diagnosis_results (
                scan_id, pneumonia_conf, covid_conf, tuberculosis_conf,
                pleural_effusion_conf, cardiomegaly_conf, atelectasis_conf,
                normal_conf, final_diagnosis, confidence_score,
                report_text, heatmap_path
            ) VALUES (
                %(scan_id)s, %(pneumonia_conf)s, %(covid_conf)s, %(tuberculosis_conf)s,
                %(pleural_effusion_conf)s, %(cardiomegaly_conf)s, %(atelectasis_conf)s,
                %(normal_conf)s, %(final_diagnosis)s, %(confidence_score)s,
                %(report_text)s, %(heatmap_path)s
            ) RETURNING id
        """, {**result_data, "scan_id": scan_id})
        result_id = cur.fetchone()["id"]

        conn.commit()
        return patient_id, scan_id, result_id

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "PneumoScan AI API is running ✅", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "classes": CLASS_NAMES}

# ── MAIN ENDPOINT: Analyze X-Ray ──────────────────────────────────────────────
@app.post("/analyze")
async def analyze_xray(
    file:     UploadFile = File(...),
    name:     str = Form(default="Anonymous"),
    age:      int = Form(default=0),
    gender:   str = Form(default="Unknown"),
    symptoms: str = Form(default="")
):
    # 1. Save uploaded file
    ext       = Path(file.filename).suffix or ".jpg"
    unique_fn = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / unique_fn

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size_kb = os.path.getsize(save_path) / 1024

    # 2. Run ML model
    confidences, top_class, top_conf, heatmap_path = run_prediction(str(save_path))

    # 3. Build report text
    report = build_report(name, age, gender, symptoms, top_class, top_conf, confidences)

    # 4. Save to PostgreSQL
    patient_id, scan_id, result_id = save_to_database(
        patient_data={"name": name, "age": age, "gender": gender, "symptoms": symptoms},
        scan_data={
            "image_filename": file.filename,
            "image_path": str(save_path),
            "image_size_kb": file_size_kb
        },
        result_data={
            "pneumonia_conf":       confidences.get("PNEUMONIA", 0),
            "covid_conf":           confidences.get("COVID19", 0),
            "tuberculosis_conf":    confidences.get("TUBERCULOSIS", 0),
            "pleural_effusion_conf":confidences.get("PLEURAL_EFFUSION", 0),
            "cardiomegaly_conf":    confidences.get("CARDIOMEGALY", 0),
            "atelectasis_conf":     confidences.get("ATELECTASIS", 0),
            "normal_conf":          confidences.get("NORMAL", 0),
            "final_diagnosis":      top_class,
            "confidence_score":     top_conf,
            "report_text":          report,
            "heatmap_path":         str(heatmap_path) if heatmap_path else None
        }
    )

    return {
        "success":        True,
        "patient_id":     patient_id,
        "scan_id":        scan_id,
        "result_id":      result_id,
        "diagnosis":      top_class,
        "confidence":     round(top_conf, 2),
        "confidences":    {k: round(v, 2) for k, v in confidences.items()},
        "report":         report,
        "image_url":      f"/uploads/{unique_fn}",
        "heatmap_url":    f"/heatmaps/{Path(heatmap_path).name}" if heatmap_path else None,
        "analyzed_at":    datetime.datetime.now().isoformat()
    }

# ── GET ALL RECORDS ────────────────────────────────────────────────────────────
@app.get("/records")
def get_all_records(limit: int = 50):
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT p.id, p.name, p.age, p.gender, p.symptoms,
               d.final_diagnosis, d.confidence_score, d.analyzed_at,
               s.image_filename
        FROM patients p
        JOIN xray_scans s ON s.patient_id = p.id
        JOIN diagnosis_results d ON d.scan_id = s.id
        ORDER BY d.analyzed_at DESC LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return {"records": [dict(r) for r in rows]}

# ── GET STATS FOR DASHBOARD ────────────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    conn = get_db()
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) AS total FROM patients")
    total = cur.fetchone()["total"]

    cur.execute("""
        SELECT final_diagnosis, COUNT(*) AS count
        FROM diagnosis_results GROUP BY final_diagnosis ORDER BY count DESC
    """)
    by_disease = [dict(r) for r in cur.fetchall()]

    cur.execute("SELECT ROUND(AVG(confidence_score)::numeric, 2) AS avg_conf FROM diagnosis_results")
    avg_conf = cur.fetchone()["avg_conf"]

    cur.close(); conn.close()
    return {"total_patients": total, "by_disease": by_disease, "avg_confidence": avg_conf}

# ── GET SINGLE PATIENT RECORD ──────────────────────────────────────────────────
@app.get("/record/{patient_id}")
def get_record(patient_id: int):
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT * FROM v_patient_diagnoses WHERE patient_id = %s
    """, (patient_id,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Patient not found")
    return dict(row)

# ── HELPER ────────────────────────────────────────────────────────────────────
def build_report(name, age, gender, symptoms, diagnosis, conf, confidences):
    date = datetime.datetime.now().strftime("%d %b %Y %H:%M")
    findings = {
        "PNEUMONIA":  "Increased opacity in lower lobe. Consolidation pattern present.",
        "COVID19":    "Bilateral peripheral ground-glass opacities. Lower lobe predominance.",
        "TUBERCULOSIS": "Apical shadowing with possible cavitary lesion.",
        "NORMAL":     "Both lung fields clear. No acute cardiopulmonary process identified.",
    }
    finding = findings.get(diagnosis, "Findings noted — clinical correlation recommended.")
    return f"""PneumoScan AI Report | {date}
Patient: {name}, {gender}, Age {age}
Symptoms: {symptoms or 'Not provided'}
---
FINDINGS: {finding}
IMPRESSION: {diagnosis} detected with {conf:.1f}% confidence.
---
⚠ AI-generated report. Consult a radiologist for confirmation."""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
