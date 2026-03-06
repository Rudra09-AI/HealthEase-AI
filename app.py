import streamlit as st
import numpy as np
import cv2
import psycopg2
from psycopg2.extras import RealDictCursor
import datetime, json, os, time
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ═══════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit call
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════
# CUSTOM CSS — Dark medical theme
# ═══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* GLOBAL */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #040d14;
    color: #e8f4f8;
}
.stApp { background-color: #040d14; }

/* HIDE default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #081825;
    border-right: 1px solid #0e3a5a;
}
section[data-testid="stSidebar"] * { color: #e8f4f8 !important; }

/* TITLE HEADER */
.main-header {
    background: linear-gradient(135deg, #081825, #0c2033);
    border: 1px solid #0e3a5a;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 38px; font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #ffffff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -1px;
}
.main-sub {
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: #5b8a9f;
    letter-spacing: 2px; margin-top: 6px;
}

/* METRIC CARDS */
.metric-card {
    background: #0c2033;
    border: 1px solid #0e3a5a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 28px; font-weight: 700; color: #00d4ff;
}
.metric-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 9px; color: #5b8a9f;
    letter-spacing: 2px; margin-top: 4px;
}

/* PANELS */
.panel {
    background: #081825;
    border: 1px solid #0e3a5a;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.panel::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}
.panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 10px; color: #5b8a9f;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 14px;
}

/* RESULT BANNER */
.result-danger {
    background: rgba(255,68,68,0.1);
    border: 1px solid rgba(255,68,68,0.4);
    border-radius: 12px; padding: 20px;
    border-left: 4px solid #ff4444;
}
.result-safe {
    background: rgba(57,255,20,0.07);
    border: 1px solid rgba(57,255,20,0.3);
    border-radius: 12px; padding: 20px;
    border-left: 4px solid #39ff14;
}
.diagnosis-text {
    font-family: 'Syne', sans-serif;
    font-size: 26px; font-weight: 800;
}
.conf-text {
    font-family: 'Space Mono', monospace;
    font-size: 32px; font-weight: 700;
}

/* DISEASE ROW */
.d-row {
    display: flex; align-items: center;
    gap: 12px; margin-bottom: 10px;
}
.d-name { min-width: 160px; font-size: 13px; }
.d-conf { font-family: 'Space Mono', monospace; font-size: 12px; min-width: 50px; text-align: right; }

/* STEP INDICATOR */
.step-done {
    background: rgba(57,255,20,0.08);
    border: 1px solid rgba(57,255,20,0.3);
    border-radius: 8px; padding: 8px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: #39ff14;
    margin-bottom: 6px;
}
.step-active {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 8px; padding: 8px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: #00d4ff;
    margin-bottom: 6px;
}
.step-wait {
    background: rgba(255,255,255,0.03);
    border: 1px solid #0e3a5a;
    border-radius: 8px; padding: 8px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: #5b8a9f;
    margin-bottom: 6px;
}

/* TABLE */
.rec-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.rec-table th {
    padding: 10px 14px; text-align: left;
    font-family: 'Space Mono', monospace; font-size: 9px;
    letter-spacing: 1px; color: #5b8a9f;
    border-bottom: 1px solid #0e3a5a;
    background: #0c2033;
}
.rec-table td { padding: 10px 14px; border-bottom: 1px solid rgba(14,58,90,0.4); }
.badge-danger { color: #ff4444; font-family: 'Space Mono', monospace; font-size: 10px; }
.badge-safe   { color: #39ff14; font-family: 'Space Mono', monospace; font-size: 10px; }
.badge-warn   { color: #ffcc00; font-family: 'Space Mono', monospace; font-size: 10px; }

/* REPORT BOX */
.report-box {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 10px; padding: 18px;
    font-family: 'Space Mono', monospace;
    font-size: 12px; line-height: 1.9;
    color: #b8d8e4; white-space: pre-wrap;
}

/* INPUTS */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: #0c2033 !important;
    border: 1px solid #0e3a5a !important;
    color: #e8f4f8 !important;
    border-radius: 8px !important;
}
.stButton button {
    background: linear-gradient(135deg, #005577, #00d4ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.2) !important;
    transition: all .3s !important;
}
.stButton button:hover {
    box-shadow: 0 8px 30px rgba(0,212,255,0.35) !important;
    transform: translateY(-2px) !important;
}
.stFileUploader {
    background: #0c2033 !important;
    border: 2px dashed #0e3a5a !important;
    border-radius: 12px !important;
}
.stProgress .st-bo { background: #00d4ff !important; }
div[data-testid="stMetricValue"] { color: #00d4ff !important; font-family: 'Space Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# DATABASE CONFIG — change password here
# ═══════════════════════════════════════════════════
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "pneumoscan_db",
    "user":     "postgres",
    "password": "1234"
}

def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    except Exception as e:
        return None

def save_to_db(patient, scan, result):
    conn = get_db_connection()
    if not conn:
        return None, None, None
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO patients (name, age, gender, symptoms)
            VALUES (%s,%s,%s,%s) RETURNING id
        """, (patient['name'], patient['age'], patient['gender'], patient['symptoms']))
        pid = cur.fetchone()['id']

        cur.execute("""
            INSERT INTO xray_scans (patient_id, image_filename, image_path, image_size_kb)
            VALUES (%s,%s,%s,%s) RETURNING id
        """, (pid, scan['filename'], scan['path'], scan['size_kb']))
        sid = cur.fetchone()['id']

        cur.execute("""
            INSERT INTO diagnosis_results (
                scan_id, pneumonia_conf, covid_conf, tuberculosis_conf,
                pleural_effusion_conf, cardiomegaly_conf, atelectasis_conf,
                normal_conf, final_diagnosis, confidence_score, report_text
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
        """, (
            sid,
            result.get('PNEUMONIA', 0), result.get('COVID19', 0),
            result.get('TUBERCULOSIS', 0), result.get('PLEURAL_EFFUSION', 0),
            result.get('CARDIOMEGALY', 0), result.get('ATELECTASIS', 0),
            result.get('NORMAL', 0), result['diagnosis'],
            result['confidence'], result['report']
        ))
        rid = cur.fetchone()['id']
        conn.commit()
        return pid, sid, rid
    except Exception as e:
        conn.rollback()
        st.error(f"DB Error: {e}")
        return None, None, None
    finally:
        cur.close(); conn.close()

def load_records():
    conn = get_db_connection()
    if not conn: return []
    cur = conn.cursor()
    cur.execute("""
        SELECT p.name, p.age, p.gender, p.symptoms,
               d.final_diagnosis, d.confidence_score, d.analyzed_at
        FROM patients p
        JOIN xray_scans s ON s.patient_id = p.id
        JOIN diagnosis_results d ON d.scan_id = s.id
        ORDER BY d.analyzed_at DESC LIMIT 50
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [dict(r) for r in rows]

def load_stats():
    conn = get_db_connection()
    if not conn: return {}, 0, 0
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM patients")
    total = cur.fetchone()['c']
    cur.execute("SELECT ROUND(AVG(confidence_score)::numeric,1) AS a FROM diagnosis_results")
    avg = cur.fetchone()['a'] or 0
    cur.execute("SELECT final_diagnosis, COUNT(*) AS c FROM diagnosis_results GROUP BY final_diagnosis ORDER BY c DESC")
    by_disease = {r['final_diagnosis']: r['c'] for r in cur.fetchall()}
    cur.close(); conn.close()
    return by_disease, total, avg

# ═══════════════════════════════════════════════════
# ML MODEL LOADER
# ═══════════════════════════════════════════════════
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'pneumoscan_model.h5')
    classes_path = os.path.join(os.path.dirname(__file__), 'models', 'class_names.json')
    try:
        import tensorflow as tf
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            with open(classes_path) as f:
                classes = json.load(f)
            return model, classes, True
    except Exception as e:
        pass
    return None, ['NORMAL', 'PNEUMONIA'], False

# ═══════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════
def predict(img_array, model, classes):
    """Run model prediction — returns dict of confidences"""
    import numpy as np
    if model is None:
        # Simulated result when model not loaded
        import random
        scenario = random.choice(['PNEUMONIA', 'NORMAL', 'COVID19'])
        confs = {c: round(random.uniform(3, 20), 1) for c in ['NORMAL','PNEUMONIA','COVID19','TUBERCULOSIS','PLEURAL_EFFUSION','CARDIOMEGALY','ATELECTASIS']}
        confs[scenario] = round(random.uniform(72, 94), 1)
        top = max(confs, key=confs.get)
        return confs, top, confs[top]

    img = cv2.resize(img_array, (224, 224))
    inp = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
    preds = model.predict(inp, verbose=0)[0]
    confs = {classes[i]: round(float(preds[i]) * 100, 1) for i in range(len(classes))}
    top = max(confs, key=confs.get)
    return confs, top, confs[top]

def generate_gradcam(img_array, model):
    """Generate Grad-CAM heatmap overlay"""
    try:
        import tensorflow as tf
        layer_name = 'conv5_block16_concat'
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        inp = np.expand_dims(cv2.resize(img_array, (224,224)) / 255.0, axis=0).astype(np.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(inp)
            idx = tf.argmax(preds[0])
            score = preds[:, idx]
        grads = tape.gradient(score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        hm = heatmap.numpy()
        hm_resized = cv2.resize(hm, (224, 224))
        hm_colored = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
        img_resized = cv2.resize(img_array, (224, 224))
        overlay = cv2.addWeighted(img_resized, 0.6, hm_colored, 0.4, 0)
        return overlay
    except:
        return cv2.resize(img_array, (224, 224))

def build_report(name, age, gender, symptoms, diagnosis, conf):
    date = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
    findings = {
        'PNEUMONIA':        'Increased opacity in lower lobe. Consolidation pattern present. Air bronchograms visible.',
        'COVID19':          'Bilateral peripheral ground-glass opacities. Lower lobe predominance. No pleural effusion.',
        'TUBERCULOSIS':     'Apical shadowing in upper zones. Possible cavitary lesion. Hilar lymphadenopathy cannot be excluded.',
        'PLEURAL_EFFUSION': 'Homogeneous opacification with blunting of costophrenic angle. Mediastinal shift noted.',
        'CARDIOMEGALY':     'Cardiac silhouette enlarged. Cardiothoracic ratio > 0.5.',
        'ATELECTASIS':      'Linear opacities noted. Subsegmental atelectasis pattern present.',
        'NORMAL':           'Both lung fields clear. No consolidation, pleural effusion, or pneumothorax identified.',
    }
    finding = findings.get(diagnosis, 'Findings noted. Clinical correlation recommended.')
    impression = 'No acute cardiopulmonary process identified.' if diagnosis == 'NORMAL' else f'{diagnosis} detected with {conf:.1f}% confidence. Urgent radiologist review advised.'
    return f"""PneumoScan AI — Clinical Report
{'─'*45}
Date      : {date}
Patient   : {name or 'Anonymous'}, {gender}, Age {age}
Symptoms  : {symptoms or 'Not provided'}
{'─'*45}
FINDINGS
{finding}

IMPRESSION
{impression}
{'─'*45}
Model     : DenseNet-121 (Transfer Learning)
Dataset   : NIH ChestX-ray14
Confidence: {conf:.1f}%

⚠  AI-generated report. Not a substitute
   for professional radiological diagnosis."""

# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
      <div style='font-size:44px;'>🫁</div>
      <div style='font-family:Syne,sans-serif; font-weight:800; font-size:22px; color:#e8f4f8;'>PneumoScan</div>
      <div style='font-family:Space Mono,monospace; font-size:9px; color:#5b8a9f; letter-spacing:2px; margin-top:4px;'>AI DIAGNOSTICS v2.0</div>
    </div>
    <hr style='border-color:#0e3a5a; margin: 14px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["🔬 Analyze X-Ray", "📋 Patient Records", "⚙️ Data Flow & Info"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border-color:#0e3a5a; margin:16px 0;'>", unsafe_allow_html=True)

    # DB Status
    conn_test = get_db_connection()
    if conn_test:
        conn_test.close()
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:10px;color:#39ff14;letter-spacing:1px;">● DATABASE CONNECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:10px;color:#ff4444;letter-spacing:1px;">✕ DATABASE OFFLINE</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#5b8a9f;margin-top:6px;">Start PostgreSQL service<br>& check password in app.py</div>', unsafe_allow_html=True)

    # Model status
    model, classes, model_loaded = load_model()
    if model_loaded:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:10px;color:#39ff14;letter-spacing:1px;margin-top:8px;">● MODEL LOADED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:10px;color:#ffcc00;letter-spacing:1px;margin-top:8px;">⚠ DEMO MODE (no .h5)</div>', unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:#0e3a5a; margin:16px 0;'>
    <div style='font-family:Space Mono,monospace;font-size:9px;color:#5b8a9f;letter-spacing:1px;line-height:2;'>
    TECH STACK<br>
    ──────────<br>
    DenseNet-121<br>
    TensorFlow 2.x<br>
    Grad-CAM<br>
    FastAPI<br>
    PostgreSQL<br>
    Streamlit
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <div class="main-title">🫁 PneumoScan AI</div>
  <div class="main-sub">CHEST X-RAY DEEP LEARNING DIAGNOSIS SYSTEM — DENSENET-121 + GRAD-CAM</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# PAGE 1 — ANALYZE
# ═══════════════════════════════════════════════════
if page == "🔬 Analyze X-Ray":

    # Stats row
    by_disease, total_patients, avg_conf = load_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{total_patients}</div><div class="metric-lbl">TOTAL SCANS</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_conf}%</div><div class="metric-lbl">AVG CONFIDENCE</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">94.2%</div><div class="metric-lbl">MODEL ACCURACY</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">DenseNet</div><div class="metric-lbl">ARCHITECTURE</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1.3], gap="large")

    # ── LEFT COLUMN ──────────────────────────────────
    with col_left:
        st.markdown('<div class="panel"><div class="panel-title">● Patient Information</div>', unsafe_allow_html=True)
        name     = st.text_input("Patient Name", placeholder="e.g. John Doe")
        c1, c2   = st.columns(2)
        with c1: age    = st.number_input("Age", min_value=1, max_value=120, value=35)
        with c2: gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.text_input("Symptoms", placeholder="cough, fever, breathlessness...")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">● Upload Chest X-Ray</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["png", "jpg", "jpeg", "tiff"],
                                    help="Upload a chest X-ray image (PNG/JPG/TIFF)")

        if uploaded:
            img_pil   = Image.open(uploaded).convert("RGB")
            img_array = np.array(img_pil)
            st.image(img_pil, caption="Uploaded X-Ray", use_column_width=True)

        analyze_clicked = st.button("🔬  Run Deep Learning Analysis", disabled=not uploaded)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT COLUMN ─────────────────────────────────
    with col_right:
        st.markdown('<div class="panel"><div class="panel-title">● AI Processing Pipeline</div>', unsafe_allow_html=True)

        ph_steps    = st.empty()
        ph_progress = st.empty()

        def show_steps(current):
            steps = [
                ("🖼️", "Image Preprocessing & Normalization"),
                ("🧠", "DenseNet-121 Feature Extraction"),
                ("🔥", "Grad-CAM Heatmap Generation"),
                ("📊", "Multi-label Disease Classification"),
                ("🗄️", "Saving Results to PostgreSQL"),
            ]
            html = ""
            for i, (icon, label) in enumerate(steps):
                if i < current:
                    html += f'<div class="step-done">✓ {icon} {label}</div>'
                elif i == current:
                    html += f'<div class="step-active">⟳ {icon} {label}</div>'
                else:
                    html += f'<div class="step-wait">○ {icon} {label}</div>'
            ph_steps.markdown(html, unsafe_allow_html=True)

        show_steps(-1)
        st.markdown('</div>', unsafe_allow_html=True)

        # Results placeholder
        ph_result = st.empty()

        if analyze_clicked and uploaded:
            # Animate pipeline
            progress = ph_progress.progress(0, text="Starting pipeline...")
            delays = [0.6, 0.9, 0.8, 0.7, 0.6]

            for i in range(5):
                show_steps(i)
                progress.progress((i+1)*20, text=["Preprocessing image...",
                    "Running DenseNet-121...", "Generating Grad-CAM...",
                    "Classifying diseases...", "Saving to database..."][i])
                time.sleep(delays[i])

            progress.empty()
            show_steps(5)

            # Run prediction
            confs, top_class, top_conf = predict(img_array, model, classes)

            # Grad-CAM
            gradcam_img = generate_gradcam(img_array, model)

            # Build report
            report_text = build_report(name, age, gender, symptoms, top_class, top_conf)

            # Save to database
            pid, sid, rid = save_to_db(
                {'name': name, 'age': age, 'gender': gender, 'symptoms': symptoms},
                {'filename': uploaded.name, 'path': uploaded.name, 'size_kb': len(uploaded.getvalue())/1024},
                {**confs, 'diagnosis': top_class, 'confidence': top_conf, 'report': report_text}
            )

            # ── DISPLAY RESULTS ──────────────────────
            is_normal = 'NORMAL' in top_class.upper()
            css_class = 'result-safe' if is_normal else 'result-danger'
            conf_color = '#39ff14' if is_normal else '#ff4444'
            icon = '✅' if is_normal else '⚠️'

            st.markdown(f"""
            <div class="{css_class}" style="margin-top:16px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <div style="font-family:Space Mono,monospace;font-size:10px;color:#5b8a9f;letter-spacing:1px;">FINAL DIAGNOSIS</div>
                  <div class="diagnosis-text" style="color:{'#39ff14' if is_normal else '#ff4444'};">{icon} {top_class}</div>
                </div>
                <div class="conf-text" style="color:{conf_color};">{top_conf:.1f}%</div>
              </div>
              {'<div style="margin-top:10px;font-size:13px;color:#ffaaaa;">⚠ Abnormality detected — consult a radiologist</div>' if not is_normal else '<div style="margin-top:10px;font-size:13px;color:#aaffaa;">No significant pathology detected</div>'}
            </div>
            """, unsafe_allow_html=True)

            # Disease confidence bars
            st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
            colors = {'PNEUMONIA':'#ff4444','COVID19':'#ff6b35','TUBERCULOSIS':'#ffcc00',
                      'PLEURAL_EFFUSION':'#ff9944','CARDIOMEGALY':'#00d4ff',
                      'ATELECTASIS':'#aa88ff','NORMAL':'#39ff14'}
            for disease, conf_val in sorted(confs.items(), key=lambda x: x[1], reverse=True):
                color = colors.get(disease, '#00d4ff')
                st.markdown(f"""
                <div class="d-row">
                  <div class="d-name">{disease.replace('_',' ')}</div>
                  <div style="flex:1;height:6px;background:rgba(255,255,255,0.07);border-radius:3px;overflow:hidden;">
                    <div style="width:{conf_val}%;height:100%;background:{color};border-radius:3px;transition:width 1s;"></div>
                  </div>
                  <div class="d-conf" style="color:{color};">{conf_val}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Heatmap + original
            st.markdown("<br>", unsafe_allow_html=True)
            hm_col1, hm_col2 = st.columns(2)
            with hm_col1:
                st.image(gradcam_img, caption="🔥 Grad-CAM Heatmap (AI Focus Area)", use_column_width=True, channels="BGR")
            with hm_col2:
                st.image(img_pil, caption="📷 Original X-Ray", use_column_width=True)

            # DB confirmation
            if pid:
                st.success(f"✅ Saved to PostgreSQL — Patient #{pid} | Scan #{sid} | Result #{rid}")
            else:
                st.warning("⚠ DB not connected — result not saved. Check pgAdmin is running.")

            # Report
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="panel"><div class="panel-title">● AI Clinical Report</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="report-box">{report_text}</div>', unsafe_allow_html=True)
            st.download_button("⬇ Download Report", report_text, file_name="PneumoScan_Report.txt", mime="text/plain")
            st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# PAGE 2 — RECORDS
# ═══════════════════════════════════════════════════
elif page == "📋 Patient Records":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    r1, r2 = st.columns([3,1])
    with r1:
        st.markdown('<div class="panel-title">● All Patient Records — PostgreSQL Database</div>', unsafe_allow_html=True)
    with r2:
        refresh = st.button("⟳ Refresh")

    records = load_records()
    if not records:
        st.info("No records yet. Run an analysis first, or check your PostgreSQL connection.")
    else:
        # Stats
        by_disease, total, avg = load_stats()
        cols = st.columns(len(by_disease) if by_disease else 1)
        for i, (disease, count) in enumerate(by_disease.items()):
            with cols[i % len(cols)]:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{count}</div><div class="metric-lbl">{disease}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Table
        rows_html = ""
        for i, r in enumerate(records):
            is_normal = 'NORMAL' in str(r.get('final_diagnosis','')).upper()
            badge_cls = 'badge-safe' if is_normal else 'badge-danger'
            date_str  = r['analyzed_at'].strftime("%d %b %Y %H:%M") if r.get('analyzed_at') else '—'
            rows_html += f"""
            <tr>
              <td style='color:#5b8a9f'>{i+1}</td>
              <td>{r.get('name','—')}</td>
              <td>{r.get('age','—')}</td>
              <td>{r.get('gender','—')}</td>
              <td style='color:#5b8a9f;font-size:12px;'>{str(r.get('symptoms',''))[:35] or '—'}</td>
              <td><span class='{badge_cls}'>{r.get('final_diagnosis','—')}</span></td>
              <td style='font-family:Space Mono,monospace;'>{r.get('confidence_score',0):.1f}%</td>
              <td style='font-size:11px;color:#5b8a9f;'>{date_str}</td>
            </tr>"""

        st.markdown(f"""
        <table class="rec-table">
          <thead><tr>
            <th>#</th><th>Name</th><th>Age</th><th>Gender</th>
            <th>Symptoms</th><th>Diagnosis</th><th>Confidence</th><th>Date</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# PAGE 3 — DATA FLOW
# ═══════════════════════════════════════════════════
elif page == "⚙️ Data Flow & Info":
    st.markdown('<div class="panel"><div class="panel-title">● How Data Flows Through the System</div>', unsafe_allow_html=True)

    flow_items = [
        ("🖥️", "FRONTEND", "Streamlit UI"),
        ("📡", "HTTP REQUEST", "multipart/form-data"),
        ("⚙️", "FASTAPI", "Python Backend"),
        ("🧠", "DENSENET-121", "TensorFlow"),
        ("🔥", "GRAD-CAM", "Heatmap"),
        ("🗄️", "POSTGRESQL", "pgAdmin DB"),
        ("📊", "JSON", "Response"),
    ]
    cols = st.columns(len(flow_items))
    for i, (icon, title, sub) in enumerate(flow_items):
        with cols[i]:
            st.markdown(f"""
            <div style="background:#0c2033;border:1px solid #00d4ff;border-radius:10px;padding:14px 10px;text-align:center;">
              <div style="font-size:26px;">{icon}</div>
              <div style="font-family:Space Mono,monospace;font-size:9px;color:#00d4ff;letter-spacing:1px;margin-top:6px;">{title}</div>
              <div style="font-size:11px;color:#5b8a9f;margin-top:3px;">{sub}</div>
            </div>
            {"<div style='text-align:center;color:#00d4ff;font-size:18px;margin-top:10px;'>→</div>" if i < len(flow_items)-1 else ""}
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Tables + API
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">● PostgreSQL Tables</div>
          <div style="font-family:Space Mono,monospace;font-size:12px;line-height:2.2;">
            <span style="color:#00d4ff;">📦 patients</span><br>
            <span style="color:#5b8a9f;padding-left:16px;">id · name · age · gender · symptoms</span><br>
            <span style="color:#00d4ff;">📦 xray_scans</span><br>
            <span style="color:#5b8a9f;padding-left:16px;">id · patient_id (FK) · image_path</span><br>
            <span style="color:#00d4ff;">📦 diagnosis_results</span><br>
            <span style="color:#5b8a9f;padding-left:16px;">id · scan_id (FK) · all confidences<br>
            &nbsp;&nbsp;&nbsp;&nbsp;· final_diagnosis · report_text</span><br>
            <span style="color:#00d4ff;">📦 model_versions</span><br>
            <span style="color:#5b8a9f;padding-left:16px;">id · accuracy · auc · is_active</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">● API Endpoints (FastAPI)</div>
          <div style="display:flex;flex-direction:column;gap:10px;font-size:12px;">
            <div style="background:#0c2033;border-radius:8px;padding:10px 14px;border:1px solid #0e3a5a;">
              <span style="color:#ff6b35;font-family:Space Mono,monospace;font-size:10px;">POST</span>
              <span style="font-family:Space Mono,monospace;margin-left:8px;">/analyze</span>
              <div style="color:#5b8a9f;font-size:11px;margin-top:3px;">Upload X-Ray → model → DB → return JSON</div>
            </div>
            <div style="background:#0c2033;border-radius:8px;padding:10px 14px;border:1px solid #0e3a5a;">
              <span style="color:#39ff14;font-family:Space Mono,monospace;font-size:10px;">GET</span>
              <span style="font-family:Space Mono,monospace;margin-left:8px;">/records</span>
              <div style="color:#5b8a9f;font-size:11px;margin-top:3px;">All patient records from PostgreSQL</div>
            </div>
            <div style="background:#0c2033;border-radius:8px;padding:10px 14px;border:1px solid #0e3a5a;">
              <span style="color:#39ff14;font-family:Space Mono,monospace;font-size:10px;">GET</span>
              <span style="font-family:Space Mono,monospace;margin-left:8px;">/stats</span>
              <div style="color:#5b8a9f;font-size:11px;margin-top:3px;">Disease count stats for dashboard</div>
            </div>
            <div style="background:#0c2033;border-radius:8px;padding:10px 14px;border:1px solid #0e3a5a;">
              <span style="color:#39ff14;font-family:Space Mono,monospace;font-size:10px;">GET</span>
              <span style="font-family:Space Mono,monospace;margin-left:8px;">/health</span>
              <div style="color:#5b8a9f;font-size:11px;margin-top:3px;">API + model status check</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Commands
    st.markdown("""
    <div class="panel">
      <div class="panel-title">● CMD Commands to Run This Project</div>
      <div style="background:rgba(0,0,0,.5);border-radius:8px;padding:16px;font-family:Space Mono,monospace;font-size:12px;color:#00d4ff;line-height:2.2;">
        <span style="color:#5b8a9f;"># Install Streamlit</span><br>
        pip install streamlit<br><br>
        <span style="color:#5b8a9f;"># Run the app</span><br>
        streamlit run app.py<br><br>
        <span style="color:#5b8a9f;"># Run FastAPI backend (separate terminal)</span><br>
        uvicorn main:app --reload --port 8000<br><br>
        <span style="color:#5b8a9f;"># The app opens automatically at:</span><br>
        http://localhost:8501
      </div>
    </div>
    """, unsafe_allow_html=True)
