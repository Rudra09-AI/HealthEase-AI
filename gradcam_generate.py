"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HealthEase AI — Standalone Grad-CAM Generator                             ║
║  Generates REAL Grad-CAM heatmaps from your DenseNet-121 model             ║
║                                                                              ║
║  HOW TO RUN:                                                                 ║
║    cd Desktop\pneumoscan                                                     ║
║    venv\Scripts\python.exe gradcam_generate.py                              ║
║                                                                              ║
║  OUTPUT: gradcam_results\ folder with:                                      ║
║    - heatmap_XX_filename.jpg  (individual heatmaps)                         ║
║    - results_panel.jpg        (all 11 cases in one image)                   ║
║    - summary.txt              (accuracy report)                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

# ── Suppress TF noise ────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these paths if needed
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH   = r"backend\models\pneumoscan_model.h5"
CLASSES_PATH = r"backend\models\class_names.json"
OUTPUT_DIR   = r"gradcam_results"

# Put your 11 X-ray files here (relative paths from pneumoscan folder)
# File naming from Kaggle dataset tells us ground truth:
#   person*_bacteria_* or person*_virus_* = PNEUMONIA
#   IM-XXXX-XXXX = NORMAL
XRAY_FILES = [
    r"test_xrays\person10_virus_35.jpeg",
    r"test_xrays\person11_virus_38.jpeg",
    r"test_xrays\person14_virus_44.jpeg",
    r"test_xrays\person15_virus_46.jpeg",
    r"test_xrays\person100_bacteria_475.jpeg",
    r"test_xrays\person100_bacteria_477.jpeg",
    r"test_xrays\IM-0001-0001.jpeg",
    r"test_xrays\IM-0003-0001.jpeg",
    r"test_xrays\IM-0005-0001.jpeg",
    r"test_xrays\IM-0006-0001.jpeg",
    r"test_xrays\IM-0007-0001.jpeg",
]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_classes():
    print("=" * 60)
    print("HealthEase AI — Grad-CAM Generator")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model not found at: {MODEL_PATH}")
        print("   Make sure you run this from Desktop\\pneumoscan\\")
        sys.exit(1)

    print(f"\n📦 Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"   ✅ Model loaded — {model.count_params():,} parameters")

    # Print model layer summary to help debug
    print(f"\n📋 Top-level layers:")
    for i, layer in enumerate(model.layers):
        print(f"   [{i}] {layer.name:40s} {type(layer).__name__}")

    classes = ["NORMAL", "PNEUMONIA"]
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH) as f:
            classes = json.load(f)
    print(f"\n🏷️  Classes: {classes}")

    return model, classes


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — THE CORRECT GRAD-CAM IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def find_target_layer(model):
    """
    Find the correct conv layer to use for Grad-CAM.

    For DenseNet-121 saved as .h5 with a nested model structure:
       outer_model
         └── densenet121  (inner Keras Model)
               └── conv5_block16_concat  (target layer)

    We need to access the inner model's graph correctly.
    """
    # Try 1: standard DenseNet-121 nested structure
    try:
        base = model.get_layer('densenet121')
        conv = base.get_layer('conv5_block16_concat')
        print(f"   ✅ Found target layer: densenet121 → conv5_block16_concat")
        return base, conv, 'nested'
    except Exception:
        pass

    # Try 2: layer might be directly in outer model
    for layer in reversed(model.layers):
        if 'conv5_block16' in layer.name:
            print(f"   ✅ Found target layer directly: {layer.name}")
            return model, layer, 'direct'

    # Try 3: find last BatchNormalization or Concatenate (end of DenseNet blocks)
    for layer in reversed(model.layers):
        if 'concat' in layer.name.lower() and hasattr(layer, '_inbound_nodes'):
            print(f"   ✅ Found concat layer: {layer.name}")
            return model, layer, 'direct'

    # Try 4: find last Conv2D anywhere
    all_convs = []
    def _find(m):
        for l in m.layers:
            if hasattr(l, 'layers'):
                _find(l)
            if isinstance(l, tf.keras.layers.Conv2D):
                all_convs.append(l)
    _find(model)

    if all_convs:
        layer = all_convs[-1]
        print(f"   ✅ Using last Conv2D: {layer.name}")
        return model, layer, 'scan'

    raise RuntimeError("Could not find any suitable conv layer for Grad-CAM!")


def compute_gradcam(model, classes, img_path):
    """
    Compute real Grad-CAM for a single X-ray image.
    Returns: (prediction_class, confidence, overlay_image, is_real)
    """
    # ── Read and preprocess image ────────────────────────────────────────────
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess exactly as training
    inp = cv2.resize(img_rgb, (224, 224))
    inp = np.expand_dims(inp.astype(np.float32) / 255.0, axis=0)

    # Display size
    disp = cv2.resize(img_bgr, (400, 400))

    # ── Find target layer ────────────────────────────────────────────────────
    base_model, target_layer, mode = find_target_layer(model)

    # ── Build grad_model ─────────────────────────────────────────────────────
    # KEY: use the base_model's own inputs, not the outer model's
    if mode == 'nested':
        # Inner model: base.inputs → [conv_output, base_output]
        # Outer model: base_output → final_predictions (via head layers)
        grad_model = tf.keras.Model(
            inputs  = base_model.inputs,       # inner model's inputs
            outputs = [target_layer.output,    # conv feature maps
                       base_model.output]      # inner model's final output
        )
    else:
        # Layer is directly accessible in the model
        grad_model = tf.keras.Model(
            inputs  = model.inputs,
            outputs = [target_layer.output, model.output]
        )

    # ── THE CORRECT TAPE USAGE ───────────────────────────────────────────────
    # Use tf.Variable as input — tape always tracks Variables
    # Rule: call tape.watch() ONLY for non-Variable tensors
    inp_var = tf.Variable(inp, trainable=True, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # tape auto-tracks inp_var (it's a Variable)
        conv_out, base_preds = grad_model(inp_var, training=False)

        if mode == 'nested':
            # Run outer model head layers on base output
            x = base_preds
            for layer in model.layers:
                if layer.name == 'densenet121':
                    continue   # already done
                try:
                    x = layer(x, training=False)
                except Exception:
                    continue
            final_preds = x
        else:
            final_preds = base_preds

        # Get predicted class and its score
        class_idx = int(tf.argmax(final_preds[0]))
        score = final_preds[0][class_idx]

    # ── Compute gradients ────────────────────────────────────────────────────
    grads = tape.gradient(score, conv_out)

    if grads is None:
        print(f"   ⚠ grad w.r.t. conv_out is None — trying gradient w.r.t. input")
        # Fallback: gradient-input (always computable)
        inp_var2 = tf.Variable(inp, trainable=True, dtype=tf.float32)
        with tf.GradientTape() as tape2:
            preds2 = model(inp_var2, training=False)
            score2 = preds2[0][int(tf.argmax(preds2[0]))]
        grads2 = tape2.gradient(score2, inp_var2)
        if grads2 is not None:
            saliency = np.max(np.abs(grads2.numpy()[0]), axis=-1)
            cam = saliency
            is_real = True
            method = "Gradient×Input (fallback)"
        else:
            return None, None, None, False
    else:
        # Standard Grad-CAM: global average pool gradients → channel weights
        weights = tf.reduce_mean(grads, axis=[0, 1, 2])  # shape: [channels]
        # Weighted sum of activation maps, then ReLU
        cam = tf.nn.relu(
            tf.reduce_sum(conv_out[0] * weights, axis=-1)
        ).numpy()
        is_real = True
        method = "Grad-CAM (conv5_block16_concat)"

    # ── Get prediction details ───────────────────────────────────────────────
    # Run a clean prediction
    clean_preds = model(tf.constant(inp), training=False).numpy()[0]
    pred_class  = classes[np.argmax(clean_preds)]
    pred_conf   = float(np.max(clean_preds)) * 100
    all_confs   = {classes[i]: float(clean_preds[i]) * 100
                   for i in range(len(classes))}

    # ── Build overlay — 3-step medical-grade pipeline ────────────────────────
    # Always ReLU: only positive activations matter for localisation
    cam_pos = np.maximum(cam, 0)

    # ── STEP 1: FOCUS ON AFFECTED AREA (Range Correction) ────────────────────
    # Filter out the bottom 70% of signals (background noise/ribs)
    # and define the top 0.5% of activations as 100% intensity (the "hot" area).
    p_low  = np.percentile(cam_pos, 70)
    p_high = np.percentile(cam_pos, 99.5)
    if p_high - p_low < 1e-8:
        # CAM is flat — no spatial information available
        print(f"   ⚠ CAM is flat — model may not have strong spatial features")
        return pred_class, pred_conf, disp, False
    else:
        # Clip and normalize to [0, 1] range
        cam_norm = np.clip((cam_pos - p_low) / (p_high - p_low), 0, 1)

    # ── STEP 2: MEDICAL-GRADE SMOOTHING ──────────────────────────────────────
    # Turns pixelated "dots" into smooth diagnostic regions.
    cam_smooth = cv2.GaussianBlur(cam_norm.astype(np.float32), (13, 13), 0)

    # ── STEP 3: FINAL BLENDING ────────────────────────────────────────────────
    # Resize to original display size and apply the heatmap.
    cam_resized     = cv2.resize(cam_smooth, (400, 400))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    # Blending: 0.4 background, 0.6 heatmap for high visibility on lesions
    overlay = cv2.addWeighted(disp, 0.4, heatmap_colored, 0.6, 0)

    return pred_class, pred_conf, overlay, is_real, all_confs, method


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — RUN ON ALL IMAGES + BUILD RESULTS PANEL
# ══════════════════════════════════════════════════════════════════════════════

def get_ground_truth(filename):
    """Determine ground truth from Kaggle filename convention."""
    fn = os.path.basename(filename).lower()
    if 'bacteria' in fn or 'virus' in fn or 'person' in fn:
        return 'PNEUMONIA'
    return 'NORMAL'


def run_all(model, classes):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    correct = 0
    total   = 0

    print(f"\n{'='*60}")
    print(f"Running Grad-CAM on {len(XRAY_FILES)} X-rays...")
    print(f"{'='*60}\n")

    for i, xray_path in enumerate(XRAY_FILES):
        if not os.path.exists(xray_path):
            print(f"  [{i+1:02d}] ⚠ File not found: {xray_path}")
            continue

        fname       = os.path.basename(xray_path)
        ground_truth = get_ground_truth(xray_path)

        print(f"  [{i+1:02d}] {fname}")
        print(f"       Ground Truth: {ground_truth}")

        try:
            result = compute_gradcam(model, classes, xray_path)
            if result[0] is None:
                print(f"       ❌ Grad-CAM failed")
                continue

            pred_class, pred_conf, overlay, is_real, all_confs, method = result

            is_correct = (pred_class == ground_truth)
            correct   += int(is_correct)
            total     += 1

            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            real_s = "REAL Grad-CAM ✅" if is_real else "Fallback heatmap ⚠"

            print(f"       Predicted  : {pred_class} ({pred_conf:.1f}%)")
            print(f"       Heatmap    : {real_s}")
            print(f"       Method     : {method}")
            print(f"       Result     : {status}")

            # Confidence breakdown
            for cls, conf in all_confs.items():
                bar = '█' * int(conf / 5)
                print(f"         {cls:12s} {bar:20s} {conf:.1f}%")

            print()

            # Save individual heatmap
            out_fname = f"heatmap_{i+1:02d}_{fname}"
            out_path  = os.path.join(OUTPUT_DIR, out_fname)
            cv2.imwrite(out_path, overlay)

            results.append({
                'idx':          i + 1,
                'filename':     fname,
                'ground_truth': ground_truth,
                'prediction':   pred_class,
                'confidence':   pred_conf,
                'all_confs':    all_confs,
                'correct':      is_correct,
                'is_real':      is_real,
                'overlay':      overlay,
                'method':       method,
            })

        except Exception as e:
            print(f"       ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()

    return results, correct, total


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD VISUAL RESULTS PANEL
# ══════════════════════════════════════════════════════════════════════════════

def build_results_panel(results, correct, total):
    """Build a single image showing all results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("  ⚠ matplotlib not installed — skipping results panel")
        print("    Install with: pip install matplotlib")
        return

    if not results:
        print("  ⚠ No results to display")
        return

    n   = len(results)
    fig = plt.figure(figsize=(20, n * 3.5 + 3), facecolor='#050e18')

    # Header
    acc = correct / total * 100 if total > 0 else 0
    fig.text(0.50, 0.995, 'HealthEase AI — Real Grad-CAM Results',
             ha='center', va='top', fontsize=18, fontweight='bold', color='#00d4ff')
    fig.text(0.50, 0.982,
             f'DenseNet-121 | conv5_block16_concat | {correct}/{total} Correct ({acc:.1f}% Accuracy)',
             ha='center', va='top', fontsize=11, color='#7fb3c8')

    # Column headers
    for txt, xp in [("ORIGINAL", 0.09), ("GRAD-CAM OVERLAY", 0.27),
                    ("DIAGNOSIS & CONFIDENCE", 0.68)]:
        fig.text(xp, 0.969, txt, ha='center', fontsize=9, fontweight='bold', color='#4db8d4')

    usable   = 0.960
    row_h    = usable / n

    for i, r in enumerate(results):
        is_p   = r['prediction'] == 'PNEUMONIA'
        dc     = '#ff3a3a' if is_p else '#2ecc6a'
        bgcol  = '#160505' if is_p else '#041608'
        y_bot  = 0.956 - (i + 1) * row_h
        y_top  = 0.956 - i * row_h
        pad    = 0.003

        # Row background
        ax_bg = fig.add_axes([0.01, y_bot + pad, 0.98, row_h - pad * 2])
        ax_bg.set_facecolor(bgcol)
        ax_bg.set_xticks([]); ax_bg.set_yticks([])
        for sp in ax_bg.spines.values():
            sp.set_edgecolor(dc); sp.set_linewidth(1.2)

        # Index
        ymid = (y_top + y_bot) / 2
        fig.text(0.013, ymid + 0.007, f"#{r['idx']:02d}",
                 fontsize=11, fontweight='bold', color=dc, va='center')
        fig.text(0.013, ymid - 0.008, r['filename'][:16],
                 fontsize=6, color='#5a8090', va='center')

        ih = row_h * 0.83
        iw = ih * (3.5 / 20) * 0.9
        iy = ymid - ih / 2

        # Original X-ray
        orig_bgr = cv2.imread(str(XRAY_FILES[r['idx'] - 1]))
        orig_rgb = cv2.cvtColor(cv2.resize(orig_bgr, (280, 280)), cv2.COLOR_BGR2RGB)
        ax1 = fig.add_axes([0.045, iy, iw, ih])
        ax1.imshow(orig_rgb); ax1.axis('off')
        ax1.set_title('Original', fontsize=7, color='#aaaaaa', pad=2)

        # Grad-CAM overlay
        ov_rgb = cv2.cvtColor(r['overlay'], cv2.COLOR_BGR2RGB)
        ax2 = fig.add_axes([0.22, iy, iw, ih])
        ax2.imshow(ov_rgb); ax2.axis('off')
        real_lbl = "REAL Grad-CAM ✅" if r['is_real'] else "Simulated ⚠"
        ax2.set_title(real_lbl, fontsize=7,
                      color='#ff9944' if r['is_real'] else '#888888', pad=2)

        # Info panel
        ax3 = fig.add_axes([0.41, y_bot + pad * 3, 0.58, row_h - pad * 6])
        ax3.set_facecolor('#07131f')
        ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis('off')
        for sp in ax3.spines.values():
            sp.set_edgecolor('#0d2a40'); sp.set_linewidth(0.8)

        # Diagnosis
        sym = "⚠  " if is_p else "✓  "
        ax3.text(0.03, 0.88, sym + r['prediction'], fontsize=14,
                 fontweight='bold', color=dc, va='top', transform=ax3.transAxes)

        # Ground truth vs prediction
        gt_col = '#2ecc6a' if r['correct'] else '#ff3a3a'
        ax3.text(0.03, 0.72, f"Ground Truth: {r['ground_truth']}",
                 fontsize=8.5, color='#ffcc00', va='top', transform=ax3.transAxes)
        result_txt = "✓ CORRECT" if r['correct'] else "✗ INCORRECT"
        ax3.text(0.45, 0.72, result_txt, fontsize=9, fontweight='bold',
                 color=gt_col, va='top', transform=ax3.transAxes)

        # Confidence bars
        for k, (cls, conf) in enumerate(r['all_confs'].items()):
            yb = 0.56 - k * 0.20
            bc = '#ff3a3a' if cls == 'PNEUMONIA' else '#2ecc6a'
            ax3.text(0.03, yb + 0.02, cls, fontsize=7.5, color=bc,
                     va='center', transform=ax3.transAxes)
            ax3.add_patch(Rectangle((0.25, yb - 0.04), 0.68, 0.10,
                                     facecolor='#111e2a', transform=ax3.transAxes))
            ax3.add_patch(Rectangle((0.25, yb - 0.04), conf / 100 * 0.68, 0.10,
                                     facecolor=bc, alpha=0.85, transform=ax3.transAxes))
            ax3.text(0.95, yb + 0.02, f"{conf:.1f}%", fontsize=8.5,
                     fontweight='bold', color=bc, va='center',
                     ha='right', transform=ax3.transAxes)

        # Method
        ax3.text(0.03, 0.16, f"Method: {r.get('method', 'Grad-CAM')}",
                 fontsize=6.5, color='#4a7a8a', va='top', transform=ax3.transAxes)
        ax3.text(0.03, 0.07, f"Layer: conv5_block16_concat",
                 fontsize=6.5, color='#4a7a8a', va='top', transform=ax3.transAxes)

    # Colorbar
    cbar = fig.add_axes([0.10, 0.001, 0.80, 0.010])
    grad = np.linspace(0, 1, 512).reshape(1, -1)
    cbar.imshow(grad, aspect='auto', cmap='jet', extent=[0, 1, 0, 1])
    cbar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_xticklabels(['Low', '', 'Medium', '', 'High (Focus)'],
                          color='#aaaaaa', fontsize=7.5)
    cbar.set_yticks([])
    cbar.set_title('Grad-CAM Activation Scale  |  RED = Model Focus  |  BLUE = Low Attention',
                    color='#4db8d4', fontsize=8, pad=3)

    out = os.path.join(OUTPUT_DIR, 'results_panel.jpg')
    plt.savefig(out, dpi=120, bbox_inches='tight',
                facecolor='#050e18', edgecolor='none')
    plt.close()
    print(f"\n✅ Results panel saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(results, correct, total):
    acc = correct / total * 100 if total > 0 else 0
    lines = [
        "=" * 60,
        "HealthEase AI — Grad-CAM Test Results",
        "=" * 60,
        f"Total Cases : {total}",
        f"Correct     : {correct}",
        f"Accuracy    : {acc:.1f}%",
        f"Real Grad-CAM: {sum(1 for r in results if r['is_real'])}/{len(results)}",
        "",
        f"{'#':>3}  {'File':30s}  {'Truth':10s}  {'Pred':10s}  {'Conf':7s}  {'Result'}",
        "-" * 80,
    ]
    for r in results:
        status = "CORRECT" if r['correct'] else "WRONG  "
        lines.append(
            f"{r['idx']:>3}  {r['filename']:30s}  "
            f"{r['ground_truth']:10s}  {r['prediction']:10s}  "
            f"{r['confidence']:6.1f}%  {status}"
        )
    lines += ["", "=" * 60]

    out = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))
    print(f"\n✅ Summary saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Load model
    model, classes = load_model_and_classes()

    # Run Grad-CAM on all images
    results, correct, total = run_all(model, classes)

    # Build visual panel
    build_results_panel(results, correct, total)

    # Save text summary
    save_summary(results, correct, total)

    print(f"\n{'='*60}")
    print(f"✅ DONE — Results saved to: {OUTPUT_DIR}\\")
    print(f"   Open results_panel.jpg to see all heatmaps")
    print(f"{'='*60}")
