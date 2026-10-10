# Electric_Meter_Digitizer
# ⚡ AI-Powered Electricity Meter Digitizer  An intelligent computer vision project that aims to automate electricity meter reading from images using deep learning and AI-powered vision models.  The project combines **YOLO object detection** and **CNN-based digit recognition** to identify and read meter digits. 
"""
analyze_errors.py
-----------------
Error analysis for the Meter Digitizer pipeline.

Usage:
    python analyze_errors.py
    python analyze_errors.py --csv exact_reading_results.csv --out error_analysis
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ----------------------------------------------------------------------
# Configuration – change these if your CSV columns have different names
# ----------------------------------------------------------------------
DEFAULT_CSV = "exact_reading_results.csv"
PRED_COL = "predicted"
TRUE_COL = "ground_truth"
IMAGE_COL = "image"
CONF_COL = "confidence"

DIGIT_CLASSES = [str(i) for i in range(10)]


def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Load and basic-clean the results CSV."""
    df = pd.read_csv(csv_path)

    # Auto-detect common column names
    col_map = {}
    for col in df.columns:
        low = col.lower().strip()
        if low in ("predicted", "pred", "y_pred", "pred_reading", "prediction"):
            col_map[PRED_COL] = col
        elif low in ("ground_truth", "true", "y_true", "true_reading", "label", "gt"):
            col_map[TRUE_COL] = col
        elif low in ("image", "filename", "file", "path", "img"):
            col_map[IMAGE_COL] = col
        elif low in ("confidence", "conf", "score", "avg_conf"):
            col_map[CONF_COL] = col

    df = df.rename(columns={v: k for k, v in col_map.items()})

    required = [PRED_COL, TRUE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Please edit PRED_COL / TRUE_COL at the top of this script."
        )

    # Keep only digits
    df[PRED_COL] = (
        df[PRED_COL]
        .astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.strip()
    )
    df[TRUE_COL] = (
        df[TRUE_COL]
        .astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.strip()
    )

    df = df[(df[PRED_COL] != "") & (df[TRUE_COL] != "")].copy()
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} valid samples from {csv_path}")
    return df


def exact_match_accuracy(df: pd.DataFrame) -> float:
    return (df[PRED_COL] == df[TRUE_COL]).mean()


def digit_level_metrics(df: pd.DataFrame):
    y_true, y_pred = [], []

    for _, row in df.iterrows():
        t = row[TRUE_COL]
        p = row[PRED_COL]

        max_len = max(len(t), len(p))
        t = t.rjust(max_len, " ")
        p = p.rjust(max_len, " ")

        for tc, pc in zip(t, p):
            if tc == " " or pc == " ":
                y_true.append(tc if tc != " " else "?")
                y_pred.append(pc if pc != " " else "?")
            else:
                y_true.append(tc)
                y_pred.append(pc)

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, out_dir: Path, title="Digit Confusion Matrix"):
    mask = [(t in DIGIT_CLASSES) and (p in DIGIT_CLASSES) for t, p in zip(y_true, y_pred)]
    yt = [t for t, m in zip(y_true, mask) if m]
    yp = [p for p, m in zip(y_pred, mask) if m]

    if not yt:
        print("No pure digit pairs for confusion matrix.")
        return

    cm = confusion_matrix(yt, yp, labels=DIGIT_CLASSES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=DIGIT_CLASSES, yticklabels=DIGIT_CLASSES, ax=axes[0],
    )
    axes[0].set_title(f"{title} (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=DIGIT_CLASSES, yticklabels=DIGIT_CLASSES, ax=axes[1],
    )
    axes[1].set_title(f"{title} (row-normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    path = out_dir / "digit_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix → {path}")


def analyze_common_errors(df: pd.DataFrame, top_n: int = 15):
    errors = df[df[PRED_COL] != df[TRUE_COL]]
    print(f"\n=== Whole-reading errors: {len(errors)} / {len(df)} "
          f"({100 * len(errors) / len(df):.1f}%) ===")

    if errors.empty:
        print("No errors – perfect score!")
        return

    pairs = Counter(zip(errors[TRUE_COL], errors[PRED_COL]))
    print(f"\nTop {top_n} reading errors (true → predicted):")
    for (t, p), cnt in pairs.most_common(top_n):
        print(f"  {t:>10} → {p:<10}  ({cnt} times)")

    subst = Counter()
    for _, row in errors.iterrows():
        t, p = row[TRUE_COL], row[PRED_COL]
        if len(t) != len(p):
            continue
        for tc, pc in zip(t, p):
            if tc != pc:
                subst[(tc, pc)] += 1

    if subst:
        print(f"\nTop digit substitutions (true → pred):")
        for (t, p), cnt in subst.most_common(top_n):
            print(f"  {t} → {p}  ({cnt} times)")


def length_mismatch_stats(df: pd.DataFrame):
    same = (df[PRED_COL].str.len() == df[TRUE_COL].str.len()).mean()
    print(f"\nLength match rate: {100 * same:.1f}%")
    mismatches = df[df[PRED_COL].str.len() != df[TRUE_COL].str.len()]
    if not mismatches.empty:
        print("Examples of length mismatches:")
        for _, row in mismatches.head(8).iterrows():
            img = row.get(IMAGE_COL, "")
            print(f"  true={row[TRUE_COL]!r}  pred={row[PRED_COL]!r}  {img}")


def confidence_analysis(df: pd.DataFrame):
    if CONF_COL not in df.columns:
        return

    df = df.copy()
    df["correct"] = df[PRED_COL] == df[TRUE_COL]
    df[CONF_COL] = pd.to_numeric(df[CONF_COL], errors="coerce")
    df = df.dropna(subset=[CONF_COL])

    if df.empty:
        return

    print("\n=== Confidence analysis ===")
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
    labels = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", ">=0.95"]
    df["conf_bin"] = pd.cut(df[CONF_COL], bins=bins, labels=labels, right=False)

    summary = df.groupby("conf_bin", observed=True).agg(
        count=("correct", "size"),
        accuracy=("correct", "mean"),
    )
    print(summary.to_string())


def save_error_samples(df: pd.DataFrame, out_dir: Path, n: int = 30):
    errors = df[df[PRED_COL] != df[TRUE_COL]].copy()
    if errors.empty:
        return

    if CONF_COL in errors.columns:
        errors = errors.sort_values(CONF_COL, ascending=True)

    path = out_dir / "error_samples.csv"
    cols = [c for c in [IMAGE_COL, TRUE_COL, PRED_COL, CONF_COL] if c in errors.columns]
    errors[cols].head(n).to_csv(path, index=False)
    print(f"Saved {min(n, len(errors))} error samples → {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze meter reading errors")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to results CSV")
    parser.add_argument("--out", default="error_analysis", help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        candidates = list(Path(".").rglob(csv_path.name))
        if candidates:
            csv_path = candidates[0]
            print(f"Found CSV at {csv_path}")
        else:
            raise FileNotFoundError(
                f"Could not find {args.csv}. "
                "Run the full pipeline first or pass --csv path/to/file.csv"
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)

    exact_acc = exact_match_accuracy(df)
    print(f"\nExact reading accuracy: {100 * exact_acc:.2f}%")

    y_true, y_pred = digit_level_metrics(df)
    pure = [(t, p) for t, p in zip(y_true, y_pred) if t in DIGIT_CLASSES and p in DIGIT_CLASSES]
    digit_acc = 0.0
    if pure:
        yt, yp = zip(*pure)
        digit_acc = accuracy_score(yt, yp)
        print(f"Digit-level accuracy  : {100 * digit_acc:.2f}%  (n={len(yt)})")
        print("\nClassification report (digits 0-9):")
        print(classification_report(yt, yp, labels=DIGIT_CLASSES, digits=3))

    length_mismatch_stats(df)
    analyze_common_errors(df)
    confidence_analysis(df)

    plot_confusion_matrix(y_true, y_pred, out_dir)
    save_error_samples(df, out_dir)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Samples          : {len(df)}\n")
        f.write(f"Exact accuracy   : {100 * exact_acc:.2f}%\n")
        if pure:
            f.write(f"Digit accuracy   : {100 * digit_acc:.2f}%\n")
    print(f"\nSummary written → {summary_path}")
    print("Done.")
    import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# ================== CONFIG ==================
YOLO_MODEL_PATH = "weights/best.pt"
CNN_MODEL_PATH = "cnn_digit_model.h5"
CONF_THRESHOLD = 0.25
CNN_INPUT_SIZE = (32, 32)
MAX_DIGITS = 6
LOW_CONFIDENCE_THRESHOLD = 0.85
# ============================================

st.set_page_config(
    page_title="Meter Digitizer",
    page_icon="⚡",
    layout="centered"
)


@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL_PATH)
    cnn = load_model(CNN_MODEL_PATH)
    return yolo, cnn


def preprocess_crop(crop):
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    resized = cv2.resize(gray, CNN_INPUT_SIZE)
    return resized.astype("float32") / 255.0


def robust_order(detections):
    if len(detections) <= 1:
        return detections

    centers = np.array([[d["cx"], d["cy"]] for d in detections])

    if np.std(centers[:, 0]) > np.std(centers[:, 1]):
        slope, _ = np.polyfit(centers[:, 0], centers[:, 1], 1)
        projections = centers[:, 0] + slope * centers[:, 1]
    else:
        slope, _ = np.polyfit(centers[:, 1], centers[:, 0], 1)
        projections = centers[:, 1] + slope * centers[:, 0]

    return [detections[i] for i in np.argsort(projections)]


def apply_length_constraint(detections, max_digits=MAX_DIGITS):
    if len(detections) <= max_digits:
        return sorted(detections, key=lambda d: d["cx"])

    top = sorted(
        detections,
        key=lambda d: d["conf"],
        reverse=True
    )[:max_digits]

    return sorted(top, key=lambda d: d["cx"])


def normalize_reading(reading):
    if not reading or reading == "NO_DETECTION":
        return reading

    cleaned = reading.lstrip("0")
    return cleaned if cleaned else "0"


def read_meter(img, yolo_model, cnn_model):
    results = yolo_model.predict(
        source=img,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        return "NO_DETECTION", 0.0, []

    crops = []
    metadata = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Keep crop coordinates inside the image
        h, w = img.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crops.append(preprocess_crop(crop))
        metadata.append({
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "box": (x1, y1, x2, y2)
        })

    if not crops:
        return "NO_DETECTION", 0.0, []

    batch = np.array(crops)[..., np.newaxis]
    predictions = cnn_model.predict(batch, verbose=0)

    detections = []

    for i, prediction in enumerate(predictions):
        detections.append({
            "digit": int(np.argmax(prediction)),
            "conf": float(np.max(prediction)),
            "cx": metadata[i]["cx"],
            "cy": metadata[i]["cy"],
            "box": metadata[i]["box"]
        })

    detections = robust_order(detections)
    detections = apply_length_constraint(detections)

    reading = "".join(str(d["digit"]) for d in detections)

    # Geometric mean is less harsh than multiplying all digit probabilities.
    # It is still a model-derived confidence estimate, not a calibrated
    # probability that the complete reading is correct.
    if detections:
        confidence = float(
            np.mean([d["conf"] for d in detections])
        )
    else:
        confidence = 0.0

    return reading, confidence, detections


# ================== STREAMLIT UI ==================
st.title("⚡ Electric Meter Digitizer")

st.write(
    "Capture a meter using your camera or upload an existing photo. "
    "Always verify the displayed reading against the physical meter."
)

try:
    yolo_model, cnn_model = load_models()
except Exception as exc:
    st.error(
        "Could not load the trained models. Check that "
        "'weights/best.pt' and 'cnn_digit_model.h5' exist."
    )
    st.exception(exc)
    st.stop()


input_method = st.radio(
    "Choose image source",
    ["📷 Take a photo", "📁 Upload an image"],
    horizontal=True
)

image_file = None

if input_method == "📷 Take a photo":
    image_file = st.camera_input("Take a photo of the meter display")
else:
    image_file = st.file_uploader(
        "Upload a meter image",
        type=["jpg", "jpeg", "png"]
    )

if image_file is not None:
    try:
        file_bytes = np.frombuffer(image_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("The image could not be read. Try another photo.")
            st.stop()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.subheader("Input Image")
        st.image(img_rgb, caption="Meter image", use_container_width=True)

        with st.spinner("Detecting and reading digits..."):
            reading, confidence, detections = read_meter(
                img,
                yolo_model,
                cnn_model
            )

        st.divider()
        st.subheader("Meter Reading")

        if reading == "NO_DETECTION":
            st.error(
                "No digits were detected. Try better lighting, "
                "a sharper photo, and a closer view of the display."
            )
        else:
            normalized = normalize_reading(reading)

            st.metric("Predicted Reading (kWh)", normalized)
            st.write(f"**Raw detected digits:** `{reading}`")
            st.write(f"**Mean digit confidence:** {confidence:.1%}")

            if confidence < LOW_CONFIDENCE_THRESHOLD:
                st.warning(
                    "Low confidence: please verify every digit manually "
                    "before recording this reading."
                )
            else:
                st.success(
                    "The digit classifier reports high average confidence. "
                    "Please still verify the complete reading manually."
                )

            if detections:
                st.write("**Individual digit confidence**")

                digit_cols = st.columns(len(detections))

                for index, detection in enumerate(detections):
                    with digit_cols[index]:
                        st.metric(
                            f"Digit {index + 1}",
                            detection["digit"],
                            f"{detection['conf']:.1%}"
                        )

            # Show the detection boxes and predicted digits
            annotated = img_rgb.copy()

            for detection in detections:
                x1, y1, x2, y2 = map(int, detection["box"])

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    str(detection["digit"]),
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            st.subheader("Detected Digits")
            st.image(
                annotated,
                caption="Predicted digits and detection boxes",
                use_container_width=True
            )

    except Exception as exc:
        st.error("An error occurred while processing the image.")
        st.exception(exc)

st.divider()
st.caption(
    "Meter Digitizer | YOLO digit detection + CNN classification"
)
from ultralytics import YOLO

model = YOLO("weights/best.pt")

metrics = model.val(
    data="data/synthetic/data.yaml",
    imgsz=640,
    batch=16
)

print("\n===== YOLO Detector Metrics =====")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
import pandas as pd

# Load evaluation results
df = pd.read_csv("exact_reading_results.csv")

# Normalize Boolean columns
df["correct"] = df["correct"].astype(str).str.lower().eq("true")
df["flagged"] = df["flagged"].astype(str).str.lower().eq("true")

# Overall performance
print("\n===== MODEL PERFORMANCE =====")
print(f"Total evaluated: {len(df)}")
print(f"Correct: {int(df['correct'].sum())}")
print(f"Incorrect: {int((~df['correct']).sum())}")
print(f"Accuracy: {df['correct'].mean() * 100:.2f}%")
print(f"Flagged for review: {int(df['flagged'].sum())}")

# Accuracy by difficulty
print("\n===== ACCURACY BY DIFFICULTY =====")

summary = df.groupby("difficulty", dropna=False)["correct"].agg(
    total="count",
    accuracy="mean"
)

summary["accuracy"] = (summary["accuracy"] * 100).round(2)
print(summary.to_string())

# Display incorrect predictions
print("\n===== INCORRECT PREDICTIONS =====")

errors = df.loc[~df["correct"]]

if errors.empty:
    print("No incorrect predictions found.")
else:
    columns = [
        "image",
        "true_raw",
        "true_norm",
        "pred_raw",
        "pred_norm",
        "conf",
        "difficulty",
        "flagged"
    ]

    # Handle any optional columns safely
    available_columns = [col for col in columns if col in errors.columns]

    print(errors[available_columns].head(20).to_string(index=False))

print("\n===== EVALUATION COMPLETE =====")
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ================== CONFIG ==================
YOLO_MODEL_PATH = "weights/best.pt"
CNN_MODEL_PATH  = "cnn_digit_model.h5"
IMAGES_DIR      = "data/synthetic/images"
META_CSV        = "data/synthetic/meta.csv"
CONF_THRESHOLD  = 0.25
CNN_INPUT_SIZE  = (32, 32)
CONFIDENCE_FLAG = 0.85
SAMPLE_SIZE     = 400
# ============================================

print("Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
cnn_model  = load_model(CNN_MODEL_PATH)
print("Models loaded.\n")

def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    resized = cv2.resize(gray, CNN_INPUT_SIZE)
    normalized = resized.astype("float32") / 255.0
    return normalized

def robust_order(detections):
    if len(detections) <= 1:
        return detections
    centers = np.array([[d["cx"], d["cy"]] for d in detections])
    if np.std(centers[:, 0]) > np.std(centers[:, 1]):
        m, _ = np.polyfit(centers[:, 0], centers[:, 1], 1)
        projections = centers[:, 0] + m * centers[:, 1]
    else:
        m, _ = np.polyfit(centers[:, 1], centers[:, 0], 1)
        projections = centers[:, 1] + m * centers[:, 0]
    order = np.argsort(projections)
    return [detections[i] for i in order]

def read_meter(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return "NO_DETECTION", 0.0

    results = yolo_model.predict(source=img, conf=CONF_THRESHOLD, verbose=False)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return "NO_DETECTION", 0.0

    crops, meta = [], []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(preprocess_crop(crop))
        meta.append({"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2})

    if not crops:
        return "NO_DETECTION", 0.0

    batch = np.array(crops)[..., np.newaxis]
    preds = cnn_model.predict(batch, verbose=0)

    detections = []
    for i, p in enumerate(preds):
        detections.append({
            "digit": int(np.argmax(p)),
            "conf": float(np.max(p)),
            "cx": meta[i]["cx"],
            "cy": meta[i]["cy"]
        })

    detections = robust_order(detections)
    reading = "".join(str(d["digit"]) for d in detections)
    conf_product = float(np.prod([d["conf"] for d in detections]))
    return reading, conf_product

def evaluate():
    meta = pd.read_csv(META_CSV)
    print("meta.csv columns:", meta.columns.tolist())

    # ---------- Better column detection ----------
    img_col = None
    reading_col = None
    diff_col = None

    for c in meta.columns:
        cl = c.lower().strip()
        if img_col is None and any(k in cl for k in ["image", "file", "name", "path"]):
            img_col = c
        if reading_col is None and any(k in cl for k in ["read", "gt", "true", "label", "value"]):
            reading_col = c
        if diff_col is None and any(k in cl for k in ["diff", "hard", "tag", "level", "type", "class"]):
            diff_col = c

    # Fallbacks
    if img_col is None:
        img_col = meta.columns[0]
    if reading_col is None:
        reading_col = meta.columns[1] if len(meta.columns) > 1 else meta.columns[0]

    # Guarantee difficulty column always exists
    if diff_col is None:
        print("WARNING: No difficulty-like column found. Creating 'difficulty' = 'unknown'")
        meta["difficulty"] = "unknown"
        diff_col = "difficulty"
    else:
        # Make sure the column is clean
        meta[diff_col] = meta[diff_col].fillna("unknown").astype(str)

    print(f"Using columns → image: '{img_col}', reading: '{reading_col}', difficulty: '{diff_col}'\n")

    # Create stem for matching images
    meta["stem"] = meta[img_col].astype(str).apply(lambda x: Path(x).stem)

    # ---------- Stratified sample ----------
    if meta[diff_col].nunique() > 1:
        sample = (
            meta.groupby(diff_col, group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), SAMPLE_SIZE // 3 + 30), random_state=42))
            .reset_index(drop=True)
        )
    else:
        sample = meta.sample(n=min(len(meta), SAMPLE_SIZE), random_state=42)

    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(n=SAMPLE_SIZE, random_state=42)

    results = []
    print(f"Evaluating {len(sample)} images...\n")

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Progress"):
        stem = row["stem"]
        img_path = Path(IMAGES_DIR) / f"{stem}.jpg"
        if not img_path.exists():
            continue

        pred_reading, pred_conf = read_meter(img_path)
        true_reading = str(row[reading_col]).zfill(6)

        # ---------- SAFE access (this was the crash point) ----------
        difficulty = row.get(diff_col, "unknown")

        correct = (pred_reading == true_reading)

        results.append({
            "image": img_path.name,
            "true": true_reading,
            "pred": pred_reading,
            "conf": round(pred_conf, 4),
            "correct": correct,
            "difficulty": difficulty,
            "flagged": pred_conf < CONFIDENCE_FLAG
        })

    df = pd.DataFrame(results)
    df.to_csv("exact_reading_results.csv", index=False)
    print("\nSaved → exact_reading_results.csv")

    overall = df["correct"].mean() * 100
    print(f"\n{'='*50}")
    print(f"OVERALL Exact Reading Accuracy : {overall:.2f}%")
    print(f"{'='*50}")

    print("\nBreakdown by difficulty:")
    for tag, group in df.groupby("difficulty"):
        acc = group["correct"].mean() * 100
        print(f"  {str(tag):12s} : {acc:6.2f}%  (n={len(group)})")

    wrong = df[~df["correct"]]
    if len(wrong) > 0:
        print(f"\nWrong readings : {len(wrong)}")
        print(f"  Flagged      : {wrong['flagged'].sum()}")
        print(f"  Silent       : {(~wrong['flagged']).sum()}")
    else:
        print("\nNo wrong readings in this sample.")

if __name__ == "__main__":
    test = "data/synthetic/images/meter_00000.jpg"
    r, c = read_meter(test)
    print(f"Test image : {test}")
    print(f"Reading    : {r}")
    print(f"Confidence : {c:.4f}\n")
    evaluate()
    """
Synthetic WAPDA DDS-65 style meter image generator.

Renders single phase static energy meters with an 8 digit seven segment LCD,
in the style of the meters DISCOs install in Pakistan (Vertex VTX-12,
KBK HXE12 and similar). Every image comes out with YOLO format labels for the
digits, because the generator knows exactly where it drew each one.

Segments are drawn as real polygons rather than with a font. A seven segment
digit has a defined geometry, so drawing it directly gives pixel accurate
digits instead of a font's approximation of them.

Run:  python generate_synthetic_meters.py --n 3000
Out:  data/synthetic/images/*.jpg
      data/synthetic/labels/*.txt     YOLO format, class = digit 0-9
      data/synthetic/meta.csv         true reading per image
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# --- Seven segment geometry ------------------------------------------------
#
#      aaaa
#     f    b
#     f    b
#      gggg
#     e    c
#     e    c
#      dddd

SEGMENTS = {
    "0": "abcdef",
    "1": "bc",
    "2": "abged",
    "3": "abgcd",
    "4": "fgbc",
    "5": "afgcd",
    "6": "afgedc",
    "7": "abc",
    "8": "abcdefg",
    "9": "afgbcd",
}

ALL_SEGMENTS = "abcdefg"


def segment_polys(x: float, y: float, w: float, h: float, t: float) -> dict:
    """
    Polygon for each segment of a digit whose bounding box is (x, y, w, h).
    t is segment thickness. Ends are mitred, which is what gives a seven
    segment display its characteristic notched corners.
    """
    hm = y + h / 2.0
    return {
        "a": [(x + t, y), (x + w - t, y), (x + w - t * 1.5, y + t), (x + t * 1.5, y + t)],
        "b": [(x + w, y + t), (x + w, y + h / 2 - t / 2), (x + w - t, hm - t / 2),
              (x + w - t, y + t * 1.5)],
        "c": [(x + w, hm + t / 2), (x + w, y + h - t), (x + w - t, y + h - t * 1.5),
              (x + w - t, hm + t / 2)],
        "d": [(x + t, y + h), (x + w - t, y + h), (x + w - t * 1.5, y + h - t),
              (x + t * 1.5, y + h - t)],
        "e": [(x, hm + t / 2), (x + t, hm + t / 2), (x + t, y + h - t * 1.5),
              (x, y + h - t)],
        "f": [(x, y + t), (x + t, y + t * 1.5), (x + t, hm - t / 2), (x, hm - t / 2)],
        "g": [(x + t * 0.6, hm), (x + t * 1.6, hm - t / 2), (x + w - t * 1.6, hm - t / 2),
              (x + w - t * 0.6, hm), (x + w - t * 1.6, hm + t / 2),
              (x + t * 1.6, hm + t / 2)],
    }


# --- Meter body ------------------------------------------------------------

BODY_GREYS = [(214, 214, 210), (226, 226, 222), (203, 205, 202), (231, 230, 226)]
LCD_GREENS = [(150, 168, 132), (158, 174, 140), (140, 158, 126), (166, 180, 150)]
LCD_GREYS = [(176, 182, 176), (186, 190, 184)]

BRANDS = ["VERTEX", "KBK", "MICROTEK", "PEL", "SIEMENS PAK", "ESCORTS", "TESLA IND"]
TYPES = ["VTX-12", "HXE-12", "DDS-65", "SP-240", "EM-1P", "MTR-40A"]


WALL_COLOURS = [
    (198, 190, 176), (172, 168, 160), (214, 206, 190), (150, 148, 144),
    (186, 176, 158), (206, 202, 198), (160, 152, 138), (222, 216, 204),
]


def background(W: int, H: int, rng: random.Random) -> Image.Image:
    """
    Wall or meter cupboard behind the meter.

    A plain white background would teach the detector that anything not white
    is a meter. Real photos have plaster, paint, shadow and conduit, and the
    detector needs to have seen that before it meets one.
    """
    base = rng.choice(WALL_COLOURS)
    arr = np.full((H, W, 3), base, dtype=np.float32)

    # Plaster grain
    arr += np.random.normal(0, rng.uniform(4, 12), (H, W, 1))

    # Uneven lighting across the wall
    gy, gx = np.mgrid[0:H, 0:W]
    ang = rng.uniform(0, 2 * math.pi)
    ramp = (gx * math.cos(ang) + gy * math.sin(ang)) / max(W, H)
    arr += (ramp[..., None] - 0.5) * rng.uniform(10, 55)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)

    # Occasional conduit or cable running behind the meter
    if rng.random() < 0.45:
        cx = rng.randint(0, W)
        wdt = rng.randint(6, 20)
        shade = tuple(max(0, c - rng.randint(25, 60)) for c in base)
        d.rectangle([cx, 0, cx + wdt, H], fill=shade)

    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))

    return img


def draw_meter(
    reading: str,
    W: int = 640,
    H: int = 640,
    rng: random.Random | None = None,
) -> tuple[Image.Image, list[tuple[int, float, float, float, float]]]:
    """
    Draw one meter face. Returns the image and a list of
    (digit_class, cx, cy, w, h) in pixel coordinates.
    """
    rng = rng or random

    body = rng.choice(BODY_GREYS)
    img = background(W, H, rng)
    d = ImageDraw.Draw(img)

    # Meter housing. Kept large in frame because that is how people actually
    # photograph a meter, standing close enough to read it.
    mx, my = int(W * 0.07), int(H * 0.03)
    mw, mh = int(W * 0.86), int(H * 0.94)
    d.rounded_rectangle([mx, my, mx + mw, my + mh], radius=18, fill=body,
                        outline=(150, 150, 148), width=3)

    # Upper panel, slightly lighter
    d.rounded_rectangle(
        [mx + 8, my + 8, mx + mw - 8, my + int(mh * 0.52)],
        radius=12, fill=tuple(min(255, c + 10) for c in body),
        outline=(180, 180, 176), width=2,
    )

    # Header text
    d.text((mx + int(mw * 0.16), my + 18),
           "Static Single Phase Two Wire", fill=(70, 70, 70))
    d.text((mx + int(mw * 0.30), my + 32), "Energy Meter", fill=(70, 70, 70))

    brand = rng.choice(BRANDS)
    d.text((mx + 16, my + int(mh * 0.40)), brand, fill=(40, 40, 40))
    d.text((mx + mw - 96, my + 20), f"Type:", fill=(70, 70, 70))
    d.text((mx + mw - 96, my + 32), rng.choice(TYPES), fill=(70, 70, 70))

    # --- LCD window --------------------------------------------------------
    lcd_w = int(mw * 0.70)
    lcd_h = int(mh * 0.22)
    lcd_x = mx + int((mw - lcd_w) / 2)
    lcd_y = my + int(mh * 0.12)

    lcd_bg = rng.choice(LCD_GREENS if rng.random() < 0.75 else LCD_GREYS)
    d.rectangle([lcd_x, lcd_y, lcd_x + lcd_w, lcd_y + lcd_h],
                fill=lcd_bg, outline=(90, 95, 90), width=2)

    # Digit layout inside the LCD
    n = len(reading)
    pad_x = lcd_w * 0.07
    pad_y = lcd_h * 0.16
    avail_w = lcd_w - 2 * pad_x
    gap = avail_w * 0.035
    dw = (avail_w - gap * (n - 1)) / n
    dh = lcd_h - 2 * pad_y
    thick = max(2.5, dh * 0.13)

    on_colour = (30, 38, 30)
    # Unlit segments are faintly visible on a real LCD. Including them matters:
    # a model trained without ghost segments learns that absence of ink means
    # absence of segment, and then fails on real photos where the ghosts show.
    ghost = tuple(int(c * 0.83) for c in lcd_bg)

    boxes = []
    for i, ch in enumerate(reading):
        dx = lcd_x + pad_x + i * (dw + gap)
        dy = lcd_y + pad_y
        polys = segment_polys(dx, dy, dw, dh, thick)

        for seg in ALL_SEGMENTS:
            if seg not in SEGMENTS[ch]:
                d.polygon(polys[seg], fill=ghost)
        for seg in SEGMENTS[ch]:
            d.polygon(polys[seg], fill=on_colour)

        boxes.append((int(ch), dx + dw / 2, dy + dh / 2, dw * 1.20, dh * 1.10))

    # kWh label at the right edge of the LCD
    d.text((lcd_x + lcd_w - 26, lcd_y + lcd_h - 14), "kWh", fill=(45, 52, 45))

    # --- Lower details -----------------------------------------------------
    labels = ["Earth", "Neutral", "Imp/kWh", "Reverse", "Power"]
    lx = mx + int(mw * 0.10)
    ly = my + int(mh * 0.33)
    for i, lab in enumerate(labels):
        cx = lx + i * int(mw * 0.17)
        led = (220, 60, 50) if rng.random() < 0.35 else (150, 148, 145)
        d.ellipse([cx, ly, cx + 7, ly + 7], fill=led)
        d.text((cx - 6, ly + 10), lab, fill=(80, 80, 80))

    serial = f"{rng.randint(10**7, 10**8 - 1)}"
    d.text((mx + 16, my + int(mh * 0.46)), f"Sr.No. {serial}", fill=(60, 60, 60))
    d.text((mx + 16, my + int(mh * 0.50)),
           "240V  10(40)A  50Hz", fill=(60, 60, 60))
    d.text((mx + mw - 110, my + int(mh * 0.46)), "Made in Pakistan", fill=(60, 60, 60))

    # Terminal block cover
    tb_y = my + int(mh * 0.58)
    d.rounded_rectangle([mx + 10, tb_y, mx + mw - 10, my + mh - 10],
                        radius=8, fill=(206, 208, 206),
                        outline=(160, 160, 158), width=2)
    for i in range(4):
        cx = mx + int(mw * 0.16) + i * int(mw * 0.20)
        d.rounded_rectangle([cx, tb_y + 22, cx + 34, tb_y + 62],
                            radius=4, fill=(178, 180, 178), outline=(140, 140, 138))
        d.ellipse([cx + 10, tb_y + 32, cx + 24, tb_y + 46], fill=(120, 120, 118))

    return img, boxes


# --- Degradation -----------------------------------------------------------

def apply_perspective(img: Image.Image, boxes: list, rng: random.Random,
                      strength: float) -> tuple[Image.Image, list]:
    """Tilt the meter as if photographed off axis."""
    import cv2

    W, H = img.size
    m = strength * min(W, H)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([
        [rng.uniform(0, m), rng.uniform(0, m)],
        [W - rng.uniform(0, m), rng.uniform(0, m)],
        [W - rng.uniform(0, m), H - rng.uniform(0, m)],
        [rng.uniform(0, m), H - rng.uniform(0, m)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)

    arr = cv2.warpPerspective(
        np.array(img), M, (W, H),
        borderMode=cv2.BORDER_REPLICATE,
    )

    new_boxes = []
    for cls, cx, cy, bw, bh in boxes:
        corners = np.float32([
            [[cx - bw / 2, cy - bh / 2]], [[cx + bw / 2, cy - bh / 2]],
            [[cx + bw / 2, cy + bh / 2]], [[cx - bw / 2, cy + bh / 2]],
        ])
        warped = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
        x0, y0 = warped.min(axis=0)
        x1, y1 = warped.max(axis=0)
        new_boxes.append((cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))

    return Image.fromarray(arr), new_boxes


def add_glare(img: Image.Image, rng: random.Random) -> Image.Image:
    """
    Specular highlight off the polycarbonate cover. This is the single most
    common reason a real meter photo fails, so it has to be in training.
    """
    W, H = img.size
    overlay = Image.new("L", (W, H), 0)
    od = ImageDraw.Draw(overlay)

    cx = rng.uniform(W * 0.15, W * 0.85)
    cy = rng.uniform(H * 0.05, H * 0.55)
    rw = rng.uniform(W * 0.12, W * 0.45)
    rh = rng.uniform(H * 0.05, H * 0.22)
    angle = rng.uniform(-40, 40)

    box = Image.new("L", (int(rw * 2), int(rh * 2)), 0)
    ImageDraw.Draw(box).ellipse([0, 0, int(rw * 2) - 1, int(rh * 2) - 1],
                                fill=rng.randint(90, 210))
    box = box.rotate(angle, expand=True)
    overlay.paste(box, (int(cx - box.width / 2), int(cy - box.height / 2)), box)
    overlay = overlay.filter(ImageFilter.GaussianBlur(rng.uniform(12, 34)))

    white = Image.new("RGB", (W, H), (255, 255, 255))
    return Image.composite(white, img, overlay.point(lambda v: int(v * 0.85)))


def add_dust(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, rng.uniform(3, 11), arr.shape)
    film = rng.uniform(0.0, 0.16)
    arr = arr * (1 - film) + 205 * film + noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def degrade(img: Image.Image, boxes: list, rng: random.Random, level: str):
    """
    level: clean | normal | hard
    Controls how much of the real world gets simulated.
    """
    if level == "clean":
        return img, boxes

    strength = 0.055 if level == "normal" else 0.115
    if rng.random() < (0.7 if level == "normal" else 0.95):
        img, boxes = apply_perspective(img, boxes, rng, strength)

    if rng.random() < (0.35 if level == "normal" else 0.75):
        img = add_glare(img, rng)

    if rng.random() < 0.6:
        img = add_dust(img, rng)

    if rng.random() < (0.4 if level == "normal" else 0.8):
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.9)))

    # Brightness and contrast, covering dim meter cupboards and harsh sun
    arr = np.array(img).astype(np.float32)
    brightness = rng.uniform(0.55, 1.35) if level == "hard" else rng.uniform(0.78, 1.18)
    contrast = rng.uniform(0.62, 1.25) if level == "hard" else rng.uniform(0.85, 1.12)
    arr = (arr - 128) * contrast + 128 * brightness
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-6, 6), resample=Image.BICUBIC,
                         fillcolor=(230, 230, 228))

    return img, boxes


# --- Readings --------------------------------------------------------------

def random_reading(rng: random.Random, n_digits: int = 6) -> str:
    """
    Digit strings that look like real meter totals rather than uniform noise.

    Uniform random digits would give every class equal frequency, which is not
    what a meter shows. Leading zeros are common on newer meters and rare on
    old ones, and that imbalance is worth reproducing so students meet it.
    """
    style = rng.random()
    if style < 0.30:
        value = rng.randint(0, 999)
    elif style < 0.70:
        value = rng.randint(1000, 99999)
    else:
        value = rng.randint(100000, 999999)
    return str(value).zfill(n_digits)


# --- Main ------------------------------------------------------------------

def to_yolo(boxes: list, W: int, H: int) -> list[str]:
    lines = []
    for cls, cx, cy, bw, bh in boxes:
        # Clip to frame. A perspective warp can push a digit partly outside.
        x0 = max(0.0, cx - bw / 2)
        y0 = max(0.0, cy - bh / 2)
        x1 = min(float(W), cx + bw / 2)
        y1 = min(float(H), cy + bh / 2)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        lines.append(
            f"{cls} {((x0 + x1) / 2) / W:.6f} {((y0 + y1) / 2) / H:.6f} "
            f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="images to generate")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--digits", type=int, default=6)
    ap.add_argument("--seed", type=int, default=365)
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    crop_dir = os.path.join(args.out, "digit_crops")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for c in range(10):
        os.makedirs(os.path.join(crop_dir, str(c)), exist_ok=True)

    meta = []
    counts = {str(c): 0 for c in range(10)}

    # Difficulty mix. Most images are "normal"; a slice is clean so the model
    # has easy examples to anchor on, and a slice is hard so it sees the worst
    # case during training rather than for the first time on a real photo.
    levels = ["clean"] * int(args.n * 0.15) + \
             ["normal"] * int(args.n * 0.55) + \
             ["hard"] * (args.n - int(args.n * 0.15) - int(args.n * 0.55))
    rng.shuffle(levels)

    for i in range(args.n):
        reading = random_reading(rng, args.digits)
        level = levels[i]

        img, boxes = draw_meter(reading, args.size, args.size, rng)
        img, boxes = degrade(img, boxes, rng, level)

        name = f"meter_{i:05d}"
        img.save(os.path.join(img_dir, f"{name}.jpg"), quality=rng.randint(62, 94))

        lines = to_yolo(boxes, args.size, args.size)
        with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(lines))

        # Digit crops for the classifier arm, taken after degradation so they
        # carry the same glare and blur the detector will hand over at run time.
        for (cls, cx, cy, bw, bh) in boxes:
            x0, y0 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
            x1, y1 = int(min(args.size, cx + bw / 2)), int(min(args.size, cy + bh / 2))
            if x1 - x0 < 6 or y1 - y0 < 6:
                continue
            crop = img.crop((x0, y0, x1, y1)).resize((32, 48), Image.BILINEAR)
            crop.save(os.path.join(crop_dir, str(cls), f"{name}_{x0}.jpg"), quality=92)
            counts[str(cls)] += 1

        meta.append({"image": f"{name}.jpg", "reading": reading, "difficulty": level})

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.n}")

    with open(os.path.join(args.out, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "reading", "difficulty"])
        w.writeheader()
        w.writerows(meta)

    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(
            f"path: {os.path.abspath(args.out)}\n"
            "train: images\n"
            "val: images\n\n"
            "names:\n"
            + "".join(f"  {c}: '{c}'\n" for c in range(10))
        )

    print(f"\nGenerated {args.n} images in {args.out}")
    print("Digit class counts:", counts)
    total = sum(counts.values())
    print(f"Total digit crops: {total}")
    print(
        "\nNote the class imbalance. Zero is far more common than the rest "
        "because of leading zeros on meter readings. Check your per-class "
        "recall, not just overall accuracy."
    )


if __name__ == "__main__":
    main()
    
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tensorflow.keras.models import load_model

IMAGE_DIR = Path("data/synthetic/images")
OUTPUT_DIR = Path("error_crops")
OUTPUT_DIR.mkdir(exist_ok=True)

yolo_model = YOLO("weights/best.pt")
cnn_model = load_model("cnn_digit_model.h5")

errors = {
    "meter_01652.jpg": "133805",
    "meter_02084.jpg": "19444",
    "meter_00678.jpg": "696",
}

for filename, expected in errors.items():
    image_path = IMAGE_DIR / filename
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read {image_path}")
        continue

    result = yolo_model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )[0]

    boxes = sorted(
        list(result.boxes),
        key=lambda b: float(b.xyxy[0][0].item())
    )

    crops = []
    positions = []

    for box in boxes:
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        crops.append(resized.astype("float32") / 255.0)
        positions.append((x1, y1, x2, y2))

    if not crops:
        print(f"No valid crops for {filename}")
        continue

    batch = np.array(crops)[..., np.newaxis]
    predictions = cnn_model.predict(batch, verbose=0)

    predicted_digits = []

    print(f"\n{'=' * 55}")
    print(f"Image: {filename}")
    print(f"Expected reading: {expected}")
    print(f"{'=' * 55}")

    for i, (p, coords) in enumerate(zip(predictions, positions)):
        digit = int(np.argmax(p))
        confidence = float(np.max(p))
        predicted_digits.append(str(digit))

        top3 = np.argsort(p)[-3:][::-1]
        alternatives = ", ".join(
            f"{int(d)} ({float(p[d]):.3f})"
            for d in top3
        )

        print(
            f"Position {i + 1}: predicted={digit}, "
            f"CNN confidence={confidence:.3f}, "
            f"alternatives=[{alternatives}], "
            f"box={coords}"
        )

        x1, y1, x2, y2 = coords
        crop = image[y1:y2, x1:x2]

        if crop.size:
            enlarged = cv2.resize(
                crop, (128, 128),
                interpolation=cv2.INTER_CUBIC
            )
            cv2.imwrite(
                str(OUTPUT_DIR / f"{Path(filename).stem}_digit_{i + 1}.png"),
                enlarged
            )

    print("Predicted reading:", "".join(predicted_digits))

print(f"\nDigit crops saved in: {OUTPUT_DIR.resolve()}")

import os
import sys
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# Load API key from the project's .env file
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY was not found in the project .env file.")
    sys.exit(1)

# The actual implementation should be in this project script.
# This wrapper locates it; it does not implement meter reading itself.
print("API key loaded successfully.")
print("Project directory:", PROJECT_DIR)
print("Please verify the original meter-reader implementation before running predictions.")

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ================== CONFIG ==================
YOLO_MODEL_PATH = "weights/best.pt"
CNN_MODEL_PATH = "cnn_digit_model.h5"
IMAGES_DIR = "data/synthetic/images"
META_CSV = "data/synthetic/meta.csv"

CONF_THRESHOLD = 0.25
DUPLICATE_IOU_THRESHOLD = 0.80
CNN_INPUT_SIZE = (32, 32)
CONFIDENCE_FLAG = 0.85
SAMPLE_SIZE = 400
MAX_DIGITS = 6
DEBUG = True
# ============================================

print("Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
cnn_model = load_model(CNN_MODEL_PATH)
print("Models loaded.\n")


def preprocess_crop(crop):
    gray = (
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if len(crop.shape) == 3
        else crop
    )
    resized = cv2.resize(gray, CNN_INPUT_SIZE)
    return resized.astype("float32") / 255.0


def calculate_iou(box_a, box_b):
    """Calculate intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0, ix2 - ix1) * max(0, iy2 - iy1)
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def remove_duplicate_boxes(boxes, iou_threshold=DUPLICATE_IOU_THRESHOLD):
    """
    Greedy non-maximum suppression.
    Keep the highest-confidence box when boxes overlap heavily.
    """
    if boxes is None or len(boxes) == 0:
        return []

    sorted_boxes = sorted(
        list(boxes),
        key=lambda b: float(b.conf[0].item()),
        reverse=True
    )

    kept_boxes = []

    for box in sorted_boxes:
        coords = tuple(
            map(int, box.xyxy[0].tolist())
        )

        duplicate = False

        for kept_box in kept_boxes:
            kept_coords = tuple(
                map(int, kept_box.xyxy[0].tolist())
            )

            if calculate_iou(coords, kept_coords) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept_boxes.append(box)

    return kept_boxes


def robust_order(detections):
    """Order detections along the meter's likely reading direction."""
    if len(detections) <= 1:
        return detections

    # Meter digits are normally arranged horizontally.
    # Sorting by horizontal center avoids unnecessary line-fit errors.
    return sorted(detections, key=lambda d: d["cx"])


def normalize_reading(reading: str) -> str:
    """Strip leading zeros while keeping at least one digit."""
    if not reading or reading == "NO_DETECTION":
        return reading

    cleaned = reading.lstrip("0")
    return cleaned if cleaned else "0"


def apply_length_constraint(detections, max_digits=MAX_DIGITS):
    """
    Retain the highest-confidence detections if too many remain.
    This is a fallback, not a substitute for duplicate removal.
    """
    if len(detections) <= max_digits:
        return detections

    top = sorted(
        detections,
        key=lambda d: d["yolo_conf"],
        reverse=True
    )[:max_digits]

    return sorted(top, key=lambda d: d["cx"])


def length_penalty(num_digits: int) -> float:
    if 1 <= num_digits <= MAX_DIGITS:
        return 1.0
    if num_digits == 0:
        return 0.1
    return 0.3


def read_meter(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        if DEBUG:
            print(f"\n[ERROR] Could not read image: {image_path}")
        return "NO_DETECTION", 0.0

    # 1. Detect digit boxes
    results = yolo_model.predict(
        source=img,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        if DEBUG:
            print(f"\n[WARNING] No digit boxes detected: {image_path}")
        return "NO_DETECTION", 0.0

    original_count = len(boxes)

    # 2. Remove duplicate boxes before CNN classification
    kept_boxes = remove_duplicate_boxes(boxes)

    if DEBUG:
        print("\n" + "=" * 65)
        print(f"Image: {image_path}")
        print(f"YOLO boxes before deduplication: {original_count}")
        print(f"YOLO boxes after deduplication: {len(kept_boxes)}")
        print(f"Duplicate boxes removed: {original_count - len(kept_boxes)}")
        print("-" * 65)

    # 3. Crop detected digits
    height, width = img.shape[:2]
    crops = []
    meta = []

    for box in kept_boxes:
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crops.append(preprocess_crop(crop))

        meta.append({
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "yolo_conf": float(box.conf[0].item())
        })

    if not crops:
        return "NO_DETECTION", 0.0

    # 4. Classify digit crops with CNN
    batch = np.array(crops)[..., np.newaxis]
    preds = cnn_model.predict(batch, verbose=0)

    detections = []

    for i, p in enumerate(preds):
        detections.append({
            "digit": int(np.argmax(p)),
            "conf": float(np.max(p)),
            "cx": meta[i]["cx"],
            "cy": meta[i]["cy"],
            "x1": meta[i]["x1"],
            "y1": meta[i]["y1"],
            "x2": meta[i]["x2"],
            "y2": meta[i]["y2"],
            "yolo_conf": meta[i]["yolo_conf"]
        })

    # 5. Print diagnostic predictions
    if DEBUG:
        print(f"Digits classified: {len(detections)}")

        for i, d in enumerate(detections):
            print(
                f"Box {i + 1:02d}: "
                f"digit={d['digit']} | "
                f"YOLO_conf={d['yolo_conf']:.3f} | "
                f"CNN_conf={d['conf']:.3f} | "
                f"box=({d['x1']},{d['y1']},"
                f"{d['x2']},{d['y2']})"
            )

    # 6. Order digits left to right
    detections = robust_order(detections)

    if DEBUG:
        before_limit = "".join(
            str(d["digit"]) for d in detections
        )
        print(f"Reading before length limit: {before_limit}")

    # 7. Apply maximum-length fallback
    detections = apply_length_constraint(
        detections, MAX_DIGITS
    )

    reading = "".join(
        str(d["digit"]) for d in detections
    )

    # 8. Calculate confidence score
    if detections:
        conf_product = float(
            np.prod([d["conf"] for d in detections])
        )
        conf_product *= length_penalty(len(detections))
    else:
        conf_product = 0.0

    if DEBUG:
        print(f"Digits retained: {len(detections)}")
        print(f"Final reading: {reading}")
        print(f"Final confidence: {conf_product:.4f}")
        print("=" * 65)

    return reading, conf_product


def evaluate():
    meta = pd.read_csv(META_CSV)
    print("meta.csv columns:", meta.columns.tolist())

    img_col = reading_col = diff_col = None

    for c in meta.columns:
        cl = c.lower().strip()

        if img_col is None and any(
            k in cl for k in ["image", "file", "name", "path"]
        ):
            img_col = c

        if reading_col is None and any(
            k in cl for k in ["read", "gt", "true", "label", "value"]
        ):
            reading_col = c

        if diff_col is None and any(
            k in cl for k in ["diff", "hard", "tag", "level", "type", "class"]
        ):
            diff_col = c

    if img_col is None:
        img_col = meta.columns[0]

    if reading_col is None:
        reading_col = (
            meta.columns[1] if len(meta.columns) > 1
            else meta.columns[0]
        )

    if diff_col is None:
        meta["difficulty"] = "unknown"
        diff_col = "difficulty"
    else:
        meta[diff_col] = (
            meta[diff_col].fillna("unknown").astype(str)
        )

    print(
        f"Using -> image: '{img_col}', "
        f"reading: '{reading_col}', "
        f"difficulty: '{diff_col}'\n"
    )

    meta["stem"] = meta[img_col].astype(str).apply(
        lambda x: Path(x).stem
    )

    # Stratified sample
    if meta[diff_col].nunique() > 1:
        sample = (
            meta.groupby(diff_col, group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(len(x), SAMPLE_SIZE // 3 + 30),
                    random_state=42
                )
            )
            .reset_index(drop=True)
        )
    else:
        sample = meta.sample(
            n=min(len(meta), SAMPLE_SIZE),
            random_state=42
        )

    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(
            n=SAMPLE_SIZE,
            random_state=42
        )

    results = []
    print(f"Evaluating {len(sample)} images...\n")

    # Avoid printing every box during the full evaluation
    global DEBUG
    previous_debug = DEBUG
    DEBUG = False

    try:
        for _, row in tqdm(
            sample.iterrows(),
            total=len(sample),
            desc="Progress"
        ):
            stem = row["stem"]
            img_path = Path(IMAGES_DIR) / f"{stem}.jpg"

            if not img_path.exists():
                continue

            pred_reading, pred_conf = read_meter(img_path)

            true_raw = str(row[reading_col])
            true_norm = normalize_reading(true_raw)
            pred_norm = normalize_reading(pred_reading)

            results.append({
                "image": img_path.name,
                "true_raw": true_raw,
                "true_norm": true_norm,
                "pred_raw": pred_reading,
                "pred_norm": pred_norm,
                "conf": round(pred_conf, 4),
                "correct": pred_norm == true_norm,
                "difficulty": row.get(diff_col, "unknown"),
                "flagged": pred_conf < CONFIDENCE_FLAG
            })
    finally:
        DEBUG = previous_debug

    df = pd.DataFrame(results)
    df.to_csv("exact_reading_results.csv", index=False)

    print("\nSaved -> exact_reading_results.csv")

    if df.empty:
        print("No images were evaluated. Check the dataset paths.")
        return

    overall = df["correct"].mean() * 100

    print(f"\n{'=' * 50}")
    print(f"OVERALL Exact Reading Accuracy: {overall:.2f}%")
    print(f"{'=' * 50}")

    print("\nBreakdown by difficulty:")
    for tag, group in df.groupby("difficulty"):
        acc = group["correct"].mean() * 100
        print(f"  {str(tag):12s}: {acc:6.2f}% (n={len(group)})")

    wrong = df[~df["correct"]]

    if len(wrong) > 0:
        print(f"\nWrong readings: {len(wrong)}")
        print(f"  Flagged: {wrong['flagged'].sum()}")
        print(f"  Silent: {(~wrong['flagged']).sum()}")

        print("\nSample of remaining errors:")
        print(
            wrong[
                ["image", "true_norm", "pred_norm", "conf"]
            ].head(10).to_string(index=False)
        )
    else:
        print("\nNo wrong readings in this sample.")


if __name__ == "__main__":
    # Test one image first with diagnostic output
    test = "data/synthetic/images/meter_02746.jpg"
    r, c = read_meter(test)

    print(f"\nTest image: {test}")
    print(f"Reading: {r}")
    print(f"Confidence: {c:.4f}\n")

    # Then evaluate the sample
    evaluate()

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ================== CONFIG ==================
YOLO_MODEL_PATH = "weights/best.pt"
CNN_MODEL_PATH = "cnn_digit_model.h5"
IMAGES_DIR = "data/synthetic/images"
META_CSV = "data/synthetic/meta.csv"

CONF_THRESHOLD = 0.25
DUPLICATE_IOU_THRESHOLD = 0.80
CNN_INPUT_SIZE = (32, 32)
CONFIDENCE_FLAG = 0.85
SAMPLE_SIZE = 400
MAX_DIGITS = 6
DEBUG = True
# ============================================

print("Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
cnn_model = load_model(CNN_MODEL_PATH)
print("Models loaded.\n")


def preprocess_crop(crop):
    gray = (
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if len(crop.shape) == 3
        else crop
    )
    resized = cv2.resize(gray, CNN_INPUT_SIZE)
    return resized.astype("float32") / 255.0


def calculate_iou(box_a, box_b):
    """Calculate intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0, ix2 - ix1) * max(0, iy2 - iy1)
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def remove_duplicate_boxes(boxes, iou_threshold=DUPLICATE_IOU_THRESHOLD):
    """
    Greedy non-maximum suppression.
    Keep the highest-confidence box when boxes overlap heavily.
    """
    if boxes is None or len(boxes) == 0:
        return []

    sorted_boxes = sorted(
        list(boxes),
        key=lambda b: float(b.conf[0].item()),
        reverse=True
    )

    kept_boxes = []

    for box in sorted_boxes:
        coords = tuple(
            map(int, box.xyxy[0].tolist())
        )

        duplicate = False

        for kept_box in kept_boxes:
            kept_coords = tuple(
                map(int, kept_box.xyxy[0].tolist())
            )

            if calculate_iou(coords, kept_coords) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept_boxes.append(box)

    return kept_boxes


def robust_order(detections):
    """Order detections along the meter's likely reading direction."""
    if len(detections) <= 1:
        return detections

    # Meter digits are normally arranged horizontally.
    # Sorting by horizontal center avoids unnecessary line-fit errors.
    return sorted(detections, key=lambda d: d["cx"])


def normalize_reading(reading: str) -> str:
    """Strip leading zeros while keeping at least one digit."""
    if not reading or reading == "NO_DETECTION":
        return reading

    cleaned = reading.lstrip("0")
    return cleaned if cleaned else "0"


def apply_length_constraint(detections, max_digits=MAX_DIGITS):
    """
    Retain the highest-confidence detections if too many remain.
    This is a fallback, not a substitute for duplicate removal.
    """
    if len(detections) <= max_digits:
        return detections

    top = sorted(
        detections,
        key=lambda d: d["yolo_conf"],
        reverse=True
    )[:max_digits]

    return sorted(top, key=lambda d: d["cx"])


def length_penalty(num_digits: int) -> float:
    if 1 <= num_digits <= MAX_DIGITS:
        return 1.0
    if num_digits == 0:
        return 0.1
    return 0.3


def read_meter(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        if DEBUG:
            print(f"\n[ERROR] Could not read image: {image_path}")
        return "NO_DETECTION", 0.0

    # 1. Detect digit boxes
    results = yolo_model.predict(
        source=img,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        if DEBUG:
            print(f"\n[WARNING] No digit boxes detected: {image_path}")
        return "NO_DETECTION", 0.0

    original_count = len(boxes)

    # 2. Remove duplicate boxes before CNN classification
    kept_boxes = remove_duplicate_boxes(boxes)

    if DEBUG:
        print("\n" + "=" * 65)
        print(f"Image: {image_path}")
        print(f"YOLO boxes before deduplication: {original_count}")
        print(f"YOLO boxes after deduplication: {len(kept_boxes)}")
        print(f"Duplicate boxes removed: {original_count - len(kept_boxes)}")
        print("-" * 65)

    # 3. Crop detected digits
    height, width = img.shape[:2]
    crops = []
    meta = []

    for box in kept_boxes:
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crops.append(preprocess_crop(crop))

        meta.append({
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "yolo_conf": float(box.conf[0].item())
        })

    if not crops:
        return "NO_DETECTION", 0.0

    # 4. Classify digit crops with CNN
    batch = np.array(crops)[..., np.newaxis]
    preds = cnn_model.predict(batch, verbose=0)

    detections = []

    for i, p in enumerate(preds):
        detections.append({
            "digit": int(np.argmax(p)),
            "conf": float(np.max(p)),
            "cx": meta[i]["cx"],
            "cy": meta[i]["cy"],
            "x1": meta[i]["x1"],
            "y1": meta[i]["y1"],
            "x2": meta[i]["x2"],
            "y2": meta[i]["y2"],
            "yolo_conf": meta[i]["yolo_conf"]
        })

    # 5. Print diagnostic predictions
    if DEBUG:
        print(f"Digits classified: {len(detections)}")

        for i, d in enumerate(detections):
            print(
                f"Box {i + 1:02d}: "
                f"digit={d['digit']} | "
                f"YOLO_conf={d['yolo_conf']:.3f} | "
                f"CNN_conf={d['conf']:.3f} | "
                f"box=({d['x1']},{d['y1']},"
                f"{d['x2']},{d['y2']})"
            )

    # 6. Order digits left to right
    detections = robust_order(detections)

    if DEBUG:
        before_limit = "".join(
            str(d["digit"]) for d in detections
        )
        print(f"Reading before length limit: {before_limit}")

    # 7. Apply maximum-length fallback
    detections = apply_length_constraint(
        detections, MAX_DIGITS
    )

    reading = "".join(
        str(d["digit"]) for d in detections
    )

    # 8. Calculate confidence score
    if detections:
        conf_product = float(
            np.prod([d["conf"] for d in detections])
        )
        conf_product *= length_penalty(len(detections))
    else:
        conf_product = 0.0

    if DEBUG:
        print(f"Digits retained: {len(detections)}")
        print(f"Final reading: {reading}")
        print(f"Final confidence: {conf_product:.4f}")
        print("=" * 65)

    return reading, conf_product


def evaluate():
    meta = pd.read_csv(META_CSV)
    print("meta.csv columns:", meta.columns.tolist())

    img_col = reading_col = diff_col = None

    for c in meta.columns:
        cl = c.lower().strip()

        if img_col is None and any(
            k in cl for k in ["image", "file", "name", "path"]
        ):
            img_col = c

        if reading_col is None and any(
            k in cl for k in ["read", "gt", "true", "label", "value"]
        ):
            reading_col = c

        if diff_col is None and any(
            k in cl for k in ["diff", "hard", "tag", "level", "type", "class"]
        ):
            diff_col = c

    if img_col is None:
        img_col = meta.columns[0]

    if reading_col is None:
        reading_col = (
            meta.columns[1] if len(meta.columns) > 1
            else meta.columns[0]
        )

    if diff_col is None:
        meta["difficulty"] = "unknown"
        diff_col = "difficulty"
    else:
        meta[diff_col] = (
            meta[diff_col].fillna("unknown").astype(str)
        )

    print(
        f"Using -> image: '{img_col}', "
        f"reading: '{reading_col}', "
        f"difficulty: '{diff_col}'\n"
    )

    meta["stem"] = meta[img_col].astype(str).apply(
        lambda x: Path(x).stem
    )

    # Stratified sample
    if meta[diff_col].nunique() > 1:
        sample = (
            meta.groupby(diff_col, group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(len(x), SAMPLE_SIZE // 3 + 30),
                    random_state=42
                )
            )
            .reset_index(drop=True)
        )
    else:
        sample = meta.sample(
            n=min(len(meta), SAMPLE_SIZE),
            random_state=42
        )

    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(
            n=SAMPLE_SIZE,
            random_state=42
        )

    results = []
    print(f"Evaluating {len(sample)} images...\n")

    # Avoid printing every box during the full evaluation
    global DEBUG
    previous_debug = DEBUG
    DEBUG = False

    try:
        for _, row in tqdm(
            sample.iterrows(),
            total=len(sample),
            desc="Progress"
        ):
            stem = row["stem"]
            img_path = Path(IMAGES_DIR) / f"{stem}.jpg"

            if not img_path.exists():
                continue

            pred_reading, pred_conf = read_meter(img_path)

            true_raw = str(row[reading_col])
            true_norm = normalize_reading(true_raw)
            pred_norm = normalize_reading(pred_reading)

            results.append({
                "image": img_path.name,
                "true_raw": true_raw,
                "true_norm": true_norm,
                "pred_raw": pred_reading,
                "pred_norm": pred_norm,
                "conf": round(pred_conf, 4),
                "correct": pred_norm == true_norm,
                "difficulty": row.get(diff_col, "unknown"),
                "flagged": pred_conf < CONFIDENCE_FLAG
            })
    finally:
        DEBUG = previous_debug

    df = pd.DataFrame(results)
    df.to_csv("exact_reading_results.csv", index=False)

    print("\nSaved -> exact_reading_results.csv")

    if df.empty:
        print("No images were evaluated. Check the dataset paths.")
        return

    overall = df["correct"].mean() * 100

    print(f"\n{'=' * 50}")
    print(f"OVERALL Exact Reading Accuracy: {overall:.2f}%")
    print(f"{'=' * 50}")

    print("\nBreakdown by difficulty:")
    for tag, group in df.groupby("difficulty"):
        acc = group["correct"].mean() * 100
        print(f"  {str(tag):12s}: {acc:6.2f}% (n={len(group)})")

    wrong = df[~df["correct"]]

    if len(wrong) > 0:
        print(f"\nWrong readings: {len(wrong)}")
        print(f"  Flagged: {wrong['flagged'].sum()}")
        print(f"  Silent: {(~wrong['flagged']).sum()}")

        print("\nSample of remaining errors:")
        print(
            wrong[
                ["image", "true_norm", "pred_norm", "conf"]
            ].head(10).to_string(index=False)
        )
    else:
        print("\nNo wrong readings in this sample.")


if __name__ == "__main__":
    # Test one image first with diagnostic output
    test = "data/synthetic/images/meter_02746.jpg"
    r, c = read_meter(test)

    print(f"\nTest image: {test}")
    print(f"Reading: {r}")
    print(f"Confidence: {c:.4f}\n")

    # Then evaluate the sample
    evaluate()

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ================== CONFIG ==================
YOLO_MODEL_PATH = "weights/best.pt"
CNN_MODEL_PATH = "cnn_digit_model.h5"
IMAGES_DIR = "data/synthetic/images"
META_CSV = "data/synthetic/meta.csv"

CONF_THRESHOLD = 0.25
DUPLICATE_IOU_THRESHOLD = 0.80
CNN_INPUT_SIZE = (32, 32)
CONFIDENCE_FLAG = 0.85
SAMPLE_SIZE = 400
MAX_DIGITS = 6
DEBUG = True
# ============================================

print("Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
cnn_model = load_model(CNN_MODEL_PATH)
print("Models loaded.\n")


def preprocess_crop(crop):
    gray = (
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if len(crop.shape) == 3
        else crop
    )
    resized = cv2.resize(gray, CNN_INPUT_SIZE)
    return resized.astype("float32") / 255.0


def calculate_iou(box_a, box_b):
    """Calculate intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0, ix2 - ix1) * max(0, iy2 - iy1)
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def remove_duplicate_boxes(boxes, iou_threshold=DUPLICATE_IOU_THRESHOLD):
    """
    Greedy non-maximum suppression.
    Keep the highest-confidence box when boxes overlap heavily.
    """
    if boxes is None or len(boxes) == 0:
        return []

    sorted_boxes = sorted(
        list(boxes),
        key=lambda b: float(b.conf[0].item()),
        reverse=True
    )

    kept_boxes = []

    for box in sorted_boxes:
        coords = tuple(
            map(int, box.xyxy[0].tolist())
        )

        duplicate = False

        for kept_box in kept_boxes:
            kept_coords = tuple(
                map(int, kept_box.xyxy[0].tolist())
            )

            if calculate_iou(coords, kept_coords) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept_boxes.append(box)

    return kept_boxes


def robust_order(detections):
    """Order detections along the meter's likely reading direction."""
    if len(detections) <= 1:
        return detections

    # Meter digits are normally arranged horizontally.
    # Sorting by horizontal center avoids unnecessary line-fit errors.
    return sorted(detections, key=lambda d: d["cx"])


def normalize_reading(reading: str) -> str:
    """Strip leading zeros while keeping at least one digit."""
    if not reading or reading == "NO_DETECTION":
        return reading

    cleaned = reading.lstrip("0")
    return cleaned if cleaned else "0"


def apply_length_constraint(detections, max_digits=MAX_DIGITS):
    """
    Retain the highest-confidence detections if too many remain.
    This is a fallback, not a substitute for duplicate removal.
    """
    if len(detections) <= max_digits:
        return detections

    top = sorted(
        detections,
        key=lambda d: d["yolo_conf"],
        reverse=True
    )[:max_digits]

    return sorted(top, key=lambda d: d["cx"])


def length_penalty(num_digits: int) -> float:
    if 1 <= num_digits <= MAX_DIGITS:
        return 1.0
    if num_digits == 0:
        return 0.1
    return 0.3


def read_meter(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        if DEBUG:
            print(f"\n[ERROR] Could not read image: {image_path}")
        return "NO_DETECTION", 0.0

    # 1. Detect digit boxes
    results = yolo_model.predict(
        source=img,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        if DEBUG:
            print(f"\n[WARNING] No digit boxes detected: {image_path}")
        return "NO_DETECTION", 0.0

    original_count = len(boxes)

    # 2. Remove duplicate boxes before CNN classification
    kept_boxes = remove_duplicate_boxes(boxes)

    if DEBUG:
        print("\n" + "=" * 65)
        print(f"Image: {image_path}")
        print(f"YOLO boxes before deduplication: {original_count}")
        print(f"YOLO boxes after deduplication: {len(kept_boxes)}")
        print(f"Duplicate boxes removed: {original_count - len(kept_boxes)}")
        print("-" * 65)

    # 3. Crop detected digits
    height, width = img.shape[:2]
    crops = []
    meta = []

    for box in kept_boxes:
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crops.append(preprocess_crop(crop))

        meta.append({
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "yolo_conf": float(box.conf[0].item())
        })

    if not crops:
        return "NO_DETECTION", 0.0

    # 4. Classify digit crops with CNN
    batch = np.array(crops)[..., np.newaxis]
    preds = cnn_model.predict(batch, verbose=0)

    detections = []

    for i, p in enumerate(preds):
        detections.append({
            "digit": int(np.argmax(p)),
            "conf": float(np.max(p)),
            "cx": meta[i]["cx"],
            "cy": meta[i]["cy"],
            "x1": meta[i]["x1"],
            "y1": meta[i]["y1"],
            "x2": meta[i]["x2"],
            "y2": meta[i]["y2"],
            "yolo_conf": meta[i]["yolo_conf"]
        })

    # 5. Print diagnostic predictions
    if DEBUG:
        print(f"Digits classified: {len(detections)}")

        for i, d in enumerate(detections):
            print(
                f"Box {i + 1:02d}: "
                f"digit={d['digit']} | "
                f"YOLO_conf={d['yolo_conf']:.3f} | "
                f"CNN_conf={d['conf']:.3f} | "
                f"box=({d['x1']},{d['y1']},"
                f"{d['x2']},{d['y2']})"
            )

    # 6. Order digits left to right
    detections = robust_order(detections)

    if DEBUG:
        before_limit = "".join(
            str(d["digit"]) for d in detections
        )
        print(f"Reading before length limit: {before_limit}")

    # 7. Apply maximum-length fallback
    detections = apply_length_constraint(
        detections, MAX_DIGITS
    )

    reading = "".join(
        str(d["digit"]) for d in detections
    )

    # 8. Calculate confidence score
    if detections:
        conf_product = float(
            np.prod([d["conf"] for d in detections])
        )
        conf_product *= length_penalty(len(detections))
    else:
        conf_product = 0.0

    if DEBUG:
        print(f"Digits retained: {len(detections)}")
        print(f"Final reading: {reading}")
        print(f"Final confidence: {conf_product:.4f}")
        print("=" * 65)

    return reading, conf_product


def evaluate():
    meta = pd.read_csv(META_CSV)
    print("meta.csv columns:", meta.columns.tolist())

    img_col = reading_col = diff_col = None

    for c in meta.columns:
        cl = c.lower().strip()

        if img_col is None and any(
            k in cl for k in ["image", "file", "name", "path"]
        ):
            img_col = c

        if reading_col is None and any(
            k in cl for k in ["read", "gt", "true", "label", "value"]
        ):
            reading_col = c

        if diff_col is None and any(
            k in cl for k in ["diff", "hard", "tag", "level", "type", "class"]
        ):
            diff_col = c

    if img_col is None:
        img_col = meta.columns[0]

    if reading_col is None:
        reading_col = (
            meta.columns[1] if len(meta.columns) > 1
            else meta.columns[0]
        )

    if diff_col is None:
        meta["difficulty"] = "unknown"
        diff_col = "difficulty"
    else:
        meta[diff_col] = (
            meta[diff_col].fillna("unknown").astype(str)
        )

    print(
        f"Using -> image: '{img_col}', "
        f"reading: '{reading_col}', "
        f"difficulty: '{diff_col}'\n"
    )

    meta["stem"] = meta[img_col].astype(str).apply(
        lambda x: Path(x).stem
    )

    # Stratified sample
    if meta[diff_col].nunique() > 1:
        sample = (
            meta.groupby(diff_col, group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(len(x), SAMPLE_SIZE // 3 + 30),
                    random_state=42
                )
            )
            .reset_index(drop=True)
        )
    else:
        sample = meta.sample(
            n=min(len(meta), SAMPLE_SIZE),
            random_state=42
        )

    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(
            n=SAMPLE_SIZE,
            random_state=42
        )

    results = []
    print(f"Evaluating {len(sample)} images...\n")

    # Avoid printing every box during the full evaluation
    global DEBUG
    previous_debug = DEBUG
    DEBUG = False

    try:
        for _, row in tqdm(
            sample.iterrows(),
            total=len(sample),
            desc="Progress"
        ):
            stem = row["stem"]
            img_path = Path(IMAGES_DIR) / f"{stem}.jpg"

            if not img_path.exists():
                continue

            pred_reading, pred_conf = read_meter(img_path)

            true_raw = str(row[reading_col])
            true_norm = normalize_reading(true_raw)
            pred_norm = normalize_reading(pred_reading)

            results.append({
                "image": img_path.name,
                "true_raw": true_raw,
                "true_norm": true_norm,
                "pred_raw": pred_reading,
                "pred_norm": pred_norm,
                "conf": round(pred_conf, 4),
                "correct": pred_norm == true_norm,
                "difficulty": row.get(diff_col, "unknown"),
                "flagged": pred_conf < CONFIDENCE_FLAG
            })
    finally:
        DEBUG = previous_debug

    df = pd.DataFrame(results)
    df.to_csv("exact_reading_results.csv", index=False)

    print("\nSaved -> exact_reading_results.csv")

    if df.empty:
        print("No images were evaluated. Check the dataset paths.")
        return

    overall = df["correct"].mean() * 100

    print(f"\n{'=' * 50}")
    print(f"OVERALL Exact Reading Accuracy: {overall:.2f}%")
    print(f"{'=' * 50}")

    print("\nBreakdown by difficulty:")
    for tag, group in df.groupby("difficulty"):
        acc = group["correct"].mean() * 100
        print(f"  {str(tag):12s}: {acc:6.2f}% (n={len(group)})")

    wrong = df[~df["correct"]]

    if len(wrong) > 0:
        print(f"\nWrong readings: {len(wrong)}")
        print(f"  Flagged: {wrong['flagged'].sum()}")
        print(f"  Silent: {(~wrong['flagged']).sum()}")

        print("\nSample of remaining errors:")
        print(
            wrong[
                ["image", "true_norm", "pred_norm", "conf"]
            ].head(10).to_string(index=False)
        )
    else:
        print("\nNo wrong readings in this sample.")


if __name__ == "__main__":
    # Test one image first with diagnostic output
    test = "data/synthetic/images/meter_02746.jpg"
    r, c = read_meter(test)

    print(f"\nTest image: {test}")
    print(f"Reading: {r}")
    print(f"Confidence: {c:.4f}\n")

    # Then evaluate the sample
    evaluate()
python read_meter.py# Create the file
@"
from ultralytics import YOLO
from pathlib import Path
import cv2

# ================== CONFIG ==================
MODEL_PATH = "weights/best.pt"          # your trained model
IMAGE_PATH = "data/synthetic/images/meter_00000.jpg"  # change this
CONF_THRESHOLD = 0.40                   # adjust if needed
# ============================================

def read_meter(image_path: str, model_path: str = MODEL_PATH, conf: float = CONF_THRESHOLD):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False)[0]

    # Extract boxes + class ids + confidences
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return "No digits detected", None

    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "digit": cls_id,
            "conf": conf_score,
            "x_center": (x1 + x2) / 2,
            "box": (x1, y1, x2, y2)
        })

    # Sort left → right
    detections = sorted(detections, key=lambda d: d["x_center"])

    # Build the reading
    reading = "".join(str(d["digit"]) for d in detections)

    return reading, detections


if __name__ == "__main__":
    image = IMAGE_PATH
    reading, dets = read_meter(image)

    print("=" * 50)
    print(f"Image   : {image}")
    print(f"Reading : {reading}")
    print("=" * 50)

    if dets:
        print(f"{'Digit':<8} {'Conf':<8} {'X-center'}")
        print("-" * 30)
        for d in dets:
            print(f"{d['digit']:<8} {d['conf']:.3f}    {d['x_center']:.1f}")
"@ | Out-File -FilePath "read_meter.py" -Encoding utf8
"""
Show genuine incorrect meter readings from exact_reading_results.csv.

Run from the project root:
    python show_failure.py

Controls:
    N / Right Arrow : next failure
    P / Left Arrow  : previous failure
    Q / Esc         : quit

The script does not invent predictions. It reads the true and predicted
readings already recorded in exact_reading_results.csv.
"""

import csv
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = PROJECT_DIR / "exact_reading_results.csv"

IMAGE_DIRS = [
    PROJECT_DIR / "data" / "synthetic" / "images",
    PROJECT_DIR / "data" / "real_test" / "images",
    PROJECT_DIR / "data" / "images",
]

WINDOW_NAME = "Meter Digitizer - Genuine Failure Cases"


def clean(value):
    """Convert CSV values to clean strings."""
    if value is None:
        return ""
    return str(value).strip()


def find_image(image_value):
    """Find an image using its recorded path or filename."""
    raw = Path(image_value)
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(PROJECT_DIR / raw)
        candidates.append(PROJECT_DIR / "data" / raw)

    # Search known image folders using the filename from the CSV.
    for folder in IMAGE_DIRS:
        candidates.append(folder / raw.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Fallback: search within data/ for the exact filename.
    data_dir = PROJECT_DIR / "data"
    if data_dir.exists():
        matches = list(data_dir.rglob(raw.name))
        for match in matches:
            if match.is_file() and match.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }:
                return match

    return None


def load_failures():
    if not RESULTS_CSV.is_file():
        raise FileNotFoundError(
            f"Could not find {RESULTS_CSV}\n"
            "Run your evaluation first so exact_reading_results.csv exists."
        )

    failures = []
    with RESULTS_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"image", "true_norm", "pred_norm"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain these columns: image, true_norm, pred_norm. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            true_reading = clean(row.get("true_norm")) or clean(row.get("true_raw"))
            predicted_reading = clean(row.get("pred_norm")) or clean(row.get("pred_raw"))

            # Compare normalized readings, so we only display real mismatches.
            if true_reading and predicted_reading and true_reading != predicted_reading:
                row["_true"] = true_reading
                row["_pred"] = predicted_reading
                row["_image_path"] = find_image(clean(row.get("image")))
                failures.append(row)

    return failures


def draw_wrapped_text(image, text, x, y, font_scale=0.7, thickness=2,
                      color=(255, 255, 255), max_width=900):
    """Draw text with a dark background for better recording visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()
        width = cv2.getTextSize(test, font, font_scale, thickness)[0][0]
        if width > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    line_height = int(30 * font_scale + 18)
    for line in lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            line, font, font_scale, thickness
        )
        cv2.rectangle(
            image,
            (x - 8, y - text_height - 8),
            (x + text_width + 8, y + baseline + 8),
            (25, 25, 25),
            -1,
        )
        cv2.putText(
            image, line, (x, y), font, font_scale, color, thickness,
            cv2.LINE_AA
        )
        y += line_height
    return y


def make_display(image_path, row, index, total):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    # Resize only for display; keep the original aspect ratio.
    max_width, max_height = 1200, 850
    scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.8)
    if scale != 1:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    # Add space above the image so labels do not hide the meter digits.
    header_height = 180
    canvas = cv2.copyMakeBorder(
        image, header_height, 0, 0, 0,
        cv2.BORDER_CONSTANT, value=(35, 35, 35)
    )

    y = 38
    y = draw_wrapped_text(
        canvas, f"FAILURE CASE {index + 1} OF {total}", 20, y,
        font_scale=0.9, thickness=2, color=(80, 170, 255)
    )
    y = draw_wrapped_text(
        canvas, f"Correct reading: {row['_true']}", 20, y,
        font_scale=0.75, thickness=2, color=(100, 255, 100)
    )
    y = draw_wrapped_text(
        canvas, f"Model prediction: {row['_pred']}", 20, y,
        font_scale=0.75, thickness=2, color=(100, 100, 255)
    )

    confidence = clean(row.get("conf"))
    flagged = clean(row.get("flagged"))
    extra = f"Confidence estimate: {confidence or 'not recorded'}"
    if flagged:
        extra += f" | Flagged for review: {flagged}"
    draw_wrapped_text(
        canvas, extra, 20, y, font_scale=0.58, thickness=1,
        color=(230, 230, 230)
    )

    return canvas


def main():
    failures = load_failures()

    if not failures:
        print("No mismatched readings were found in exact_reading_results.csv.")
        print("Check that the CSV contains evaluation predictions and true labels.")
        return

    print(f"Found {len(failures)} genuine failure case(s).")
    print("Use N/right arrow for next, P/left arrow for previous, Q/Esc to quit.")
    print("For the recording, keep the failure image and this console visible.")

    index = 0
    while True:
        row = failures[index]
        image_path = row.get("_image_path")

        if image_path is None:
            print(
                f"\nImage file not found for CSV entry: {clean(row.get('image'))}\n"
                "Expected it under data/synthetic/images or data/real_test/images."
            )
            print(
                f"Correct reading: {row['_true']} | "
                f"Model prediction: {row['_pred']}"
            )
            key = cv2.waitKey(0) & 0xFF
        else:
            print(
                f"\n[{index + 1}/{len(failures)}] {clean(row.get('image'))} | "
                f"Correct: {row['_true']} | Predicted: {row['_pred']}"
            )
            display = make_display(image_path, row, index, len(failures))
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("n"), ord("N"), 83, 3):  # next / right arrow
            index = (index + 1) % len(failures)
        elif key in (ord("p"), ord("P"), 81, 2):  # previous / left arrow
            index = (index - 1) % len(failures)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models

# ====================== 1. LOAD DATA ======================
print("Loading digit crops...")

X = []
y = []

crop_dir = "data/synthetic/digit_crops"

for digit in range(10):
    folder = os.path.join(crop_dir, str(digit))
    files = os.listdir(folder)
    
    for f in files:
        img_path = os.path.join(folder, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
            
        # Resize to 32x32 (as required for ANN)
        img = cv2.resize(img, (32, 32))
        X.append(img)
        y.append(digit)

X = np.array(X, dtype=np.float32) / 255.0   # normalize
y = np.array(y)

print(f"Total samples: {len(X)}")
print("Class distribution:", np.bincount(y))

# ====================== 2. TRAIN / TEST SPLIT ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ====================== 3. DENSE ANN BASELINE ======================
print("\n===== Training Dense ANN =====")

# Flatten for ANN
X_train_flat = X_train.reshape(-1, 32*32)
X_test_flat  = X_test.reshape(-1, 32*32)

ann = models.Sequential([
    layers.Input(shape=(1024,)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

ann.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

ann.summary()

history_ann = ann.fit(
    X_train_flat, y_train,
    validation_data=(X_test_flat, y_test),
    epochs=15,
    batch_size=64,
    verbose=1
)

# Evaluate ANN
y_pred_ann = np.argmax(ann.predict(X_test_flat), axis=1)

print("\n===== ANN Classification Report =====")
print(classification_report(y_test, y_pred_ann, digits=4))

# Confusion Matrix - ANN
cm_ann = confusion_matrix(y_test, y_pred_ann)
plt.figure(figsize=(10,8))
sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues')
plt.title("ANN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("ann_confusion_matrix.png")
plt.close()
print("Saved: ann_confusion_matrix.png")

# ====================== 4. CUSTOM CNN ======================
print("\n===== Training Custom CNN =====")

# CNN needs channel dimension
X_train_cnn = X_train[..., np.newaxis]   # (N, 32, 32, 1)
X_test_cnn  = X_test[..., np.newaxis]

cnn = models.Sequential([
    layers.Input(shape=(32, 32, 1)),
    
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn.summary()

history_cnn = cnn.fit(
    X_train_cnn, y_train,
    validation_data=(X_test_cnn, y_test),
    epochs=15,               # same as ANN
    batch_size=64,           # same as ANN
    verbose=1
)

# Evaluate CNN
y_pred_cnn = np.argmax(cnn.predict(X_test_cnn), axis=1)

print("\n===== CNN Classification Report =====")
print(classification_report(y_test, y_pred_cnn, digits=4))

# Confusion Matrix - CNN
cm_cnn = confusion_matrix(y_test, y_pred_cnn)
plt.figure(figsize=(10,8))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens')
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("cnn_confusion_matrix.png")
plt.close()
print("Saved: cnn_confusion_matrix.png")

# ====================== 5. SAVE MODELS ======================
ann.save("ann_digit_model.h5")
cnn.save("cnn_digit_model.h5")
print("\nModels saved: ann_digit_model.h5 and cnn_digit_model.h5")

print("\n===== DONE =====")
from ultralytics import YOLO

# Load a small YOLOv8 model
model = YOLO("yolov8n.pt")

# Train
results = model.train(
    data="data/synthetic/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="meter_digit_detector",
    project="runs/detect",
    exist_ok=True
)

print("\nTraining finished!")
print("Best model saved at: runs/detect/meter_digit_detector/weights/best.pt")


if __name__ == "__main__":
import os
import cv2
import random
import glob

# Paths
IMG_DIR = "data/synthetic/images"
LBL_DIR = "data/synthetic/labels"
OUT_DIR = "verified_samples"

os.makedirs(OUT_DIR, exist_ok=True)

# Get all image files
image_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))

if not image_files:
    raise FileNotFoundError("No images found in data/synthetic/images/")

# Randomly select 10 images
selected = random.sample(image_files, min(10, len(image_files)))

print(f"Verifying {len(selected)} images...\n")

for img_path in selected:
    name = os.path.basename(img_path)
    label_path = os.path.join(LBL_DIR, name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {name}")
        continue

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"Missing label for {name}")
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, cx, cy, bw, bh = parts
        cls = int(cls)
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)

        # Convert YOLO normalized coordinates to pixel coordinates
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write digit class
        cv2.putText(img, str(cls), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = os.path.join(OUT_DIR, name)
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")

print("\nDone! Check the 'verified_samples' folder.")



    main()
