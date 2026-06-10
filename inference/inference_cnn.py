import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from custom_loss import get_custom_objects
from custom_layer import get_custom_layers


CLASS_NAMES = [f"type_{i}" for i in range(1, 7)]
LABEL_TO_CLASS = {idx: class_name for idx, class_name in enumerate(CLASS_NAMES)}

IMG_SIZE = 224
MIN_FACE_SCORE = 0.90
BBOX_PRIMARY_MARGIN = 0.45

MIN_SKIN_RATIO_FALLBACK = 0.05
MAX_SKIN_RATIO_WARNING = 0.92
MIN_SKIN_PIXELS = 80


def read_image_rgb(image_path):
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return np.array(img)


def resize_rgb(img_rgb, size=224):
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)


def crop_face_bbox(img_rgb, facial_area, margin=0.45, output_size=224):
    h, w = img_rgb.shape[:2]

    x1, y1, x2, y2 = facial_area
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    mx = int(bw * margin)
    my = int(bh * margin)

    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)

    crop = img_rgb[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        return resize_rgb(img_rgb, output_size), "fallback_resize_original"

    return resize_rgb(crop, output_size), "bbox_crop_primary"


def retinaface_crop_or_fallback(img_rgb):
    try:
        from retinaface import RetinaFace

        detections = RetinaFace.detect_faces(img_rgb)

        if not isinstance(detections, dict):
            return resize_rgb(img_rgb, IMG_SIZE), "fallback_resize_original", None

        faces = []

        for key, data in detections.items():
            score = float(data.get("score", 0.0))
            area = data.get("facial_area", None)

            if area is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in area]
            bbox_area = max(0, x2 - x1) * max(0, y2 - y1)

            faces.append({
                "score": score,
                "facial_area": [x1, y1, x2, y2],
                "area": bbox_area
            })

        faces = sorted(faces, key=lambda x: (x["score"], x["area"]), reverse=True)
        valid_faces = [f for f in faces if f["score"] >= MIN_FACE_SCORE]

        if len(valid_faces) == 0:
            return resize_rgb(img_rgb, IMG_SIZE), "fallback_resize_original", None

        best_face = valid_faces[0]

        crop, crop_status = crop_face_bbox(
            img_rgb,
            best_face["facial_area"],
            margin=BBOX_PRIMARY_MARGIN,
            output_size=IMG_SIZE
        )

        return crop, crop_status, best_face

    except Exception:
        return resize_rgb(img_rgb, IMG_SIZE), "fallback_resize_original", None


def illumination_normalize_rgb(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    L_eq = clahe.apply(L)

    lab_eq = cv2.merge([L_eq, A, B])
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return rgb_eq


def rgb_to_lab_scaled(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    L = lab[:, :, 0] * (100.0 / 255.0)
    A = lab[:, :, 1] - 128.0
    B = lab[:, :, 2] - 128.0

    return L, A, B


def rgb_to_hsv_scaled(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    H = hsv[:, :, 0] * 2.0
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0

    return H, S, V


def compute_ita(L, B):
    eps = 1e-6
    ita = np.degrees(np.arctan2((L - 50.0), (B + eps)))
    ita = np.clip(ita, -90, 90)
    return ita


def create_face_ellipse_mask(h, w, scale_x=0.46, scale_y=0.58, center_y=0.52):
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (int(w * 0.50), int(h * center_y))
    axes = (int(w * scale_x), int(h * scale_y))

    cv2.ellipse(
        mask,
        center,
        axes,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1
    )

    return mask > 0


def create_central_face_mask(h, w):
    return create_face_ellipse_mask(
        h=h,
        w=w,
        scale_x=0.28,
        scale_y=0.34,
        center_y=0.52
    )


def generic_skin_mask(img_rgb):
    h, w = img_rgb.shape[:2]

    roi_mask = create_face_ellipse_mask(h, w)

    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y = ycrcb[:, :, 0]
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]

    H, S, V = rgb_to_hsv_scaled(img_rgb)
    L, A, B = rgb_to_lab_scaled(img_rgb)

    ycrcb_cond = (
        (Y > 20) &
        (Cr > 120) & (Cr < 185) &
        (Cb > 65) & (Cb < 155)
    )

    hsv_cond = (
        (V > 0.10) &
        (S > 0.03) & (S < 0.85) &
        ((H < 70) | (H > 330))
    )

    lab_cond = (
        (L > 8) &
        (A > -5) & (A < 40) &
        (B > -5) & (B < 55)
    )

    return roi_mask & (ycrcb_cond | hsv_cond | lab_cond)


def adaptive_skin_mask(img_rgb, base_mask):
    h, w = img_rgb.shape[:2]

    central_mask = create_central_face_mask(h, w)

    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]

    H, S, V = rgb_to_hsv_scaled(img_rgb)
    L, A, B = rgb_to_lab_scaled(img_rgb)

    sample_mask = central_mask & base_mask

    if np.count_nonzero(sample_mask) < 80:
        sample_mask = central_mask & (V > 0.08) & (L > 8)

    if np.count_nonzero(sample_mask) < 80:
        return base_mask

    cr_vals = Cr[sample_mask]
    cb_vals = Cb[sample_mask]
    a_vals = A[sample_mask]
    b_vals = B[sample_mask]

    def robust_range(vals, min_width=12.0, scale=2.8):
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        width = max(min_width, scale * mad)
        return med - width, med + width

    cr_low, cr_high = robust_range(cr_vals, min_width=14.0)
    cb_low, cb_high = robust_range(cb_vals, min_width=14.0)
    a_low, a_high = robust_range(a_vals, min_width=8.0)
    b_low, b_high = robust_range(b_vals, min_width=10.0)

    roi_mask = create_face_ellipse_mask(h, w)

    adaptive = (
        roi_mask &
        (V > 0.06) &
        (L > 5) &
        (Cr >= cr_low) & (Cr <= cr_high) &
        (Cb >= cb_low) & (Cb <= cb_high) &
        (A >= a_low) & (A <= a_high) &
        (B >= b_low) & (B <= b_high)
    )

    return adaptive


def postprocess_mask(mask):
    mask_uint = mask.astype(np.uint8) * 255

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    mask_uint = cv2.morphologyEx(mask_uint, cv2.MORPH_OPEN, kernel_open)
    mask_uint = cv2.morphologyEx(mask_uint, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint, connectivity=8)

    if num_labels <= 1:
        return mask_uint > 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    return labels == largest_label


def create_final_skin_mask(img_rgb):
    h, w = img_rgb.shape[:2]

    base_mask = generic_skin_mask(img_rgb)
    adapt_mask = adaptive_skin_mask(img_rgb, base_mask)

    combined = base_mask | adapt_mask
    combined = postprocess_mask(combined)

    skin_pixel_ratio = np.count_nonzero(combined) / combined.size
    segmentation_status = "ok"

    if skin_pixel_ratio < MIN_SKIN_RATIO_FALLBACK:
        roi_mask = create_face_ellipse_mask(h, w)
        H, S, V = rgb_to_hsv_scaled(img_rgb)
        L, A, B = rgb_to_lab_scaled(img_rgb)

        combined = roi_mask & (V > 0.06) & (L > 5)
        combined = postprocess_mask(combined)

        segmentation_status = "fallback_face_roi"

    skin_pixel_ratio = np.count_nonzero(combined) / combined.size

    if skin_pixel_ratio > MAX_SKIN_RATIO_WARNING:
        segmentation_status = "warning_high_skin_ratio"

    if np.count_nonzero(combined) < MIN_SKIN_PIXELS:
        combined = create_face_ellipse_mask(h, w)
        segmentation_status = "fallback_full_face_ellipse"

    return combined, segmentation_status


def describe_values(values, prefix):
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values))
    }


def extract_color_features(img_rgb, mask):
    feature_img = illumination_normalize_rgb(img_rgb)

    L, A, B = rgb_to_lab_scaled(feature_img)
    H, S, V = rgb_to_hsv_scaled(feature_img)
    ITA = compute_ita(L, B)

    mask_bool = mask.astype(bool)

    features = {}

    features.update(describe_values(L[mask_bool], "lab_l"))
    features.update(describe_values(A[mask_bool], "lab_a"))
    features.update(describe_values(B[mask_bool], "lab_b"))

    features.update(describe_values(H[mask_bool], "hsv_h"))
    features.update(describe_values(S[mask_bool], "hsv_s"))
    features.update(describe_values(V[mask_bool], "hsv_v"))

    features.update(describe_values(ITA[mask_bool], "ita"))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    features["image_mean_brightness"] = float(np.mean(gray))
    features["image_std_brightness"] = float(np.std(gray))
    features["image_min_brightness"] = float(np.min(gray))
    features["image_max_brightness"] = float(np.max(gray))

    features["skin_pixel_count"] = int(np.count_nonzero(mask_bool))
    features["total_pixel_count"] = int(mask_bool.size)
    features["skin_pixel_ratio"] = float(np.count_nonzero(mask_bool) / mask_bool.size)

    return features


def load_scaler(scaler_path):
    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)

    return scaler


def build_color_input(features, scaler):
    cols = scaler["color_feature_cols"]
    median = scaler["median"]
    mean = scaler["mean"]
    std = scaler["std"]

    values = []

    for col in cols:
        raw_value = features.get(col, np.nan)

        if raw_value is None or np.isnan(raw_value):
            raw_value = median[col]

        scaled_value = (raw_value - mean[col]) / (std[col] + 1e-8)
        values.append(float(scaled_value))

    return np.array(values, dtype=np.float32).reshape(1, -1)


def load_model_for_inference(model_path):
    custom_objects = {}
    custom_objects.update(get_custom_objects())
    custom_objects.update(get_custom_layers())

    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False
    )

    return model


def predict_skin_type(image_path, model_path, scaler_path, save_face_crop=None):
    image_path = Path(image_path)

    img_rgb = read_image_rgb(image_path)
    face_rgb, crop_status, face_info = retinaface_crop_or_fallback(img_rgb)

    if face_rgb.shape[0] != IMG_SIZE or face_rgb.shape[1] != IMG_SIZE:
        face_rgb = resize_rgb(face_rgb, IMG_SIZE)

    if save_face_crop:
        Image.fromarray(face_rgb).save(save_face_crop, format="JPEG", quality=95)

    mask, segmentation_status = create_final_skin_mask(face_rgb)
    features = extract_color_features(face_rgb, mask)

    scaler = load_scaler(scaler_path)
    color_input = build_color_input(features, scaler)

    image_input = face_rgb.astype(np.float32)[None, ...]

    model = load_model_for_inference(model_path)

    probs = model.predict(
        {
            "image_input": image_input,
            "color_input": color_input
        },
        verbose=0
    )[0]

    pred_label = int(np.argmax(probs))
    pred_class = LABEL_TO_CLASS[pred_label]
    confidence = float(np.max(probs))

    top2_labels = np.argsort(probs)[-2:][::-1]
    top2_classes = [LABEL_TO_CLASS[int(i)] for i in top2_labels]

    result = {
        "image_path": str(image_path),
        "pred_label": pred_label,
        "pred_class": pred_class,
        "confidence": confidence,
        "top2_labels": [int(i) for i in top2_labels],
        "top2_classes": top2_classes,
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        },
        "crop_status": crop_status,
        "segmentation_status": segmentation_status,
        "skin_pixel_ratio": float(features.get("skin_pixel_ratio", np.nan)),
        "ita_mean": float(features.get("ita_mean", np.nan)),
        "lab_l_mean": float(features.get("lab_l_mean", np.nan)),
        "face_info": face_info
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="SYNAR Skin Type CNN Hybrid Inference")

    parser.add_argument("--image", required=True, help="Path gambar input")
    parser.add_argument(
        "--model",
        default="/content/drive/MyDrive/SYNAR/pipeline_v3/model_effnetb3_hybrid/best_effnetb3_hybrid.keras",
        help="Path model .keras"
    )
    parser.add_argument(
        "--scaler",
        default="/content/drive/MyDrive/SYNAR/pipeline_v3/model_effnetb3_hybrid/feature_scaler.json",
        help="Path feature_scaler.json"
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="Path output JSON opsional"
    )
    parser.add_argument(
        "--save_face_crop",
        default="",
        help="Path untuk menyimpan face crop opsional"
    )

    args = parser.parse_args()

    result = predict_skin_type(
        image_path=args.image,
        model_path=args.model,
        scaler_path=args.scaler,
        save_face_crop=args.save_face_crop if args.save_face_crop else None
    )

    print(json.dumps(result, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print("Saved:", args.output_json)


if __name__ == "__main__":
    main()
