import os
import io
import json
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# File lokal di folder deployment
from .custom_loss import get_custom_objects
from .custom_layer import get_custom_layers

from .inference_cnn import (
    retinaface_crop_or_fallback,
    create_final_skin_mask,
    extract_color_features,
    load_scaler,
    build_color_input,
    CLASS_NAMES,
    LABEL_TO_CLASS,
    IMG_SIZE,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = BASE_DIR.parent

DEFAULT_KERAS_MODEL_PATH = PIPELINE_DIR / "model_cnn_skintype.keras"
DEFAULT_SAVED_MODEL_DIR = PIPELINE_DIR / "model_cnn_skintype_saved"

DEFAULT_SCALER_PATH = DEFAULT_SAVED_MODEL_DIR / "feature_scaler.json"

if not DEFAULT_SCALER_PATH.exists():
    DEFAULT_SCALER_PATH = PIPELINE_DIR / "model_effnetb3_hybrid" / "feature_scaler.json"

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "keras").lower()

KERAS_MODEL_PATH = Path(
    os.getenv(
        "KERAS_MODEL_PATH",
        str(DEFAULT_KERAS_MODEL_PATH)
    )
)

SAVED_MODEL_DIR = Path(
    os.getenv(
        "SAVED_MODEL_DIR",
        str(DEFAULT_SAVED_MODEL_DIR)
    )
)

SCALER_PATH = Path(
    os.getenv(
        "SCALER_PATH",
        str(DEFAULT_SCALER_PATH)
    )
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def image_bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    """
    Mengubah file upload menjadi RGB numpy array.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            return np.array(img)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"File yang diupload bukan gambar valid. Error: {str(exc)}"
        )


def load_keras_model(model_path: Path):
    """
    Load model .keras dengan custom loss, metric, dan layer.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Keras model tidak ditemukan: {model_path}")

    custom_objects = {}
    custom_objects.update(get_custom_objects())
    custom_objects.update(get_custom_layers())

    model = keras.models.load_model(
        str(model_path),
        custom_objects=custom_objects,
        compile=False
    )

    return model


def load_saved_model(saved_model_dir: Path):
    """
    Load model TensorFlow SavedModel.
    """
    if not saved_model_dir.exists():
        raise FileNotFoundError(f"SavedModel folder tidak ditemukan: {saved_model_dir}")

    saved_model = tf.saved_model.load(str(saved_model_dir))
    signatures = saved_model.signatures

    if "serve" in signatures:
        serving_key = "serve"
    elif "serving_default" in signatures:
        serving_key = "serving_default"
    else:
        available = list(signatures.keys())
        raise ValueError(f"Tidak ada serving signature. Available signatures: {available}")

    serving_fn = signatures[serving_key]

    return saved_model, serving_fn, serving_key


def run_saved_model_inference(serving_fn, image_input, color_input):
    """
    Menjalankan inference untuk backend SavedModel.
    """
    image_tensor = tf.constant(image_input, dtype=tf.float32)
    color_tensor = tf.constant(color_input, dtype=tf.float32)

    try:
        result = serving_fn(
            image_input=image_tensor,
            color_input=color_tensor
        )

    except TypeError:
        try:
            result = serving_fn({
                "image_input": image_tensor,
                "color_input": color_tensor
            })

        except TypeError:
            result = serving_fn(
                args_0={
                    "image_input": image_tensor,
                    "color_input": color_tensor
                }
            )

    if isinstance(result, dict):
        output_tensor = list(result.values())[0]
    else:
        output_tensor = result

    probs = output_tensor.numpy()

    return probs


def predict_from_rgb_image(
    app: FastAPI,
    img_rgb: np.ndarray,
    filename: str = "",
    return_debug: bool = False
):
    """
    Pipeline inference lengkap:
    image upload -> face crop -> skin segmentation -> color features -> model prediction.
    """

    # 1. Face crop
    face_rgb, crop_status, face_info = retinaface_crop_or_fallback(img_rgb)

    # 2. Pastikan ukuran 224x224
    if face_rgb.shape[0] != IMG_SIZE or face_rgb.shape[1] != IMG_SIZE:
        face_rgb = np.array(
            Image.fromarray(face_rgb).resize((IMG_SIZE, IMG_SIZE))
        )

    # 3. Skin segmentation
    mask, segmentation_status = create_final_skin_mask(face_rgb)

    # 4. Extract LAB/HSV/ITA features
    features = extract_color_features(face_rgb, mask)

    # 5. Standardize color features
    color_input = build_color_input(features, app.state.scaler)

    # 6. Prepare image input
    image_input = face_rgb.astype(np.float32)[None, ...]

    # 7. Predict
    if app.state.model_backend == "savedmodel":
        probs = run_saved_model_inference(
            app.state.serving_fn,
            image_input,
            color_input
        )

    else:
        probs = app.state.model.predict(
            {
                "image_input": image_input,
                "color_input": color_input
            },
            verbose=0
        )

    probs = np.asarray(probs)[0]

    # 8. Format output
    pred_label = int(np.argmax(probs))
    pred_class = LABEL_TO_CLASS[pred_label]
    confidence = float(np.max(probs))

    top2_labels = np.argsort(probs)[-2:][::-1]
    top2_classes = [LABEL_TO_CLASS[int(i)] for i in top2_labels]

    result = {
        "filename": filename,
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
        "model_backend": app.state.model_backend
    }

    if return_debug:
        result["debug_features"] = {
            "lab_l_mean": float(features.get("lab_l_mean", np.nan)),
            "lab_a_mean": float(features.get("lab_a_mean", np.nan)),
            "lab_b_mean": float(features.get("lab_b_mean", np.nan)),
            "hsv_h_mean": float(features.get("hsv_h_mean", np.nan)),
            "hsv_s_mean": float(features.get("hsv_s_mean", np.nan)),
            "hsv_v_mean": float(features.get("hsv_v_mean", np.nan)),
            "ita_mean": float(features.get("ita_mean", np.nan)),
            "skin_pixel_ratio": float(features.get("skin_pixel_ratio", np.nan))
        }
        result["face_info"] = face_info

    return result


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model dan scaler saat server pertama kali start.
    """

    print("Starting SYNAR Skin Type FastAPI service...")
    print("BASE_DIR:", BASE_DIR)
    print("PIPELINE_DIR:", PIPELINE_DIR)
    print("MODEL_BACKEND:", MODEL_BACKEND)
    print("KERAS_MODEL_PATH:", KERAS_MODEL_PATH)
    print("SAVED_MODEL_DIR:", SAVED_MODEL_DIR)
    print("SCALER_PATH:", SCALER_PATH)

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Feature scaler tidak ditemukan: {SCALER_PATH}")

    app.state.scaler = load_scaler(str(SCALER_PATH))
    app.state.model_backend = MODEL_BACKEND

    if MODEL_BACKEND == "savedmodel":
        saved_model, serving_fn, serving_key = load_saved_model(SAVED_MODEL_DIR)

        app.state.saved_model = saved_model
        app.state.serving_fn = serving_fn
        app.state.serving_key = serving_key
        app.state.model = None

        print("Loaded SavedModel:", SAVED_MODEL_DIR)
        print("Serving key:", serving_key)

    else:
        model = load_keras_model(KERAS_MODEL_PATH)

        app.state.model = model
        app.state.saved_model = None
        app.state.serving_fn = None
        app.state.serving_key = None

        print("Loaded Keras model:", KERAS_MODEL_PATH)

    print("Loaded scaler:", SCALER_PATH)
    print("Class names:", CLASS_NAMES)
    print("Service ready.")

    yield

    print("Shutting down SYNAR Skin Type FastAPI service...")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="SYNAR Skin Type CNN API",
    description="FastAPI serving untuk model CNN Hybrid Skin Type Classification.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "SYNAR Skin Type CNN API is running.",
        "model_backend": MODEL_BACKEND,
        "class_names": CLASS_NAMES,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_backend": app.state.model_backend,
        "keras_model_path": str(KERAS_MODEL_PATH),
        "saved_model_dir": str(SAVED_MODEL_DIR),
        "scaler_path": str(SCALER_PATH),
        "image_size": IMG_SIZE,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    return_debug: bool = Query(False)
):
    """
    Prediksi 1 gambar.
    """

    if file.content_type is not None:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"File harus berupa image. content_type diterima: {file.content_type}"
            )

    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="File kosong.")

        img_rgb = image_bytes_to_rgb(image_bytes)

        result = predict_from_rgb_image(
            app=app,
            img_rgb=img_rgb,
            filename=file.filename,
            return_debug=return_debug
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "traceback": traceback.format_exc()
        }


@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    return_debug: bool = Query(False)
):
    """
    Prediksi banyak gambar sekaligus.
    """

    results = []

    for file in files:
        try:
            if file.content_type is not None:
                if not file.content_type.startswith("image/"):
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": f"File bukan image: {file.content_type}"
                    })
                    continue

            image_bytes = await file.read()

            if len(image_bytes) == 0:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "File kosong."
                })
                continue

            img_rgb = image_bytes_to_rgb(image_bytes)

            result = predict_from_rgb_image(
                app=app,
                img_rgb=img_rgb,
                filename=file.filename,
                return_debug=return_debug
            )

            result["status"] = "ok"
            results.append(result)

        except Exception as exc:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(exc)
            })

    return {
        "total": len(results),
        "results": results
    }
