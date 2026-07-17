from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title="SYNAR Regression API")

# =========================
# LOAD MODEL
# =========================
correction_pipeline = joblib.load("correction_pipeline.pkl")
best_time_pipeline  = joblib.load("best_time_pipeline.pkl")

# =========================
# CONSTANT
# =========================
MED_LOWER = {1: 200, 2: 250, 3: 300, 4: 450, 5: 600, 6: 1000}

SKIN_TYPE_INFO = {
    1: {"spf_min": 30},
    2: {"spf_min": 30},
    3: {"spf_min": 30},
    4: {"spf_min": 30},
    5: {"spf_min": 15},
    6: {"spf_min": 15},
}

# =========================
# REQUEST SCHEMA
# =========================
class CorrectionRequest(BaseModel):
    uvi: float
    skin_type: int
    t2m: float
    rh2m: float
    ws2m: float
    hr: int
    bulan: int


class BestTimeRequest(BaseModel):
    bulan: int
    t2m_per_jam: dict
    rh2m_per_jam: dict
    ws2m_per_jam: dict


# =========================
# HELPER FUNCTIONS
# =========================
def create_features(t2m, rh2m, ws2m, hr, bulan):
    return pd.DataFrame([{
        "T2M": t2m,
        "RH2M": rh2m,
        "WS2M": ws2m,
        "HR_sin": np.sin(2 * np.pi * hr / 24),
        "HR_cos": np.cos(2 * np.pi * hr / 24),
        "BULAN_sin": np.sin(2 * np.pi * bulan / 12),
        "BULAN_cos": np.cos(2 * np.pi * bulan / 12),
        "RH_x_T2M": rh2m * t2m,
        "RH_x_WS": rh2m * ws2m
    }])


def predict_correction_model(t2m, rh2m, ws2m, hr, bulan):
    features = create_features(t2m, rh2m, ws2m, hr, bulan)
    correction = correction_pipeline.predict(features)[0]
    return float(np.clip(correction, 0.80, 1.20))


def generate_spf_tips(uvi, skin_type):
    spf_min = SKIN_TYPE_INFO[skin_type]["spf_min"]

    if uvi >= 8:
        spf = max(spf_min, 50)
    elif uvi >= 6:
        spf = max(spf_min, 30)
    elif uvi >= 3:
        spf = max(spf_min, 30)
    else:
        spf = spf_min

    return f"Aman untuk keluar. Gunakan sunscreen minimal SPF {spf} PA++++."


# =========================
# ENDPOINT 1: CORRECTION
# =========================
@app.post("/predict/correction")
def predict_correction(req: CorrectionRequest):

    if req.skin_type not in MED_LOWER:
        return {"error": "skin_type harus 1–6"}

    # base duration
    if req.uvi > 0:
        durasi_base = MED_LOWER[req.skin_type] / (req.uvi * 1.5)
    else:
        durasi_base = 120.0

    durasi_base = min(durasi_base, 120.0)

    # correction ML
    correction = predict_correction_model(
        req.t2m, req.rh2m, req.ws2m, req.hr, req.bulan
    )

    durasi_final = min(durasi_base * correction, 120.0)

    if durasi_final >= 120:
        display = "> 120 menit"
    elif durasi_final < 10:
        display = "< 10 menit"
    else:
        display = f"{round(durasi_final)} menit"

    return {
        "uv_index": req.uvi,
        "durasi_aman_menit": round(durasi_final, 1),
        "durasi_display": display,
        "tips": generate_spf_tips(req.uvi, req.skin_type)
    }


# =========================
# ENDPOINT 2: BEST TIME
# =========================
@app.post("/predict/best-time")
def predict_best_time(req: BestTimeRequest):

    def build_row(t2m, rh2m, ws2m, hr, bulan):
        return pd.DataFrame([{
            "T2M": t2m,
            "RH2M": rh2m,
            "WS2M": ws2m,
            "HR_sin": np.sin(2 * np.pi * hr / 24),
            "HR_cos": np.cos(2 * np.pi * hr / 24),
            "BULAN_sin": np.sin(2 * np.pi * bulan / 12),
            "BULAN_cos": np.cos(2 * np.pi * bulan / 12),
            "T2M_x_RH": t2m * rh2m,
            "LOKASI": "Bali",
            "MUSIM": "Kemarau"
        }])

    hours = sorted(req.t2m_per_jam.keys())

    all_rows = pd.concat([
        build_row(
            req.t2m_per_jam[h],
            req.rh2m_per_jam.get(h, 75),
            req.ws2m_per_jam.get(h, 2),
            h,
            req.bulan
        )
        for h in hours
    ], ignore_index=True)

    scores = best_time_pipeline.predict(all_rows)

    results = list(zip(hours, scores))

    best = sorted(results, key=lambda x: x[1], reverse=True)[:3]
    worst = sorted(results, key=lambda x: x[1])[:3]

    return {
        "best_hours": [f"{h:02d}:00" for h, _ in best],
        "worst_hours": [f"{h:02d}:00" for h, _ in worst]
    }


# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {"message": "SYNAR API is running 🚀"}