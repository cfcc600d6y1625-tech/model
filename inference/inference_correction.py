import numpy as np
import pandas as pd
import joblib
import sklearn
import xgboost

pipeline = joblib.load("/content/drive/MyDrive/Capstone_CC26-PSU349/model/correction_pipeline.pkl")

MED_LOWER = {1: 200, 2: 250, 3: 300, 4: 450, 5: 600, 6: 1000}

SKIN_TYPE_INFO = {
    1: {"spf_min": 30},
    2: {"spf_min": 30},
    3: {"spf_min": 30},
    4: {"spf_min": 30},
    5: {"spf_min": 15},
    6: {"spf_min": 15},
}

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

def predict_correction(t2m, rh2m, ws2m, hr, bulan):
    features = create_features(t2m, rh2m, ws2m, hr, bulan)
    correction = pipeline.predict(features)[0]

    correction = float(np.clip(correction, 0.80, 1.20))

    return correction

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

    return f"Aman untuk keluar. Gunakan sunscreen minimal SPF {spf} PA++++ untuk berjaga-jaga."

def predict_synar(uvi, skin_type, t2m, rh2m, ws2m, hr, bulan):
    
    # 1. Base duration (MED formula)
    if uvi > 0:
        durasi_base = MED_LOWER[skin_type] / (uvi * 1.5)
    else:
        durasi_base = 120.0

    durasi_base = min(durasi_base, 120.0)

    # 2. Correction ML
    correction = predict_correction(t2m, rh2m, ws2m, hr, bulan)
    durasi_final = min(durasi_base * correction, 120.0)

    # 3. Format display
    if durasi_final >= 120:
        durasi_display = "> 120 menit"
    elif durasi_final < 10:
        durasi_display = "< 10 menit"
    else:
        durasi_display = f"{round(durasi_final)} menit"

    # 4. Tips sederhana (UI friendly)
    tips = generate_spf_tips(uvi, skin_type)

    return {
        "uv_index": uvi,
        "durasi_aman_menit": round(durasi_final, 1),
        "durasi_display": durasi_display,
        "tips": tips
    }

if __name__ == "__main__":
    result = predict_synar(
        uvi=2,
        skin_type=4,
        t2m=28,
        rh2m=75,
        ws2m=2.0,
        hr=15,
        bulan=5
    )

    print(result)