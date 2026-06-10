"""
SYNAR — Inference Best Time to Go Outside
Model: XGBoost Regressor (dalam sklearn Pipeline)
Pipeline .pkl sudah menyimpan scaler di dalamnya.
 
Cara penggunaan:
    from inference_best_time import predict_best_time
 
    result = predict_best_time(
        bulan=6,
        t2m_per_jam ={7:27, 8:28, 9:29, 10:31, 11:33, 12:34,
                      13:34, 14:33, 15:32, 16:31},
        rh2m_per_jam={7:80, 8:78, 9:76, 10:74, 11:70, 12:68,
                      13:67, 14:68, 15:70, 16:72},
        ws2m_per_jam={7:1.5, 8:1.8, 9:2.0, 10:2.3, 11:2.5,
                      12:2.8, 13:3.0, 14:2.8, 15:2.5, 16:2.2},
    )
"""
 
import numpy as np
import pandas as pd
import joblib
 
 
# ── Load pipeline (sekali saja saat module di-import) ──────
pipeline = joblib.load("/content/drive/MyDrive/Capstone_CC26-PSU349/model/best_time_pipeline.pkl")
 
 
# ── Nama fitur harus sama persis dengan saat training ──────
FEATURES = [
    "T2M", "RH2M", "WS2M",
    "HR_sin", "HR_cos",
    "BULAN_sin", "BULAN_cos",
    "T2M_x_RH",
]
 
 
def _build_row(t2m, rh2m, ws2m, hr, bulan,
               lokasi="Bali", musim="Kemarau") -> pd.DataFrame:

    return pd.DataFrame([{
        "T2M":       t2m,
        "RH2M":      rh2m,
        "WS2M":      ws2m,
        "HR_sin":    np.sin(2 * np.pi * hr / 24),
        "HR_cos":    np.cos(2 * np.pi * hr / 24),
        "BULAN_sin": np.sin(2 * np.pi * bulan / 12),
        "BULAN_cos": np.cos(2 * np.pi * bulan / 12),
        "T2M_x_RH":  t2m * rh2m,

       
        "LOKASI": lokasi,
        "MUSIM": musim
    }])
 
 
def _hours_display(hrs: list) -> str:
    hrs = sorted(hrs)
    if len(hrs) > 1 and hrs[-1] - hrs[0] == len(hrs) - 1:
        return f"{hrs[0]:02d}:00–{hrs[-1]:02d}:00"
    return ", ".join(f"{h:02d}:00" for h in hrs)
 
 
def predict_best_time(
    bulan: int,
    t2m_per_jam:  dict,
    rh2m_per_jam: dict,
    ws2m_per_jam: dict,
    top_n:        int = 3,
    window_start: int = 7,
    window_end:   int = 16,
) -> dict:
    """
    Prediksi jam terbaik dan terburuk untuk aktivitas outdoor.
 
    Parameters
    ----------
    bulan         : bulan saat ini (1–12)
    t2m_per_jam   : dict {jam (int): suhu °C}
    rh2m_per_jam  : dict {jam (int): kelembapan %}
    ws2m_per_jam  : dict {jam (int): kecepatan angin m/s}
    top_n         : jumlah jam terbaik yang dikembalikan
    window_start  : jam paling awal yang dipertimbangkan (default 7)
    window_end    : jam paling akhir yang dipertimbangkan (default 16)
 
    Returns
    -------
    dict berisi:
        best_hours      : list jam terbaik, format "HH:00"
        worst_hours     : list jam terburuk, format "HH:00"
        hourly_forecast : list prediksi per jam lengkap
        insights        : list kalimat rekomendasi
        recommendation  : string ringkas untuk UI
        window          : string window yang digunakan
    """
    hours = sorted(t2m_per_jam.keys())
 
    # Bangun semua baris sekaligus lalu predict satu kali
    all_rows = pd.concat(
        [_build_row(
            t2m_per_jam[h],
            rh2m_per_jam.get(h, 75.0),
            ws2m_per_jam.get(h, 2.0),
            h, bulan,
            lokasi="Bali",
            musim="Kemarau"
        ) for h in hours],
        ignore_index=True
    )
 
    # Pipeline predict langsung — scaler sudah di dalam .pkl
    scores = pipeline.predict(all_rows)
 
    results = pd.DataFrame({
        "hr":    hours,
        "score": scores,
        "t2m":   [t2m_per_jam[h]          for h in hours],
        "rh2m":  [rh2m_per_jam.get(h, 75) for h in hours],
        "ws2m":  [ws2m_per_jam.get(h, 2)  for h in hours],
    })
 
    # Kandidat hanya jam dalam window berjemur
    candidates = results[
        (results["hr"] >= window_start) &
        (results["hr"] <= window_end)
    ]
    if candidates.empty:
        candidates = results
 
    sorted_c    = candidates.sort_values("score", ascending=False)
    best_hours  = sorted(sorted_c.head(top_n)["hr"].tolist())
    worst_hours = sorted(sorted_c.tail(top_n)["hr"].tolist())
 
    # Hourly forecast — semua jam ditampilkan
    hourly = []
    for _, row in results.sort_values("hr").iterrows():
        s  = float(row["score"])
        hr = int(row["hr"])
        ow = hr < window_start or hr > window_end
 
        if ow:
            label, emoji = "Di luar window berjemur", "⚪"
        elif s >= 0.65: label, emoji = "Sangat Nyaman", "🟢"
        elif s >= 0.50: label, emoji = "Cukup Nyaman",  "🟡"
        elif s >= 0.35: label, emoji = "Kurang Nyaman", "🟠"
        else:           label, emoji = "Tidak Nyaman",  "🔴"
 
        hourly.append({
            "hour":        f"{hr:02d}:00",
            "score":       round(s, 3),
            "label":       label,
            "emoji":       emoji,
            "temperature": round(float(row["t2m"]), 1),
            "humidity":    round(float(row["rh2m"]), 1),
            "in_window":   not ow,
        })
 
    # Insights
    insights = [
        f"Waktu terbaik untuk berjemur: {_hours_display(best_hours)}",
        f"Hindari berjemur saat: {_hours_display(worst_hours)}",
        f"Window rekomendasi: {window_start:02d}:00–{window_end:02d}:00",
    ]
 
    window_temps = {h: t for h, t in t2m_per_jam.items()
                    if window_start <= h <= window_end}
    if window_temps:
        peak_hr = max(window_temps, key=window_temps.get)
        if window_temps[peak_hr] > 33:
            insights.append(
                f"Suhu tertinggi {window_temps[peak_hr]:.0f}°C "
                f"sekitar {peak_hr:02d}:00 — hindari aktivitas berat."
            )
 
    return {
        "best_hours":      [f"{h:02d}:00" for h in best_hours],
        "worst_hours":     [f"{h:02d}:00" for h in worst_hours],
        "hourly_forecast": hourly,
        "insights":        insights,
        "recommendation":  " | ".join(insights[:2]),
        "window":          f"{window_start:02d}:00–{window_end:02d}:00",
    }
 
 
if __name__ == "__main__":
    result = predict_best_time(
        bulan=6,
        t2m_per_jam ={7:27, 8:28, 9:29, 10:31, 11:33, 12:34,
                      13:34, 14:33, 15:32, 16:31},
        rh2m_per_jam={7:80, 8:78, 9:76, 10:74, 11:70, 12:68,
                      13:67, 14:68, 15:70, 16:72},
        ws2m_per_jam={7:1.5, 8:1.8, 9:2.0, 10:2.3, 11:2.5,
                      12:2.8, 13:3.0, 14:2.8, 15:2.5, 16:2.2},
        window_start=7,
        window_end=16,
    )
 
    print(f"Waktu terbaik : {', '.join(result['best_hours'])}")
    print(f"Waktu terburuk: {', '.join(result['worst_hours'])}")
    print(f"\nInsight:")
    for ins in result["insights"]:
        print(f"  • {ins}")
    print(f"\nHourly forecast:")
    for h in result["hourly_forecast"]:
        if h["in_window"]:
            print(f"  {h['emoji']} {h['hour']} "
                  f"score={h['score']:.3f} "
                  f"({h['label']}) "
                  f"suhu={h['temperature']}°C")