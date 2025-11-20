import streamlit as st
import numpy as np
import librosa
import joblib
import os

# ------------------------------
# Load model & preprocessing
# ------------------------------
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")
model = joblib.load("xgb_model.joblib")

# ------------------------------
# Feature extraction function
# ------------------------------
def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    features = np.concatenate([
        mfcc, mel, chroma,
        np.array([spec_centroid, spec_bw, spec_contrast, spec_rolloff]),
        np.array([rms, zcr, pitch])
    ])
    return features

# ------------------------------
# Streamlit UI Full Responsive
# ------------------------------
st.set_page_config(page_title="Oceanecho 🌊", layout="wide")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.markdown("""
# Oceanecho 🌊
**From Echoes to Insights** 🎵🐬

Aplikasi interaktif untuk:
- Memprediksi spesies mamalia laut dari audio
- Mendukung file audio (.wav, .mp3, .ogg)
- Menggunakan **kecerdasan buatan** untuk prediksi cepat dan akurat

💡 Upload file audio sendiri atau pilih sample audio di bawah untuk mencoba prediksi.
""", unsafe_allow_html=True)

# ------------------------------
# Main Header Responsif
# ------------------------------
st.markdown("""
<div style="
    text-align:center; 
    padding:5vw 2vw; 
    border-radius:15px;
    background: linear-gradient(135deg, #0a1f44, #000000);
    font-family: 'Segoe UI', 'Helvetica', sans-serif;
">
    <h1 style="color:#00b4d8; font-size: clamp(32px, 8vw, 80px); margin-bottom:1vw;">Oceanecho 🌊</h1>
    <h3 style="color:#caf0f8; font-size: clamp(18px, 4vw, 36px); margin-top:0;">From Echoes to Insights 🎵🐬</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='font-size: clamp(14px, 2.5vw, 20px);'>Upload audio atau pilih sample audio untuk mencoba prediksi spesies mamalia laut.</p>", unsafe_allow_html=True)

# ------------------------------
# File uploader responsif
# ------------------------------
uploaded_files = st.file_uploader(
    "🎧 Upload satu atau beberapa file audio (.wav/.mp3/.ogg)", 
    type=["wav","mp3","ogg"], accept_multiple_files=True
)

# ------------------------------
# Sample audio
# ------------------------------
sample_folder = "sample"
sample_files = [f for f in os.listdir(sample_folder) if f.endswith((".wav",".mp3",".ogg"))]
st.markdown("<h4 style='font-size: clamp(16px, 3vw, 24px);'>📂 Sample Audio</h4>", unsafe_allow_html=True)
sample_selection = st.multiselect("Pilih satu atau beberapa sample audio:", sample_files)

# ------------------------------
# Tentukan file untuk prediksi
# ------------------------------
file_paths = []

if uploaded_files:
    for audio_file in uploaded_files:
        temp_path = f"temp_{audio_file.name}"
        with open(temp_path, "wb") as f:
            f.write(audio_file.getvalue())
        file_paths.append(temp_path)

for f in sample_selection:
    file_paths.append(os.path.join(sample_folder, f))

# ------------------------------
# Predict button
# ------------------------------
if file_paths and st.button("🔮 Predict", key="predict_button"):
    st.markdown("<hr>", unsafe_allow_html=True)
    for path in file_paths:
        st.audio(path)
        try:
            features = extract_features(path)
            features_scaled = scaler.transform([features])
            pred = model.predict(features_scaled)
            species = label_encoder.inverse_transform(pred)[0]

            # Ambil probabilitas prediksi untuk confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_scaled)
                confidence = np.max(proba) * 100  # dalam persen
            else:
                confidence = None

            # Card-style prediction responsif dengan clamp font
            confidence_text = f"{confidence:.2f}%" if confidence is not None else "N/A"
            st.markdown(f"""
            <div style="
                padding: clamp(10px,2vw,20px); 
                margin: clamp(8px,1vw,16px) 0; 
                border-radius:12px; 
                background: linear-gradient(135deg, #001f3f, #0d1b2a);
                font-family: 'Segoe UI', 'Helvetica', sans-serif;
            ">
                <h4 style="color:#00b4d8; margin-bottom:0.5vw; font-size: clamp(14px, 3vw, 24px);">🎯 File: {os.path.basename(path)}</h4>
                <h2 style="color:#caf0f8; margin-top:0; font-size: clamp(18px, 5vw, 36px);">Predicted Species: {species}</h2>
                <p style="color:#90e0ef; font-size: clamp(14px, 2.5vw, 20px); margin:0;">Confidence: {confidence_text}</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠ Terjadi error saat memproses {os.path.basename(path)}: {e}")
