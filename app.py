import streamlit as st
import numpy as np
import pandas as pd
import joblib
import keras
import tempfile
from pathlib import Path

from audio_preprocess import load_audio, crop_audio, split_segments, mel_spectrogram_rgb


ART_DIR = Path(__file__).resolve().parent / "exported_models"


@st.cache_resource
def load_mobilenet():
    model_path = ART_DIR / "mobilenetv2_finetuned.keras"
    meta_path = ART_DIR / "mobilenetv2_meta.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta não encontrado: {meta_path}")

    model = keras.models.load_model(model_path, compile=False)
    meta = joblib.load(meta_path)
    label_classes = meta.get("label_classes", None)

    if label_classes is None:
        try:
            n_out = int(model.output_shape[-1])
            label_classes = [str(i) for i in range(n_out)]
        except Exception:
            label_classes = [str(i) for i in range(10)]

    return model, list(label_classes), meta


def topk(probs, labels, k=3):
    idx = np.argsort(probs)[::-1][:k]
    return [(labels[i], float(probs[i])) for i in idx]


st.set_page_config(page_title="Audio Genre Classifier (Deep Learning)", layout="centered")
st.title("Audio Genre Classifier (Deep Learning)")

st.markdown(
    """
**Pipeline:** Audio upload → crop (30s) → segments (3s) → Mel-spectrogram (224×224 RGB) → MobileNetV2 → Top-K.
"""
)

audio_file = st.file_uploader(
    "Audio upload (WAV/MP3/FLAC/OGG/M4A)",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
)

st.divider()

crop_mode = st.selectbox("Crop for long files (30s)", ["middle_30s", "first_30s", "random_30s"])
seed = st.number_input("Seed (random_30s)", value=42, step=1)

inference_mode = st.radio("Inference mode", ["Single (3s central)", "Multi (Several segments average)"])
stride = st.selectbox("Stride (Multi)", [1, 3, 5], index=1)
max_segs = st.selectbox("Max segments (Multi)", [5, 10, 15], index=1)

top_k = st.slider("Top-K predicts", 1, 10, 3)

with st.expander("What this platform does?"):
    st.markdown(
        """
- **Cut (30s)**: reduces variation caused by intros/others and standardizes the duration.
- **Segments (3s)**:
  - *Single*: uses 1 central segment.
  - *Multi*: evaluates several segments and mçakes the average of probabilities.
- **Spectogram**: Mel-spectrogram → log(dB) → normalization → RGB 224×224.
"""
    )

run = st.button("Run prediction", type="primary", disabled=(audio_file is None))


if run and audio_file is not None:
    mode_map = {"middle_30s": "middle", "first_30s": "first", "random_30s": "random"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.name}") as tmp:
        tmp.write(audio_file.getbuffer())
        audio_path = tmp.name

    y, sr = load_audio(audio_path, target_sr=22050)

    y30 = crop_audio(y, sr, 30.0, mode=mode_map[crop_mode], seed=int(seed))

    if inference_mode.startswith("Single"):
        segs = [crop_audio(y30, sr, 3.0, mode="middle", seed=int(seed))]
    else:
        segs = split_segments(
            y30,
            sr,
            seg_sec=3.0,
            stride_sec=float(stride),
            max_segs=int(max_segs),
        )

    st.caption(f"Used segments: {len(segs)} (sr={sr})")

    model, labels, meta = load_mobilenet()

    probs_all = []
    preview_img = None

    for seg in segs:
        img = mel_spectrogram_rgb(seg, sr, out_size=(224, 224))
        if preview_img is None:
            preview_img = img
        x = np.expand_dims(img, axis=0) 
        p = model.predict(x, verbose=0)[0]
        probs_all.append(p)

    probs = np.mean(np.vstack(probs_all), axis=0)

    st.image(
        (preview_img * 255).astype(np.uint8),
        caption="Mel-spectrogram (1º segment preview)",
        use_column_width=True,
    )

    preds = topk(probs, labels, k=top_k)
    st.table(pd.DataFrame(preds, columns=["Class", "Probability"]))

    with st.expander("Model's metadata"):
        st.json(meta)
