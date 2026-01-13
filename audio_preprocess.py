import numpy as np
import librosa
from PIL import Image


def load_audio(path: str, target_sr: int = 22050):
    """
    Loads audio with librosa, convert to mono, e resample to target_sr.
    Returns (y, sr) with y float32.
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = np.asarray(y, dtype=np.float32)

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y, target_sr


def crop_audio(y: np.ndarray, sr: int, seconds: float, mode: str = "middle", seed: int = 42):
    """
    Crops a fixed part of seconds of the y signal.
    - middle: centered part
    - first: first
    - random: random controlled by a seed
    If the audio is small than desired, pads with zeros.
    """
    y = np.asarray(y, dtype=np.float32)
    n_target = int(seconds * sr)

    if len(y) == 0:
        return np.zeros(n_target, dtype=np.float32)

    if len(y) < n_target:
        pad = n_target - len(y)
        return np.pad(y, (0, pad), mode="constant").astype(np.float32)

    if mode == "first":
        start = 0
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        start = int(rng.integers(0, len(y) - n_target + 1))
    else:  # middle
        start = (len(y) - n_target) // 2

    return y[start:start + n_target].astype(np.float32)


def split_segments(y: np.ndarray, sr: int, seg_sec: float = 3.0, stride_sec: float = 3.0, max_segs: int = 10):
    """
    Splits y in segments of seg_sec with stride stride_sec.
    If y is small, returns a central segment (with pad if needed).
    """
    seg_n = int(seg_sec * sr)
    stride_n = int(stride_sec * sr)

    if len(y) < seg_n:
        return [crop_audio(y, sr, seg_sec, mode="middle")]

    segs = []
    for start in range(0, len(y) - seg_n + 1, stride_n):
        segs.append(y[start:start + seg_n].astype(np.float32))
        if len(segs) >= max_segs:
            break
    return segs


def mel_spectrogram_rgb(
    y: np.ndarray,
    sr: int,
    out_size=(224, 224),
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float | None = None,
    hop_length: int = 512,
):
    """
    Converts audio (1D) in RGB float32 [0,1] format (H,W,3).
    Pipeline:
    - Mel-spectrogram (power)
    - log(dB)
    - normalization min-max
    - resize to out_size
    - stack in 3 channels
    """
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if fmax is None:
        fmax = sr / 2

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        power=2.0,
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    mn = float(np.min(S_db))
    mx = float(np.max(S_db))
    if mx - mn < 1e-8:
        S_norm = np.zeros_like(S_db, dtype=np.float32)
    else:
        S_norm = (S_db - mn) / (mx - mn)
        S_norm = S_norm.astype(np.float32)

    img_u8 = (S_norm * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img_u8) 
    pil = pil.resize(out_size, resample=Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0  

    rgb = np.stack([arr, arr, arr], axis=-1).astype(np.float32)  
    return rgb
