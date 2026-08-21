import librosa
import numpy as np
import pandas as pd


def extract_features(file_path):
    """
    Extracts the exact same handcrafted audio features used during training:
    MFCCs, chroma, spectral contrast, tonnetz, spectral centroid/rolloff/bandwidth,
    zero crossing rate, mel spectrogram stats, tempo, RMS energy, and spectral flatness.

    Returns a dict of feature_name -> value (NOT yet ordered/scaled for the model).
    """
    y, sr = librosa.load(file_path, duration=30)
    features = {}

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        features[f'mfcc_{i}_max'] = np.max(mfccs[i])
        features[f'mfcc_{i}_min'] = np.min(mfccs[i])

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f'chroma_{i}_mean'] = np.mean(chroma[i])
        features[f'chroma_{i}_std'] = np.std(chroma[i])

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    for i in range(7):
        features[f'contrast_{i}_mean'] = np.mean(contrast[i])
        features[f'contrast_{i}_std'] = np.std(contrast[i])

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    for i in range(6):
        features[f'tonnetz_{i}_mean'] = np.mean(tonnetz[i])
        features[f'tonnetz_{i}_std'] = np.std(tonnetz[i])

    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    features['spectral_centroid_max'] = np.max(spectral_centroids)
    features['spectral_centroid_min'] = np.min(spectral_centroids)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_std'] = np.std(rolloff)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['bandwidth_mean'] = np.mean(bandwidth)
    features['bandwidth_std'] = np.std(bandwidth)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # Mel Spectrogram Statistics
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel_mean'] = np.mean(mel_spec_db)
    features['mel_std'] = np.std(mel_spec_db)
    features['mel_max'] = np.max(mel_spec_db)
    features['mel_min'] = np.min(mel_spec_db)
    features['mel_median'] = np.median(mel_spec_db)

    # Rhythm Features
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(np.asarray(tempo).item())

    # RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    features['rms_max'] = np.max(rms)
    features['rms_min'] = np.min(rms)

    # Spectral Flatness (how noisy vs tonal)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features['flatness_mean'] = np.mean(flatness)
    features['flatness_std'] = np.std(flatness)

    return features


def featurize_for_model(file_path, feature_columns):
    """
    Runs extract_features() on a single audio file and returns a model-ready,
    single-row DataFrame with columns in the EXACT order the model was trained on.

    Args:
        file_path: path to a .wav file (already converted from mp4/mp3/etc. if needed)
        feature_columns: list of column names in training order, loaded from
                          feature_columns.pkl (saved from the training notebook via
                          joblib.dump(x.columns.tolist(), "feature_columns.pkl"))

    Returns:
        A single-row pandas DataFrame ready to be passed to scaler.transform().
    """
    features_dict = extract_features(file_path)
    df = pd.DataFrame([features_dict])
    df = df[feature_columns]  # enforce exact training column order
    return df