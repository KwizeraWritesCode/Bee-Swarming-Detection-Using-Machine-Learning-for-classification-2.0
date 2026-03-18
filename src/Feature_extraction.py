import os
import numpy as np
import pandas as pd
import librosa

def extract_features(file_path, label):
    try:
        signal, sr = librosa.load(file_path, sr=None)

        # MFCC
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(signal))

        # Spectral Centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))

        return list(mfcc_mean) + [zcr, spectral_centroid, label]

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_folder(folder_path, label):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            features = extract_features(os.path.join(folder_path, file), label)
            if features:
                data.append(features)
    return data