import librosa
import numpy as np

def solution(audio_path):
    # Mel spectrogram
    y, sr = librosa.load(audio_path, sr=None)
    n_fft = 2048
    hop_length = 512
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=22000)
    spec_db = librosa.power_to_db(spec)

    # Normalizating in 0-255
    h,w=spec_db.shape
    for i in range(0,h):
        for j in range(0,w):
            spec_db[i][j] = (spec_db[i][j]) % 255

    # Observed that the points in the cardboard mel spectrogram was equally distributed and the standard daviation is low
    # Removing zeros
    non_zero_values = []
    for i in range(0,h):
        for j in range(0,w):
            if spec_db[i][j]!=0:
                non_zero_values.append(spec_db[i][j])
    std_dev = np.std(non_zero_values)

    # Got this thresold std_dev after performing on 15 self made test cases
    if std_dev > 45:
        return "metal"
    else:
        return "cardboard"