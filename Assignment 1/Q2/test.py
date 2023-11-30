import librosa
import librosa
from PIL import Image

# Load the audio file
audio_path = 'test/metal_banging1.mp3'  # Replace with the path to your audio file
y, sr = librosa.load(audio_path, sr=None)

# Define the parameters for the Mel spectrogram
n_fft = 2048        # Size of the FFT window
hop_length = 512    # Hop length for computing the spectrogram
fmax = 22000        # Maximum frequency for the spectrogram (adjust as needed)

# Generate the Mel spectrogram
spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=fmax)

# Convert the power spectrogram to decibels (dB)
spec_db = librosa.power_to_db(spec)

# Normalize the spectrogram to fit in the 0-255 range (optional)
spec_db_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min()) * 255
spec_db_norm = spec_db_norm.astype('uint8')

# Convert the spectrogram to a PIL image
spectrogram_image = Image.fromarray(spec_db_norm)

# Display the image using your preferred image viewer
spectrogram_image.show()