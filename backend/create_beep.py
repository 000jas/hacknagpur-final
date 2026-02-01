import numpy as np
from scipy.io import wavfile

# Generate a beep sound
sample_rate = 44100  # Hz
duration = 0.5  # seconds
frequency = 1000  # Hz (1kHz beep)

# Generate time array
t = np.linspace(0, duration, int(sample_rate * duration), False)

# Generate sine wave
beep = np.sin(frequency * 2 * np.pi * t)

# Add envelope to avoid clicking
envelope = np.ones_like(beep)
fade_samples = int(sample_rate * 0.05)  # 50ms fade
envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
beep = beep * envelope

# Convert to 16-bit audio
beep = (beep * 32767).astype(np.int16)

# Save as WAV file (pygame can play WAV files)
wavfile.write('beep.mp3', sample_rate, beep)
print("✓ Beep sound created: beep.mp3 (WAV format)")
