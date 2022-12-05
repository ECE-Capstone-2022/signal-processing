from piano_pi import *
from scipy.io import wavfile
from IPython import display

audio_path = "/Users/macea/Documents/Personal/github/signal-processing/assets/recordings/marco_speech_18_500.wav"
# audio_path = "/Users/macea/Documents/Personal/github/signal-processing/assets/recordings/D#4vH.wav"

sample_rate, audio_time_series = wavfile.read(audio_path)

print(len(np.shape(audio_time_series)))

if len(np.shape(audio_time_series)) != 1:
  audio_time_series = audio_time_series[:,0]

print(np.shape(audio_time_series))

PianoPi = PianoPi(sample_rate, 15)

PianoPi.generate_frequencies_through_time(audio_time_series)

PianoPi.plot_frequencies_through_time()

# PianoPi.generate_tsv()