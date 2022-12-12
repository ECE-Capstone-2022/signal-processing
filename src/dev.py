from piano_pi import *
from dbg import *
import os

audio_path = "/Users/macea/Documents/Personal/github/signal-processing/assets/recordings/marco_speech_18_500.wav"
# audio_path = "/Users/macea/Documents/Personal/github/signal-processing/assets/recordings/D#4vH.wav"

PianoPi = PianoPi(file_path = audio_path, play_rate=15)

PianoPi.plot_freq_through_time()

# PianoPi.generate_output_wav_file()

dbg_print("Number of cores: ", os.cpu_count())

# PianoPi.generate_tsv()

print(PianoPi.generate_piano_note_matrix())