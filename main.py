import os
import matplotlib.pyplot as plt
import numpy as np
# from pydub import AudioSegment

if 'key_frequencies.txt' not in os.listdir():
  with open('key_frequencies.txt', 'w') as f:
    for n in range(1,89):
      freq = (2**((n-49)/12))*440
      if (n != 88):
        text = "{:.3f}\n".format(freq)
      else:
        text = "{:.3f}".format(freq)
      if text != '': f.write(text)
    f.close()

with open('key_frequencies.txt', 'r') as f:
    PIANO_KEY_FREQUENCIES = []
    for n in f.read().split('\\n'):
      PIANO_KEY_FREQUENCIES.append(float(n))
    
    fig, ax = plt.subplots()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power') # TODO: Figure out what our units should be
    ax.set_xscale('log')
    ax.stem(PIANO_KEY_FREQUENCIES, np.ones(len(PIANO_KEY_FREQUENCIES)))