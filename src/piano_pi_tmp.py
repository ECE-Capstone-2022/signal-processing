##################
## Introduction ##
##################

'''
The Piano Pi Signal Processing Module

Author: Marco Acea
Andrew ID: macea
Contact: aceamarco@gmail.com / macea@andrew.cmu.edu
'''

#############
## Imports ##
#############

from SDTF import SDFTBin, PLAY_RATE, SAMPLE_RATE, MAGNITUDE_MAX
import matplotlib.pyplot as plt
from scipy.io import wavfile
from math import e
import numpy as np
import os
import uuid
import csv


###############
## Constants ##
###############

# Frequencies corresponding to each piano key
PIANO_KEY_FREQUENCIES = []

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
    freq_txt = f.read().split('\n')
    for n in freq_txt:
      PIANO_KEY_FREQUENCIES.append(float(n))

f.close()

# Generating column headers for the tsv outputs
TSV_HEADERS = ['time_stamp']
for i, key_freq in enumerate(PIANO_KEY_FREQUENCIES):
  TSV_HEADERS.append(f'key{i}_{key_freq}Hz')


class PianoPi:

  def __init__(self, sample_rate=SAMPLE_RATE, play_rate=PLAY_RATE):
    self.sample_rate = sample_rate
    self.play_rate = play_rate
    self.N_max = sample_rate // play_rate
    self.sample_window = sample_rate // play_rate
    self.uuid = uuid.uuid4()

  
  def generate_frequencies_through_time(self, audio):
    # Preconditions
    assert(PIANO_KEY_FREQUENCIES)

    self.audio_len = len(audio)
    print(self.audio_len)
    self.SDFTBins = [SDFTBin(freq, self.sample_rate) for freq in PIANO_KEY_FREQUENCIES]
    self.key_freq_through_time = [[] for i in range(len(PIANO_KEY_FREQUENCIES))]
    self.x_n_through_time = [[] for i in range(len(PIANO_KEY_FREQUENCIES))]
    assert(len(self.key_freq_through_time) == len(self.SDFTBins))

    for i, bin in enumerate(self.SDFTBins):
      print(f'Parsing audio file for key {i+1}')
      x_n, X_k = bin.parse(audio)
      self.key_freq_through_time[i] = X_k
      self.x_n_through_time[i] = x_n
    print(np.shape(self.key_freq_through_time))


  def plot_frequencies_through_time(self):
    '''Generates a 3-D plot of the frequency components for each piano key
    changing through time.'''

    assert(self.key_freq_through_time)
    assert(self.audio_len)
    assert(self.sample_window)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for n in range(len(self.key_freq_through_time[0])):
      freqs_at_n = [abs(key[n]) for key in self.key_freq_through_time]
      Y = PIANO_KEY_FREQUENCIES
      Z = freqs_at_n
      X = np.full(len(PIANO_KEY_FREQUENCIES), n * (1/self.play_rate))
      ax.plot(X,Y,Z)

    # Labels
    ax.set(title="Change in Freqeuncy Across Time Using SDFT", xlabel=r'Time $t$ [s]', ylabel=r'Frequency $\omega$ [Hz]')
    ax.set_zlabel("Amplitude")
    ax.tick_params(axis='z', which='major', pad=-3)

    # Generating file path for plot image
    id = str(self.uuid)
    file_path = f'out/{id}/plots'

    if not os.path.exists(file_path):
      os.makedirs(file_path)
    file_path = file_path + f'/{id}_3d.png'

    plt.savefig(file_path)
    
    return plt.show()

  def reconstruct_audio(self):
    '''Reconstructs an output audio file using the frequency components of every
    piano sample'''

    assert(self.key_freq_through_time)
    assert(self.audio_len)
    assert(self.sample_window)

    x = []

    for n in range(len(self.key_freq_through_time[0])):
      freqs_at_n = [np.abs(key[n]) for key in self.key_freq_through_time]
      #TODO Finish writing this line
      # x_n_per_key = [key[n] for key in self.]

      # Fill in 0's between piano samples
      x.extend([0]*(self.N_max - 1))

      # Each Key => |X_k| * delta[n-self.note_frequency] -IDFT-> |X_k|e^(j*self.note_frequency*n)
      x.append(sum(self.x_n_through_time[n]))
    
    fig, ax = plt.subplots()


    # Generating file path for audio file
    id = str(self.uuid)
    file_path = f'out/{id}/audio'

    if not os.path.exists(file_path):
      os.makedirs(file_path)
    file_path = file_path + f'/{id}_out_test.wav'

    wavfile.write(file_path, self.sample_rate, np.asarray(x))

    return file_path

  def generate_tsv(self, amplitude=MAGNITUDE_MAX):
    '''Generates a text file containing what keys to play, returns unique id
    for given recording.'''

    # Preconditions
    assert(self.key_freq_through_time)
    assert(TSV_HEADERS)
    assert(self.audio_len)
    assert(self.sample_window)
    assert(self.uuid)

    # Generate a unique id for this audio recording
    id = str(self.uuid)
    file_path = f'out/{id}'

    if not os.path.exists(file_path):
      os.makedirs(file_path)
    file_path = file_path + f'/{id}.tsv'

    # Create the text file named {uuid}.tsv
    with open(file_path,'wt') as out_file:
      # Write the column headers
      tsv_writer = csv.writer(out_file, delimiter='\t')
      tsv_writer.writerow(TSV_HEADERS)

      # Iterate through every play rate sample
      for i in range(len(self.key_freq_through_time[0])):
        time_stamp_ms = round(i * (1 / self.play_rate) * 1000)
        tsv_row = [f'{time_stamp_ms}']
        for key in self.key_freq_through_time:
          tsv_row.append(f'{100*(np.abs(key[i]) / MAGNITUDE_MAX)}')
        tsv_writer.writerow(tsv_row)

    out_file.close()

    return file_path
