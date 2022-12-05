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
    self.sample_window = sample_rate // play_rate

  
  def generate_output(self, audio):
    # Preconditions
    assert(PIANO_KEY_FREQUENCIES)

    self.audio_len = len(audio)
    print(self.audio_len)
    self.SDTFBins = [SDFTBin(freq, self.sample_rate) for freq in PIANO_KEY_FREQUENCIES]
    self.key_freq_through_time = [[] for i in range(len(PIANO_KEY_FREQUENCIES))]
    self.reconstructed_audio = [[] for i in range(len(PIANO_KEY_FREQUENCIES))]
    assert(len(self.key_freq_through_time) == len(self.SDTFBins))

    for i, bin in enumerate(self.SDTFBins):
      print(f'Parsing audio file for key {i+1}')
      X_k, x_n = bin.parse(audio)
      self.key_freq_through_time[i] = X_k # X_k[n] for this specific key
      self.reconstructed_audio[i] = x_n # x[n] for this specific key

    # Also generate transposed versions of both matrices
    self.key_freq_through_time_T = np.transpose(self.key_freq_through_time)
    self.reconstructed_audio_T = np.transpose(self.reconstructed_audio)
    print(np.shape(self.key_freq_through_time))


  def plot_frequencies_through_time(self):
    assert(self.key_freq_through_time)
    assert(self.audio_len)
    assert(self.sample_window)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # TODO: Parrallelize this code block
    for n in range(len(self.key_freq_through_time_T)):
      freqs_at_n = np.abs(self.key_freq_through_time_T[n])
      Y = PIANO_KEY_FREQUENCIES
      Z = freqs_at_n
      X = np.full(len(PIANO_KEY_FREQUENCIES), n * (1/self.play_rate))
      ax.plot(X,Y,Z)

    # Labels
    ax.set(title="Change in Freqeuncy Across Time Using SDFT", xlabel=r'Time $t$ [s]', ylabel=r'Frequency $\omega$ [Hz]')
    ax.set_zlabel("Amplitude")
    ax.tick_params(axis='z', which='major', pad=-3)
    
    return plt.show()

  def generate_output_wav_file(self):
    '''Generates an output wav file using the piano using the reconstructed
    audio
    
    ### Implementation Details
    
    Wav files require a minimum sample rate of 3000 Hz, and our ears require the
    audio from the reconstructed samples to persist for some time — because of
    this, we're multiplying the signal at time p by a decaying exponential that
    will carry the sound into the next time sample'''
    #TODO: Implement this function
    pass


  def plot_reconstructed_audio(self):
    assert(self.reconstructed_audio)

    fig = plt.figure()
    ax = fig.add_subplot()

    x_n = []

    for n in range(len(self.reconstructed_audio_T)):
      audio_at_n = self.reconstructed_audio_T[n]
      x_n.append(sum(audio_at_n))

    X = np.arange(0, len(x_n)*(1/self.play_rate), (1/self.play_rate))
    Y = np.abs(x_n)
    ax.plot(X,Y)

    # Labels
    ax.set(title="Reconstructed Audio", xlabel=r'Time $t$ [s]', ylabel=r'Amplitude')

    return plt.show()


  def generate_tsv(self, uuid=uuid.uuid4(), amplitude=MAGNITUDE_MAX):
    '''Generates a text file containing what keys to play, returns unique id
    for given recording.'''

    # Preconditions
    assert(self.key_freq_through_time)
    assert(TSV_HEADERS)
    assert(self.audio_len)
    assert(self.sample_window)

    # Generate a unique id for this audio recording
    id = str(uuid)
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
