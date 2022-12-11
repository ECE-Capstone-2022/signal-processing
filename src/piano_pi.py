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

from SDFT import SDFTBin, PLAY_RATE, SAMPLE_RATE, MAGNITUDE_MAX
from multiprocessing import Process, Value, Array
from scipy.io import wavfile
from dbg import *
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import getopt
import uuid
import csv

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


###############
## Constants ##
###############

# Audio Reconstruction
# TODO: See comment below
'''
0.001 and 0.0001 produce intelligible outputs, however this is a decay given
to every note, from what I understand each note has a varying decay
'''
DECAY_EXP = 0.0001

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

  def __init__(self, file_path, uuid = uuid.uuid4(), play_rate=PLAY_RATE):
    self.file_path = file_path
    self.sample_rate, self.audio_time_series = wavfile.read(self.file_path)
    if len(np.shape(self.audio_time_series)) != 1:
      self.audio_time_series = self.audio_time_series[:,0]
    dbg_print(np.shape(self.audio_time_series))
    self.play_rate = play_rate
    self.sample_window = self.sample_rate // play_rate
    self.uuid = uuid

    self.generate_output()

  
  def generate_output(self):
    # Preconditions
    dbg_assert(PIANO_KEY_FREQUENCIES)
    # dbg_assert(self.audio_time_series) TODO: Check for existence but not working

    self.audio_len = len(self.audio_time_series)
    self.audio = self.audio_time_series
    self.sign = (self.audio_time_series / np.abs(self.audio_time_series)).astype(int)

    dbg_print(self.audio_len)

    self.SDFTBins = [SDFTBin(freq, self.sample_rate) for freq in PIANO_KEY_FREQUENCIES]
    self.key_freq_through_time = [[] for i in range(len(PIANO_KEY_FREQUENCIES))]
    self.reconstructed_audio = [[] for i in range(len(PIANO_KEY_FREQUENCIES))]
    dbg_assert(len(self.key_freq_through_time) == len(self.SDFTBins))

    # TODO: Parrallelize this code block
    for i, bin in enumerate(self.SDFTBins):
      dbg_print(f'Parsing audio file for key {i+1}')
      X_k, x_n = bin.parse(self.audio_time_series)
      self.key_freq_through_time[i] = X_k # X_k[n] for this specific key
      self.reconstructed_audio[i] = x_n # x[n] for this specific key

    # Also generate transposed versions of both matrices
    self.key_freq_through_time_T = np.transpose(self.key_freq_through_time)
    self.reconstructed_audio_T = np.transpose(self.reconstructed_audio)
    dbg_print(np.shape(self.key_freq_through_time))


  def plot_freq_through_time(self):
    dbg_assert(self.key_freq_through_time)
    dbg_assert(self.audio_len)
    dbg_assert(self.sample_window)

    D = {
      "X" : [],
      "Y" : [],
      "Z" : [],
      "color" : [],
    }

    for n in range(len(self.key_freq_through_time_T)):
      freqs_at_n = np.abs(self.key_freq_through_time_T[n])
      D["Y"].extend(PIANO_KEY_FREQUENCIES)
      D["Z"].extend(freqs_at_n)
      D["X"].extend(np.full(len(PIANO_KEY_FREQUENCIES), n * (1/self.play_rate)))
      D["color"].extend([n]*len(freqs_at_n))

    # tight layout
    df = pd.DataFrame(D)
    fig = px.line_3d(df, x="X", y="Y", z="Z", 
                     color="color",
                     labels={
                        "X": "Time t [s]",
                        "Y": "Frequency \u03C9 [Hz]",
                        "Z": "Amplitude",
                        "color" : "Piano Sample [n]"
                    },
                     title="Change in Frequencies Through Time Using the SDFT")

    # Generate File Path
    file_path = f'out/{self.uuid}/plots/html'

    if not os.path.exists(file_path):
      os.makedirs(file_path)
    file_path = file_path + f'/{self.uuid}_3d_frequencies.html'

    fig.write_html(file_path)
    
    if (DEBUG): 
      fig.show()


  def generate_output_wav_file(self):
    '''
    Generates an output wav file and plot using the piano using the reconstructed
    audio signal
    
    ### Implementation Details
    
    Wav files require a minimum sample rate of 3000 Hz, and our ears require the
    audio from the reconstructed samples to persist for some time — because of
    this, we're multiplying the signal at time p by a decaying exponential that
    will carry the sound into the next time sample'''

    ######################################
    ## Build Reconstructed Audio Signal ##
    ######################################

    x_n = []

    for n in range(len(self.reconstructed_audio_T)):
      audio_at_n = self.reconstructed_audio_T[n]

      diff = ((n+1) * self.sample_window) - (self.audio.size)

      if diff > 0:
        n_range = np.arange(self.sample_window - diff)
      else:
        n_range = np.arange(self.sample_window)

      decay = np.exp(-1*n_range * DECAY_EXP)
      A = np.sum(audio_at_n)
      
      x_n.extend(A*decay)

    Y = np.abs(x_n) * self.sign
    X = np.arange(0, len(Y)*(1/self.play_rate), (1/self.play_rate))

    ########################
    ## Generate .wav file ##
    ########################

    # Generate a file path for this audio recording
    file_path = f'out/{self.uuid}/audio'

    if not os.path.exists(file_path):
      os.makedirs(file_path)
    file_path = file_path + f'/{self.uuid}.wav'

    wavfile.write(file_path, self.sample_rate, Y.astype(np.int16))

    ##########################################
    ## Generate plot of reconstructed audio ##
    ##########################################

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(X,Y)

    #Generate a file path for the plot image
    file_path = f'out/{self.uuid}/plots/png'

    if not os.path.exists(file_path):
      os.makedirs(file_path)
    file_path = file_path + f'/{self.uuid}_reconstructed_audio.png'

    plt.savefig(file_path)

    if (DEBUG) :
      plt.show()


  def generate_tsv(self, amplitude=MAGNITUDE_MAX):
    '''Generates a text file containing what keys to play, returns unique id
    for given recording.'''

    # Preconditions
    dbg_assert(self.key_freq_through_time)
    dbg_assert(TSV_HEADERS)
    dbg_assert(self.audio_len)
    dbg_assert(self.sample_window)

    # Generate a unique id for this audio recording
    file_path = f'out/{self.uuid}'

    if not os.path.exists(file_path):
      os.makedirs(file_path)
    file_path = file_path + f'/{self.uuid}.tsv'

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
          tsv_row.append('{0:.2f}'.format((100*(np.abs(key[i]) / amplitude))))
        tsv_writer.writerow(tsv_row)

    out_file.close()

    return file_path
