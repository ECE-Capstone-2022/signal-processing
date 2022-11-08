'''
A signal processing library for the 18-500 F22 ECE Capstone course

Author: Marco Acea
Andrew ID: macea
Contact: macea@andrew.cmu.edu
'''

import os
import numpy as np
from scipy import signal
from scipy.fftpack import fft
from scipy.io import wavfile
import uuid
import csv

# Generate Piano Key Frequencies

# Frequencies corresponding to each piano key
PIANO_KEY_FREQUENCIES = []

# A function of unit impulse functions centered around the piano key frequencies
PIANO_FILTER = []

# Max range of Piano keys to consider for the output file
PIANO_KEY_RANGE = 70

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
    for n in f.read().split('\n'):
      PIANO_KEY_FREQUENCIES.append(float(n))

    X = np.arange(0, 5000, 0.001)
    Y = np.zeros(len(X))

    seen = set()
    for k in PIANO_KEY_FREQUENCIES:
      # TODO: There is a known issue where the top 10 frequencies are not being
      # found. This is not a pressing matter because we're only using the top 69
      # keys. To see this, uncomment the line under 'except'
      try:
        i = np.where(X == k)[0][0]
      except:
        print(f'Key frequency {k} was not found')
        pass
      if i in seen: continue
      seen.add(i)
      impluse = signal.unit_impulse(len(X), i)
      Y += impluse
    PIANO_FILTER = Y
f.close()

# Generating column headers for the tsv outputs
TSV_HEADERS = ['time_stamp']
for i, key_freq in enumerate(PIANO_KEY_FREQUENCIES[:PIANO_KEY_RANGE]):
  TSV_HEADERS.append(f'key{i}_{key_freq}Hz')

def running_average():
  #TODO Implement Running Average
  pass

def comb_filter(frequency_sample) -> np.array:
  return frequency_sample * PIANO_FILTER

def neighbors_average():
  # TODO Implement neighbors average
  pass

PROPAGATION_FNS = {
  'RA' : running_average,
  'CF' : comb_filter,
  'NA' : neighbors_average
}

class PianoPi:

  def __init__(self, play_rate=14, key_range = 70):
    self.piano_keys = PIANO_KEY_FREQUENCIES[key_range]
    self.piano_key_filter = PIANO_KEY_FREQUENCIES
    # Rate at which piano keys can be played, samples/second
    self.play_rate = play_rate
    self.amplitude_max = 5 * (10**6)

  def audio_time_series(self, audio_file_path: str) -> (tuple(int, np.array) or Exception):
    '''Returns a numpy array of an audio recording in the time domain'''
    return wavfile.read(audio_file_path)

  def generate_windows(self, audio: np.array, sample_rate: int) -> np.array(np.array):
    '''Returns a NxM matrix containing sections of the time series array'''

    window_size = sample_rate // self.play_rate

    N = len(audio) # Number of origianl samples
    D = N // window_size # Duration

    res = np.array()

    for t in range (D):
      start = t*self.play_rate
      # 'end' grabs the fraction of samples within the last second of the original 
      # audio if the original audio is not exactly whole seconds long, i.e 5.18 
      # seconds as opposed to 5 seconds
      end = min((t+1) * self.play_rate, len(audio))
      audio_window = audio[start:end,0]
      res = np.append(res, audio_window)

    return res

  def freq_through_time(self, windows):
    '''Returns a NxM matrix of spectral distribution for each window in the
    time doamin'''

    res = np.array()
    for window in windows:
      frequencies = fft(window)
      res = np.append(frequencies)
    
    return res

  def freq_to_piano_keys(self, frequency_samples, avg_technique = 'CF'):
    '''Returns a NxM matrix of every windows frequencies mapped to the keys
     on a piano'''

    if avg_technique not in PROPAGATION_FNS:
      raise "Averaging Function not found"

    avg_fn = PROPAGATION_FNS[avg_technique]
    
    res = np.array()
    for freq_sample in frequency_samples:
      res = np.append(res, avg_fn(freq_sample))

    return res

  def get_duty_cycle(self, amplitude):
    '''Returns a percentage of amplitude relative to maximum amplitude'''
    return int(min(amplitude, self.amplitude_max) / self.amplitude_max)

  def get_time_stamp(self, t):
    '''Returns time stamp corresponding to a time window'''
    return t * (1/self.play_rate)

  def generate_piano_tsv(self, piano_samples) -> str:
    '''Generates a text file containing what keys to play, returns unique id
    for given recording.'''

    # Generate a unique id for this audio recording
    id = str(uuid.uuid4())
    file_path = f'/out/recordings/{id}.tsv'

    # Create the text file named {uuid}.tsv
    with open(file_path,'r') as out_file:
      # Write the column headers
      tsv_writer = csv.writer(out_file, delimiter='\t')
      tsv_writer.writerow(TSV_HEADERS)

      for t, sample in enumerate(piano_samples):
        time_stamp = self.get_time_stamp(t)
        tsv_row = [str(time_stamp)]

        for k in self.piano_keys:
          i = np.where(sample == k)
          tsv_row.append(self.get_duty_cycle(sample[i]))
        
        tsv_writer.writerow(tsv_row)
    out_file.close()

    return file_path

  def process_audio(self, audio_file_path) -> str:
    '''Generates tsv file corresponding to the output of the audio processing
     module. Returns file path to tsv file.'''

    sample_rate, audio_ts = self.audio_time_series(audio_file_path)

    windows = self.generate_windows(audio_ts, sample_rate)

    freq_through_time = self.freq_through_time(windows)

    freq_to_piano_keys = self.freq_to_piano_keys(freq_through_time)

    return self.generate_piano_tsv(freq_to_piano_keys)


    

  