'''
An implementation of the Sliding Discrete Fourier Transform

Author: Marco Acea
Email: aceamarco@gmail.com, macea@andrew.cmu.edu

------------
Explanation
------------
Given a window (x[n]) of size N containing time series data, traditional DFT returns 
an array of N evenly spaced frequency bins.

DFT(x[n]) = X = {X_1, X_2, ..., X_n}

But what if we only wanted to find X_k for some given window?

Glossing over the "why it works", this is precisely what the Sliding Discrete
Fourier Transform allows us to do.

Let x(n) = {x[n-N], ..., x[n-1], x[n]}
e.g x(15) with N=15 yields x(15) = {x[0], x[1], ..., x[15]}

The frequency component X_k at time x[n] with window x(n) is:

X_k(n) = [X_k(n-1) - x[n-N] + x[n]] (e^((j2pi)/N)) ; (Eq. 1)

--------------------------------------------------
What does this have to do with the sliding piano?
--------------------------------------------------

Well, the frequencies of each piano key are the X_k we'd like to solve for.

In order to figure out which frequency bin we're interested in for each key 
(i.e the 'k' in X_k), we need to find the corresponding window size N_i for each
key such that one of it's frequency bins lands on the key i's frequency.

This to say, for every key, we need two numbers:

- N_i : The DFT window size that produces a frequency bin centered at the 
frequency of key i

- k_i : The index of that frequency bin centered at the frequency of key i

----------------------------------------
Okay...how does this all come together?
----------------------------------------

The system is broken down into 5 parts

1. Find N and k for each key
2. Iterate through each time sample, appending it to the end of each keys
  buffer (of size N_i)
3. With every new sample, update X_k
4. Log the frequencies X_k present at the time stamps we care about, for our 
  player piano, that was at intervals of 15 times per second, i.e every 3428 
  samples.
5. Instead of simply recording X_k at samples {1*3428, 2*3428,...}, we propagate
  information captured within that window by taking a moving average of
  X_k at samples {(n*3428) - N_i, ..., (n*3428) - 1, (n*3428)}

----------------------------------------
An implementation note.
----------------------------------------

The sliding DFT assumes we have X_{k-1}, in order to compute X_k for the first
N-many time instances — the samples are shifted into a window buffer, all
intialized to 0.

Example: N = 10

t | W (window buffer)
0 | [0 0 0 0 0 0 0   0    0    0 ]: X_k = 0
1 | [0 0 0 0 0 0 0   0    0  x[1]]
2 | [0 0 0 0 0 0 0   0  x[1] x[2]]
3 | [0 0 0 0 0 0 0 x[1] x[2] x[3]]

----------------------------------------
Acronyms (An unfortunate product of DSP)
----------------------------------------
- DFT : Discrete Fourier Transform
- SDFT : Sliding Discrete Fourier Transform
- SMA : Simple Moving Average

Open an issue if you find any acronyms I haven't expanded out here!
'''
#############
## Imports ##
#############

from math import e, pi


###############
## Constants ##
###############
PLAY_RATE  = 15
SMA_WINDOW = 3200
SAMPLE_RATE = 48000
MAGNITUDE_MAX = 5 * (10**6)


########################################
## Sliding Discrete Fourier Transform ##
########################################

class SDFTBin:

  def __init__(self, note_frequency, sample_rate, play_rate = PLAY_RATE):
    print(f'Creating SDTF Bin at {note_frequency}Hz...')
    self.note_frequency = note_frequency
    self.sample_rate = sample_rate
    self.play_rate = play_rate
    self.find_N_k()
    self.w = [0 for i in range(self.N)]
    self.X_k = 0
    self.n = 0
    self.X_k_MA = MovingAverage()
    self.x_n_MA = MovingAverage()
    print(f'Done. Max window size is {self.N_max}')
    print("----------------------------------------")

  def find_N_k(self):
    '''Returns the smallest window size N that returns a frequency resoultion
      that captures a certain note frequency

      Returns
      - N : The window size
      - k : The index within the frequency bin that we're interested in'''
    print(f'Finding N and k for {self.note_frequency}Hz...')

    self.N_max = self.sample_rate // self.play_rate

    N_min = float('inf')
    error_min = float('inf')
    for N_i in range(1, self.N_max):
      f_r = self.sample_rate / N_i
      error = self.note_frequency % f_r
      if error < error_min:
        N_min = N_i
        error_min = error

    self.N = N_min
    self.effective_error = error_min

    self.effective_frequency = self.note_frequency - (error_min)
    self.effective_bandwidth = self.sample_rate // self.N

    self.k = int(self.effective_frequency // self.effective_bandwidth)
    assert(self.k < self.N)
    print(f'Done! Found N = {self.N} and k = {self.k}')

  def update_x_n(self, prev_X_k, curr_X_k, bottom_window_sample):
    '''
    Returns x[n] using the formula for X_k

    ### Parameters
    - prev_X_k : X_{k-1}[n]
    - curr_X_k : X_k[n]
    - bottom_window_sample : x[n-N]

    ### Returns
    - x_n : x[n] using Eq. (1) from file descirption
    '''
    x_n = (curr_X_k * (e**(-1j * 2*pi * (self.k / self.N)))) - prev_X_k + bottom_window_sample

    self.x_n_MA.update(x_n)
      
  def update(self, x_n):
    '''Calculates the latest X_k given a new sample
      X_k(n) = [X_k(n-1) - x[n-N] + x[n]] (e^((j2pi)/N))
    '''

    # Calculate new X_k, store previous values for our reconstruction of x[n]
    prev_X_k = self.X_k
    self.X_k = (self.X_k - self.w[0] + x_n) * (e**((1j * 2*pi * self.k)/self.N))
    bottom_window_sample = self.w.pop(0)

    # Calculate x[n], this function handles updating x[n]'s SMA
    self.update_x_n(prev_X_k, self.X_k, bottom_window_sample)

    # Wrapping up
    self.w.append(x_n)
    self.n += 1
    self.X_k_MA.update(self.X_k)


  def parse(self, x):
    '''
    Parses an array respresnting all time-series samples and returns an array
    corresponding to X_k sampled at the piano rate.

    Note: The incoming samples must have been recorded at this bins set sample 
    rate
    '''
    print(f'Parsing input audio file of length {len(x)}')
    X_k = []
    x_n = []
    for n, sample_n in enumerate(x):
      self.update(sample_n)
      if (n % self.N_max) == 0:
        X_k.append(self.X_k_MA.SMA)
        x_n.append(self.x_n_MA.SMA)
    print(f'Done! With a play rate of {self.play_rate}, we expect an output of size {len(x) // self.N_max}, and got {len(X_k)-1}')    
    return X_k, x_n
        


####################
## Moving Average ##
####################

class MovingAverage:
  '''
  Simple Moving Average Class
  '''

  def __init__(self, N=SMA_WINDOW):
    self.N = N
    self.W = [0 for i in range(self.N)]
    self.SMA = 0

  def update(self, new_val):
    self.SMA = self.SMA - (self.W[0]/self.N) + (new_val/self.N)
    self.W.pop(0)
    self.W.append(new_val)

