import numpy as np

def find_N(f_p, sample_rate, N_max):
  '''
    Returns the best time window size that generates a frequency bin at f
  '''
  estimated_error = []
  N = []
  for N_i in range(1,N_max+1):
    f_r = sample_rate / N_i;
    error = f_p % f_r
    N.append(N_i)
    estimated_error.append(error)

  min_error = np.amin(estimated_error)
  index_min = np.argmin(estimated_error)
  return (N[index_min], min_error)

n_digits = 5
sample_rate = 48000

with open("../key_frequencies.txt", "r") as f:
  frequencies = []
  for freq in f.read().splitlines():
    frequencies.append(float(freq))
  
  with open("../N_sizes.txt", "w") as N_file:
    for i, freq in enumerate(frequencies):
      N, min_error = find_N(freq, 44100, 3428)
      N_str, error_str = str(N), str(round(min_error, n_digits))
      res = f'{N_str}:{error_str}'
      if (i < len(frequencies)):
        res = res + '\n'
      N_file.write(res)

  N_file.close()
f.close()