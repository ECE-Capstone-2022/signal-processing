'''
Author: Marco Acea
Andrew ID: macea
Contact: macea@andrew.cmu.edu / aceamarco@gmail.com

This file generates test vectors for John's UI piano playback feature

'''

import numpy as np

def identity_matrix():
  '''Returns an identity matrix that should press every key'''

  M = [[0 for k in range(88)] for n in range(88)]

  print(np.shape(M))

  for i in range(88):
    M[i][i] = 1

  return M
