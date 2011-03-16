"""
Compute and display a spectrogram.
Give WAV file as input
"""
import matplotlib.pyplot as plt
import numpy as np
import sys

from audio_demo import spectrogram, display_spectrogram, read

if __name__ == "__main__":

   if len(sys.argv) != 2:
      print "Usage: python spectrogram.py wavfile.wav"
      sys.exit(1)
      
   x,sr = read(sys.argv[1])

   X = spectrogram(x,sr)
   display_spectrogram(X)
      
   plt.show()
