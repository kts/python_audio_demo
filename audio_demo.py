"""
Some demos for audio analysis, processing.

NOTE:
Most of these are little hacks just
to give an idea of how to do things.
"""
import math
import os
import tempfile
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

###
### Utilities for wav files.
###

def read(filename, dtype='float64'):
    sr,x = scipy.io.wavfile.read(filename)
    return np.array(x, dtype=dtype), sr

def write(x,sr,filename):
    """
    """
    ## ?? wasnt working with loaded wav...
    x = np.float64(x) / np.abs(x.max())

    # lets subtract the mean:
    x = x - x.mean()
    
    xx = np.int16( (32767 * x / np.abs(x).max()) )
    scipy.io.wavfile.write(filename, sr, xx)

def play(x,sr):
    """
    hack...
    """
    write(x,sr,'/tmp/tmp.wav')
    os.system("aplay /tmp/tmp.wav")


############
############
############


def tone(freq,dur,sr, phi=0):
    """
    generate a sine wave
    y = sin(wt) = sin(2 pi f t)

    phi: phase offset. (can give 'rand')
    y = sin(2 pi f t + phi)
    """
    t = np.arange(start=0, stop=dur, step=1.0/sr)

    if phi == 'rand':
        phi = 2 * np.pi * np.random.rand()
    
    return np.sin(2*np.pi*freq*t + phi)


def triad(f0,dur,sr):
    """
    Creates a major triad of pure tones.
    """
    x1 = tone(f0, dur, sr)
    x2 = tone(halfsteps(f0, 4), dur, sr)
    x3 = tone(halfsteps(f0, 7), dur, sr)
    return x1+x2+x3


def halfsteps(f, n):
    """
    For a frequency, f, return the freq n half-steps away
    """
    # r = ratio between halfsteps: (constant)
    r = math.pow(2, 1.0/12)
    return f * (r**n)

def spectrum(x,sr):
    """
    compute frequency spectrum of x (sample rate, sr)
    returns (X,f).  Both len N/2 arrays, f holds frequencies (Hz) of points
    """    
    N = len(x)
    X = np.abs(np.fft.fft(x))
    f = np.linspace(0, sr, N+1)[:N]

    Nout = N/2 + 1
    return X[:Nout], f[:Nout]




def spectrogram(x,sr):
   """
   Input waveform, x, and sample rate, sr.

   Output X, is matrix shape (num_frames, num_freq_points)
   the log-magnitude of the short-time-fourier transform.
   """
   ## Parameters: 10ms step, 30ms window
   Ts, Tw = 0.01, 0.03

   nstep = int(sr * Ts)
   nwin  = int(sr * Tw)
   nfft = nwin

   window = np.hamming(nwin)

   ## will take windows x[n1:n2].  generate
   ## and loop over n2 such that all frames
   ## fit within the waveform
   nn = range(nwin, len(x), nstep)

   X = np.zeros( (len(nn), nfft/2) )

   for i,n in enumerate(nn):
      xseg = x[n-nwin:n]
      z = np.fft.fft(window * xseg, nfft)
      X[i,:] = np.log(np.abs(z[:nfft/2]))

   return X

def display_spectrogram(X):
    plt.imshow(X.T, interpolation='nearest',
               origin='lower',
               aspect='auto')
    ax = plt.gca()
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")


def demo():
    """
    some examples, correspond
    to the slides...
    """
    sr        = 16000 # choose a sampling rate
    frequency = 440
    duration  = 1.0   # one second
    t = np.arange(start=0, stop=duration, step=1.0/sr)  # time (sec) of the samples
    omega = 2*np.pi*frequency  # Hz to radians
    x = np.sin(omega * t)

    ## shortcut:
    newax = lambda : plt.figure().add_subplot(111)

    ax = newax()
    ax.plot(t[:500], x[:500])

    #x = tone(440, 1, sr)
    #f0 = 440
    #dur = 1
    #x1 = tone(f0, dur, sr)
    #x2 = tone(halfsteps(f0, 4), dur, sr)
    #x3 = tone(halfsteps(f0, 7), dur, sr)


    #### FFT
    ax = newax()
    
    sr = 1000 
    x = triad(60, 0.2, sr) 

    X,f = spectrum(x,sr)
    ax.plot(f,X)

    ## say we want the amplitude @ 60 hz.
    freq = 60

    t = np.arange(len(x)) / 1000.0

    s1 = np.cos(2*np.pi*freq*t)
    s2 = np.sin(2*np.pi*freq*t)

    Xr =   (s1 * x).sum() # or Xr = np.inner(s1,x)
    Xi = - (s2 * x).sum() # negative sign not really needed...

    mag = np.sqrt(Xr**2 + Xi**2)

    ax.plot(freq, mag, 'ro')

    ax.set_title("Spectrum of the sum of three tones")
    ax.set_xlabel("Frequency (Hz)")

    plt.show()

if __name__ == "__main__":
    demo()
