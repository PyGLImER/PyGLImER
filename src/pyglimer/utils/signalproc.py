#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Sunday, 20th October 2019 10:31:03 am
Last Modified: Wednesday, 13th April 2022 02:55:06 pm
'''
import warnings

import numpy as np
from obspy import Stream, Trace


def resample_or_decimate(
        data: Stream, sampling_rate_new: int, filter=True) -> Stream:
    """
    Decimates the data if the desired new sampling rate allows to do so.
    Else the signal will be interpolated (a lot slower).

    :note: The stream has to be filtered a priori to avoid aliasing.

    :param data: Stream to be resampled.
    :type data: Stream
    :param sampling_rate_new: The desired new sampling rate
    :type sampling_rate_new: int
    :return: The resampled stream
    :rtype: Stream
    """
    if isinstance(data, Stream):
        sr = data[0].stats.sampling_rate
        srl = [tr.stats.sampling_rate for tr in data]
        if len(set(srl)) != 1:
            # differing sampling rates in stream
            for tr in data:
                try:
                    tr = resample_or_decimate(tr, sampling_rate_new, filter)
                except ValueError:
                    warnings.warn(
                        f'Trace {tr} not downsampled. Sampling rate is lower'
                        + ' than requested sampling rate.')
            return data
    elif isinstance(data, Trace):
        sr = data.stats.sampling_rate
    else:
        raise TypeError('Data has to be an obspy Stream or Trace.')
    srn = sampling_rate_new

    if srn > sr:
        raise ValueError('New sampling rate greater than old. This function \
            is only intended for downsampling.')
    elif srn == sr:
        return data

    # Chosen this filter design as it's exactly the same as
    # obspy.Stream.decimate uses
    # Just for RFs to avoid instabilities
    if filter and srn == 10:
        data.filter('lowpass_cheby_2', freq=4, maxorder=12)
    elif filter:
        freq = sr * 0.5 / float(sr/srn)
        data.filter('lowpass_cheby_2', freq=freq, maxorder=12)

    if sr/srn == sr//srn:
        return data.decimate(int(sr//srn), no_filter=True)
    else:
        return data.resample(srn)


def convf(u: np.ndarray, v: np.ndarray, nf: int, dt: float) -> np.ndarray:
    """
    Convolution conducted in the frequency domain.

    Parameters
    ----------
    u : np.array
        Array 1.
    v : np.array
        Array 2.
    nf : INTEGER
        Array length in frequency domain (use next power of 2).
    dt : FLOAT
        Sampling Interval [s].

    Returns
    -------
    c : np.array
        Convolution of u with v.

    """
    if not len(u) or not len(v):
        raise ValueError('The input arrays have to have a length!')
    U = np.fft.fft(u, n=nf)
    V = np.fft.fft(v, n=nf)
    C = U*V*dt
    c = np.real(np.fft.ifft(C, n=nf))
    return c


def corrf(u: np.ndarray, v: np.ndarray, nf: int) -> np.ndarray:
    """
    Cross-correlation in frequency domain,
    calculates the x=crosscorr(u,v). Hence, v is the flipped vector.

    Parameters
    ----------
    u : np.array
        flipped vector.
    v : np.array
        Vector that the correlation is measured to.
    nf : INTEGER
        Array length in frequency domain (use next power of 2).

    Returns
    -------
    x : np.array
        Correlation vector.

    """
    if not len(u) or not len(v):
        raise ValueError('The input arrays have to have a length!')

    V = np.conj(np.fft.fft(v, n=nf))
    U = np.fft.fft(u, n=nf)
    X = U*V
    x = np.real(np.fft.ifft(X, n=nf))
    return x


def gaussian(N: int, dt: float, width: float) -> np.ndarray:
    """
    Create a zero-phase Gaussian function (i.e., a low-pass filter). In
    particular meant to be convolved with the impulse
    response that is the output of an iterative deconvolution

    Parameters
    ----------
    N : INTEGER
        Length of the desired array.
    dt : FLOAT
        Sampling interval [s].
    width : FLOAT
        Width parameter of the Gaussian function.

    Returns
    -------
    G : np.array
        Gaussian window.

    """
    if width <= 0:
        raise ValueError('Gaussian width parameter has to be greater than 0.')

    df = 1/(N*dt)  # frequency step
    f = np.arange(0, round(0.5*N), 1, dtype=float)*df  # frequency array
    w = 2*np.pi*f  # angular frequency

    G = np.array([0]*N, dtype=float)
    G[0:round(N/2)] = np.exp(-w**2/(4*width**2))/dt
    G_lr = np.flip(G)
    G[round(N/2)+1:] = G_lr[-len(G[round(N/2)+1:]):]
    return G


def filter(s: np.ndarray, F: np.ndarray, dt: float) -> np.ndarray:
    """
    Convolves a filter with a signal (given in time domain).

    Parameters
    ----------
    s : np.array
        Signal given in time domain.
    F : np.array
        Filter's amplitude response (i.e., frequency domain).
    dt : FLOAT
        Sampling interval [s].


    Returns
    -------
    s_f : np.array
        Filtered signal.

    """
    nf = len(F)

    S = np.fft.fft(s, n=nf)
    S_f = np.multiply(S, F)*dt
    s_f = np.real(np.fft.ifft(S_f, n=nf))
    return s_f


def ricker(sigma: float, N2: int, dt: float) -> tuple:
    """ create a zero-phase Ricker / Mexican hat wavelet

    Parameters
    ----------
    sigma : float
        standard deviation
    dt : float
        sampling Interval
    N2 : int
        number of samples/2 (halflength). Full length = 2*N2+1
    """
    rick_tt = np.arange(-N2, N2+1, 1, dtype=float)*dt  # time vector
    rick = 2/(np.sqrt(3*sigma)*np.pi**(1/4))*(1-np.power(rick_tt, 2)/sigma**2)\
        * np.exp(-rick_tt**2/(2*sigma**2))
    return rick_tt, rick


def noise(N: int, A: float) -> np.ndarray:
    """ create random noise
    Parameters
    ----------
    N: int
        signal length
    A: float
        maximal amplitude
    """
    noise = np.random.rand(N)*A
    return noise


def sshift(s: np.ndarray, N2: int, dt: float, shift: float) -> np.ndarray:
    """ shift a signal by a given time-shift in the frequency domain

    Parameters
    ----------
    s : Arraylike
        signal
    N2 : int
        length of the signal in f-domain (usually equals the next pof 2)
    dt : float
        sampling interval
    shift : float
        the time shift in seconds

    Returns
    -------
    Timeshifted signal"""

    S = np.fft.fft(s, n=N2)

    # Omega
    phshift = np.exp(-1j*shift*np.fft.fftfreq(N2, dt)*2*np.pi)
    s_out = np.real(np.fft.ifft(phshift*S))
    return s_out[:len(s)]
