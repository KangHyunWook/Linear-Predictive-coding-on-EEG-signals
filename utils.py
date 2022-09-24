from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit

import scipy, scipy.signal
from numpy.polynomial import polynomial as P


import numpy as np
import pandas as pd
import keras.backend as K

def lpc_to_lsp(lpc):
    """Convert LPC to line spectral pairs"""
    l = len(lpc) + 1
    a = np.zeros((l,))
    a[0:-1] = lpc
    p = np.zeros((l,))
    q = np.zeros((l,))
    for i in range(l):
        j = l - i - 1
        p[i] = a[i] + a[j]
        q[i] = a[i] - a[j]

    ps = np.sort(np.angle(np.roots(p)))
    qs = np.sort(np.angle(np.roots(q)))
    lsp = np.vstack([ps[: len(ps) // 2], qs[: len(qs) // 2]]).T
    return lsp


def lsp_to_lpc(lsp):
    """Convert line spectral pairs to LPC"""
    ps = np.concatenate((lsp[:, 0], -lsp[::-1, 0], [np.pi]))
    qs = np.concatenate((lsp[:, 1], [0], -lsp[::-1, 1]))

    p = np.cos(ps) - np.sin(ps) * 1.0j
    q = np.cos(qs) - np.sin(qs) * 1.0j

    p = np.real(P.polyfromroots(p))
    q = -np.real(P.polyfromroots(q))

    a = 0.5 * (p + q)
    return a[:-1]

def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.
    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.
    Notes
    ----
    This implementation is in python, hence unsuitable for any serious
    computation. Use it as educational and reference purpose only.
    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:
                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1 / r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order + 1, r.dtype)
    # temporary array
    t = np.empty(order + 1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k[i - 1] = -acc / e
        a[i] = k[i - 1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i - 1] * np.conj(t[i - j])

        e *= 1 - k[i - 1] * np.conj(k[i - 1])

    return a, e, k

def lpc(wave, order):
    """Compute LPC of the waveform. 
    a: the LPC coefficients
    e: the total error
    k: the reflection coefficients
    
    Typically only a is required.
    """
    # only use right half of autocorrelation, normalised by total length
    autocorr = scipy.signal.correlate(wave, wave)[len(wave) - 1 :] / len(wave)
    a, e, k = levinson_1d(autocorr, order)
    return a, e, k

def lpc_vocode(
    wave,
    frame_len,
    order,
    carrier,
    residual_amp=0.0,
    vocode_amp=1.0,
    env=False,
    freq_shift=1.0,
):
    """
    Apply LPC vocoding to a pair of signals using 50% overlap-add Hamming window resynthesis
    The modulator `wave` is applied to the carrier `imposed`
    
    Parameters:
    ---
    wave: modulator wave
    frame_len: length of frames
    order: LPC order (typically 2-30)
    carrier: carrier signal; should be at least as long as wave
    residual_amp: amplitude of LPC residual to include in output
    vocode_amp: amplitude of vocoded signal 
    env: if True, the original volume envelope of wave is imposed on the output
          otherwise, no volume modulation is applied
    freq_shift: (default 1.0) shift the frequency of the resonances by the given scale factor. Warning :
        values >1.1 are usually unstable, and values <0.5 likewise.
    """

    # precompute the hamming window
    window = scipy.signal.hann(frame_len)
    t = np.arange(frame_len)
    # allocate the array for the output
    vocode = np.zeros(len(wave + frame_len))
    last = np.zeros(order)
    # 50% window steps for overlap-add
    for i in range(0, len(wave), frame_len // 2):
        # slice the wave
        wave_slice = wave[i : i + frame_len]
        carrier_slice = carrier[i : i + frame_len]
        if len(wave_slice) == frame_len:
            # compute LPC
            a, error, reflection = lpc(wave_slice, order)

            # apply shifting in LSP space
            lsp = lpc_to_lsp(a)
            lsp = (lsp * freq_shift + np.pi) % (np.pi) - np.pi
            a = lsp_to_lpc(lsp)

            # compute the LPC residual
            residual = scipy.signal.lfilter(a, 1.0, wave_slice)
            # filter, using LPC as the *IIR* component
            # vocoded, last = scipy.signal.lfilter([1.], a, carrier_slice, zi=last)
            vocoded = scipy.signal.lfilter([1.0], a, carrier_slice)

            # match RMS of original signal
            if env:
                voc_amp = 1e-5 + np.sqrt(np.mean(vocoded ** 2))
                wave_amp = 1e-5 + np.sqrt(np.mean(wave_slice ** 2))
                vocoded = vocoded * (wave_amp / voc_amp)

            # Hann window 50%-overlap-add to remove clicking
            vocode[i : i + frame_len] += (
                vocoded * vocode_amp + residual * residual_amp
            ) * window

    return vocode[: len(wave)]

def getModulatedData(data):

    window=300
    order=5
    carrier = np.random.normal(0,1,len(data[0]))
    modulated_features=[]
    for d in data:
        modulated = lpc_vocode(d, frame_len=window, order=order,
                    carrier=carrier, residual_amp=0, vocode_amp=1, env=True, freq_shift=1)
                    
        modulated_features.append(modulated)
    
    return np.array(modulated_features)

def loadData(args):
    pd_data = pd.read_csv(args.data_path)
    scaler = MinMaxScaler()
    data=pd_data.iloc[:,:-1].values
    # data=getModulatedData(data)
    # print('mo:',data.shape)
    # print(data[:10])
    # print(np.max(data),np.min(data))
    # exit()
    
    data=scaler.fit_transform(data)
    if args.model=='1':
        data=data.reshape(data.shape+(1,))
    labels=pd_data.iloc[:,-1].values
   
    if args.data_path=='emotions.csv':
        le=LabelEncoder()
        labels=le.fit_transform(labels)
            
    return data, labels

def splitTrainTestData(data, labels):

    labels=to_categorical(labels)

    xxx=StratifiedShuffleSplit(1, test_size=0.3, random_state=12)

    for train_index, test_index in xxx.split(data, labels):
        trainX, testX = data[train_index], data[test_index]
        trainy, testy = labels[train_index], labels[test_index]

    return trainX, testX, trainy, testy