# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
try:
    from enum import Enum
except ImportError:
    print("enum only exists in Python 3.4 or newer")

try:
    class Window(Enum):
        BARTLETT = 1
        BLACKMAN = 2
        HAMMING  = 3
        HANNING  = 4
except NameError:
    class Window:
        BARTLETT = 1
        BLACKMAN = 2
        HAMMING  = 3
        HANNING  = 4

RANGEIDX = 0
DOPPLERIDX = 1
PEAKVAL = 2

MAX_OBJ_OUT = 100

def WRTFT(data):
    square = []
    coeff = []
    # square sum
    for index in range(data.shape[1]):
        s = 0
        for i in data[:,index]:
            s = s + i**2
        square.append(s)
    square = np.array(square)
    # coefficient weight
    s = np.sum(square)
    for i in square:
        c = 0
        c = i / s
        coeff.append(c)
    coeff = np.array(coeff)
    # W
    WRTFT = np.dot(data, coeff)
    return WRTFT

def clutter_remove(range_matrix):
    for range_idx in range(range_matrix.shape[1]):
        range_avg = np.mean(range_matrix[:, range_idx])
#         print(range_idx, range_avg)
        range_matrix[:, range_idx] = range_matrix[:, range_idx] - range_avg
    return range_matrix

def unwrap_phase_Ambiguity(phase_queue, points_num):
    # for range_idx in range(0, points_num):
    phase_arr = phase_queue #[:, range_idx]
    phase_arr_ret = phase_arr.copy()
    phase_diff_correction_cum = 0
    for i in range(len(phase_arr)):
        if not i:
            continue
        else:
            phase_diff = phase_arr[i] - phase_arr[i - 1]
            # print(len(phase_diff))
            if phase_diff > 180:
                mod_factor = 1
            elif phase_diff < -180:
                mod_factor = -1
            else:
                mod_factor = 0
        phase_diff_mod = phase_diff - mod_factor * 2 * 180
        if phase_diff_mod == -180 and phase_diff > 0:
            phase_diff_mod = 180
        phase_diff_correction = phase_diff_mod - phase_diff
        if (phase_diff_correction < 180 and phase_diff_correction > 0) \
                or (phase_diff_correction > -180 and phase_diff_correction < 0):
            phase_diff_correction = 0
        phase_diff_correction_cum += phase_diff_correction
        phase_arr_ret[i] = phase_arr[i] + phase_diff_correction_cum
    # phase_queue[:, range_idx] = phase_arr_ret
        phase_queue = phase_arr_ret
    return phase_queue

def hampel(X):
    length = X.shape[0] - 1
    k = 3
    nsigma = 3
    iLo = np.array([i - k for i in range(0, length + 1)])
    iHi = np.array([i + k for i in range(0, length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > length] = length
    xmad = []
    xmedian = []
    for i in range(length + 1):
        w = X[iLo[i]:iHi[i] + 1]
        medj = np.median(w)
        mad = np.median(np.abs(w - medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad = np.array(xmad)
    xmedian = np.array(xmedian)
    scale = 1  # 缩放
    xsigma = scale * xmad
    xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # Find outliers (ie more than nsigma standard deviations)

    # Replace outliers with median values
    xf = X.copy()
    xf[xi] = xmedian[xi]
    return xf

def windowing(input, window_type, axis=0):
    """Window the input based on given window type.

    Args:
        input: input numpy array to be windowed.

        window_type: enum chosen between Bartlett, Blackman, Hamming, Hanning and Kaiser.

        axis: the axis along which the windowing will be applied.
    
    Returns:

    """
    window_length = input.shape[axis]
    if window_type == Window.BARTLETT:
        window = np.bartlett(window_length)
    elif window_type == Window.BLACKMAN:
        window = np.blackman(window_length)
    elif window_type == Window.HAMMING:
        window = np.hamming(window_length)
    elif window_type == Window.HANNING:
        window = np.hanning(window_length)
    else:
        raise ValueError("The specified window is not supported!!!")

    output = input * window

    return output

def XYestimation(azimuthMagSqr,
                 numAngleBins,
                 detObj2D):
    """Given the phase information from 3D FFT, calculate the XY position of the objects and populate the detObj2D array.
  
    Args:
        azimuthMagSqr: (numDetObj, numAngleBins) Magnitude square of the 3D FFT output.
        numAngelBins: hardcoded as 64 in our project.
        detObj2D: Output yet to be populated with the calculated X, Y and Z information
    """
    if extendedMaxVelocityEnabled and numVirtualAntAzim > numRxAntennas:
        azimuthMagSqrCopy = azimuthMagSqr
    else:
        azimuthMagSqrCopy = azimuthMagSqr[:, :numAngleBins]
  
    maxIdx = np.argmax(azimuthMagSqrCopy, axis=1)
    maxVal = azimuthMagSqrCopy[np.arange(azimuthMagSqrCopy.shape[0]), maxIdx]
  
    if extendedMaxVelocityEnabled and numVirtualAntAzim > numRxAntennas:
        maxIdx[maxIdx[:] > numAngleBins] -= numAngleBins

    # Due to the simplicity of the python implementatoin, MmwDemo_XYcalc is merged into here.
    # ONE_QFORRMAT is used to converting for TLV. Not used here.
    # ONE_QFORRMAT = math.ceil( math.log10(16./rangeResolution) / math.log10(2) )
    detObj2D[:, PEAKVAL] = np.sqrt(maxVal / (numRangeBins*numAngleBins*numDopplerBins))
    
    rangeInMeter = detObj2D[:, RANGEIDX] * rangeResolution
    rangeInMeter -= compRxChanCfg.rangeBias
    rangeInMeter = np.maximum(rangeInMeter-compRxChanCfg.rangeBias, 0)
    
    sMaxIdx = maxIdx
    sMaxIdx[maxIdx[:] > (numAngleBins/2-1)] -= numAngleBins
    
    Wx = 2 * sMaxIdx.astype(np.float32) / numAngleBins
    x = rangeInMeter * Wx
    y = np.maximum(np.sqrt(rangeInMeter**2 - x**2), 0)
    detObj2DAzim = np.hstack((detObj2D, np.expand_dims(x,1), np.expand_dims(y,1)))
    # Seems like we don't have mutlibeamforming here so not going ot implement the second part.

    return detObj2DAzim
