import numpy as np
import pandas as pd
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
# from scipy.signal import butter, lfilter, find_peaks, filtfilt, sosfilt
# from time import strftime
# from numpy.lib.function_base import angle
# import compensation
# import utils
# from doppler_processing import doppler_processing
# import mat73
# from cfar import ca_cfar_2d
# from statistics import mode
# import math
from utils import clutter_remove, WRTFT
import winsound


## Get data from file: (radar & references)
print("reading the file")
############################################################################################
# Change to recordings folder in your PC:  (Set once)
root_dir = r"C:\Users\FAN\Desktop\Experiment data"
############################################################################################

############################################################################################
# Change to your recorded file name:   (Modify for every recording)
file_name = r"Top-Yang-1.0-0_08-27_162452.xlsx"
save= "Top-Yang-1.0-0_08-27_162452"
############################################################################################

data_file_path = os.path.join(root_dir, file_name)
recorded_data = pd.ExcelFile(data_file_path)

# ref_hr_data_origin = pd.read_excel(recorded_data, 'polar_hr')
# ref_ecg_data_origin = pd.read_excel(recorded_data, 'polar_ecg')
# ref_rri_data_origin = pd.read_excel(recorded_data, 'polar_rri')
# ref_act_data_origin = pd.read_excel(recorded_data, 'polar_act')

# vernier_data_origin = pd.read_excel(recorded_data, 'vernier')

radar_1dfft_data_origin1 = pd.read_excel(recorded_data, 'ant_0', header=0)
radar_1dfft_data_origin2 = pd.read_excel(recorded_data, 'ant_1', header=0)
radar_1dfft_data_origin3 = pd.read_excel(recorded_data, 'ant_2', header=0)
radar_1dfft_data_origin4 = pd.read_excel(recorded_data, 'ant_3', header=0)
radar_1dfft_data_origin5 = pd.read_excel(recorded_data, 'ant_4', header=0)
radar_1dfft_data_origin6 = pd.read_excel(recorded_data, 'ant_5', header=0)
radar_1dfft_data_origin7 = pd.read_excel(recorded_data, 'ant_6', header=0)
radar_1dfft_data_origin8 = pd.read_excel(recorded_data, 'ant_7', header=0)
radar_1dfft_data_origin9 = pd.read_excel(recorded_data, 'ant_8', header=0)
radar_1dfft_data_origin10 = pd.read_excel(recorded_data, 'ant_9', header=0)
radar_1dfft_data_origin11 = pd.read_excel(recorded_data, 'ant_10', header=0)
radar_1dfft_data_origin12 = pd.read_excel(recorded_data, 'ant_11', header=0)



# Complex error because of the first line is time
radar_1dfft_data_origin1 = np.asarray(radar_1dfft_data_origin1)
radar_1dfft_data_origin2 = np.asarray(radar_1dfft_data_origin2)
radar_1dfft_data_origin3 = np.asarray(radar_1dfft_data_origin3)
radar_1dfft_data_origin4 = np.asarray(radar_1dfft_data_origin4)
radar_1dfft_data_origin5 = np.asarray(radar_1dfft_data_origin5)
radar_1dfft_data_origin6 = np.asarray(radar_1dfft_data_origin6)
radar_1dfft_data_origin7 = np.asarray(radar_1dfft_data_origin7)
radar_1dfft_data_origin8 = np.asarray(radar_1dfft_data_origin8)
radar_1dfft_data_origin9 = np.asarray(radar_1dfft_data_origin9)
radar_1dfft_data_origin10 = np.asarray(radar_1dfft_data_origin10)
radar_1dfft_data_origin11 = np.asarray(radar_1dfft_data_origin11)
radar_1dfft_data_origin12 = np.asarray(radar_1dfft_data_origin12)
radar_1dfft_data_origin1 = np.asarray(radar_1dfft_data_origin1[:,1:], dtype = complex)
radar_1dfft_data_origin2 = np.asarray(radar_1dfft_data_origin2[:,1:], dtype = complex)
radar_1dfft_data_origin3 = np.asarray(radar_1dfft_data_origin3[:,1:], dtype = complex)
radar_1dfft_data_origin4 = np.asarray(radar_1dfft_data_origin4[:,1:], dtype = complex)
radar_1dfft_data_origin5 = np.asarray(radar_1dfft_data_origin5[:,1:], dtype = complex)
radar_1dfft_data_origin6 = np.asarray(radar_1dfft_data_origin6[:,1:], dtype = complex)
radar_1dfft_data_origin7 = np.asarray(radar_1dfft_data_origin7[:,1:], dtype = complex)
radar_1dfft_data_origin8 = np.asarray(radar_1dfft_data_origin8[:,1:], dtype = complex)
radar_1dfft_data_origin9 = np.asarray(radar_1dfft_data_origin9[:,1:], dtype = complex)
radar_1dfft_data_origin10 = np.asarray(radar_1dfft_data_origin10[:,1:], dtype = complex)
radar_1dfft_data_origin11 = np.asarray(radar_1dfft_data_origin11[:,1:], dtype = complex)
radar_1dfft_data_origin12 = np.asarray(radar_1dfft_data_origin12[:,1:], dtype = complex)

radar_1dfft_data_origin = np.array([radar_1dfft_data_origin1, radar_1dfft_data_origin2, radar_1dfft_data_origin3, radar_1dfft_data_origin4, radar_1dfft_data_origin5, radar_1dfft_data_origin6,radar_1dfft_data_origin7
                                    , radar_1dfft_data_origin8, radar_1dfft_data_origin9,radar_1dfft_data_origin10, radar_1dfft_data_origin11,radar_1dfft_data_origin12])
radar_1dfft_data_origin = np.transpose(radar_1dfft_data_origin, axes=(1,0,2))
print(radar_1dfft_data_origin.shape)
print(f"There are {radar_1dfft_data_origin.shape[0]} chirps in the file")

for ante in range(12):
    radar_1dfft_data_origin[:,ante,:] = clutter_remove(radar_1dfft_data_origin[:,ante,:])

print("Declutter finished")

visual_matrix = []
visual_mean = []
WRTFT_matrix = []

print("start calculating")
for chirp in range(radar_1dfft_data_origin.shape[0]-32):
    dopplerinput = radar_1dfft_data_origin[chirp:chirp+32,:,:]
    # print(dopplerinput.shape)
    dopplerinput = np.transpose(dopplerinput, axes=(2,1,0))
    # print(dopplerinput.shape)
    DopplerFFT = np.fft.fft(dopplerinput)
    Dopplerabs = np.abs(DopplerFFT)
    Dopplerout = np.sum(Dopplerabs, axis=1)
    upper = Dopplerout[:, 1:16].T
    lower = Dopplerout[:, 16:32].T
    doppler_out = np.vstack((lower, upper))
    # dopplercfar = ca_cfar_2d(doppler_out)
    # visual_matrix.append(doppler_out)
    WRTFT_r = WRTFT(doppler_out)
    WRTFT_matrix.append(WRTFT_r)

WRTFT_plot = np.vstack((WRTFT_matrix)).T
print(WRTFT_plot.shape)
plt.imshow(WRTFT_plot, aspect='auto')             #extent=[0, 1000/30, 1.8, -1.92],
plt.xlabel("time (s)")
plt.ylabel("Velocity (m/s)")
plt.show()

print("start saving data")
for i in range(300,1000-35):
    plt.imshow(WRTFT_plot[:,i:i+35])
    plt.axis('off')
    i = str(i)
    print(i)
    plt.savefig('./Result/Supine/'+save+'_'+i+'.png',bbox_inches="tight",pad_inches=0)
    plt.clf()

for i in range(1300,2000-35):
    plt.imshow(WRTFT_plot[:,i:i+35])
    plt.axis('off')
    i = str(i)
    plt.savefig('./Result/Right/'+save+'_'+i+'.png',bbox_inches="tight",pad_inches=0)
    plt.clf()

for i in range(2300,3000-35):
    plt.imshow(WRTFT_plot[:,i:i+35])
    plt.axis('off')
    i = str(i)
    plt.savefig('./Result/Prone/'+save+'_'+i+'.png',bbox_inches="tight",pad_inches=0)
    plt.clf()

for i in range(3400,4100-35):
    plt.imshow(WRTFT_plot[:,i:i+35])
    plt.axis('off')
    i = str(i)
    plt.savefig('./Result/Left/'+save+'_'+i+'.png',bbox_inches="tight",pad_inches=0)
    plt.clf()

for i in range(4400,5100-35):
    plt.imshow(WRTFT_plot[:,i:i+35])
    plt.axis('off')
    i = str(i)
    plt.savefig('./Result/Supine/'+save+'_'+i+'.png',bbox_inches="tight",pad_inches=0)
    plt.clf()

for i in range(5300,6000-35):
    plt.imshow(WRTFT_plot[:,i:i+35])
    plt.axis('off')
    i = str(i)
    plt.savefig('./Result/Left/'+save+'_'+i+'.png',bbox_inches="tight",pad_inches=0)
    plt.clf()

for i in range(6300, 7000 - 35):
    plt.imshow(WRTFT_plot[:, i:i + 35])
    plt.axis('off')
    i = str(i)
    plt.savefig('./Result/Prone/' + save + '_' + i + '.png', bbox_inches="tight", pad_inches=0)
    plt.clf()

for i in range(7200, 7900 - 35):
    plt.imshow(WRTFT_plot[:, i:i + 35])
    plt.axis('off')
    i = str(i)
    plt.savefig('./Result/Right/' + save + '_' + i + '.png', bbox_inches="tight", pad_inches=0)
    plt.clf()
#
for i in range(8100,8800-35):
    plt.imshow(WRTFT_plot[:,i:i+35])
    plt.axis('off')
    i = str(i)
    plt.savefig('./Result/Prone/'+save+'_'+i+'.png',bbox_inches="tight",pad_inches=0)
    plt.clf()

for i in range(9100,9800-35):
    plt.imshow(WRTFT_plot[:,i:i+35])
    plt.axis('off')
    i = str(i)
    plt.savefig('./Result/Supine/'+save+'_'+i+'.png',bbox_inches="tight",pad_inches=0)
    plt.clf()
#
for i in range(10000,10700-35):
    plt.imshow(WRTFT_plot[:,i:i+35])
    plt.axis('off')
    i = str(i)
    plt.savefig('./Result/Right/'+save+'_'+i+'.png',bbox_inches="tight",pad_inches=0)
    plt.clf()

for i in range(11000,11700-35):
    plt.imshow(WRTFT_plot[:,i:i+35])
    plt.axis('off')
    i = str(i)
    plt.savefig('./Result/Left/'+save+'_'+i+'.png',bbox_inches="tight",pad_inches=0)
    plt.clf()

#











