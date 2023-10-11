import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mayfly.videocapture import VideoCapture

def calibrate():
    print('Keep sensor still on ground. Computing mean values..')
    ts = []
    ys = []
    acc_x = []
    acc_y = []
    acc_z = []
    rads_x = []
    rads_y = []
    rads_z = []

    cap = VideoCapture(list(range(64)),'')
    N = 2000
    for i in range(N):
        if i%100==1:
            print(i, ' out of ', N)
        frames = cap.read()
        if type(frames) is tuple:
            continue
       
        if i < 100: # probably not necessary
            continue

        for j in range(len(frames['imu']['t'])):
            ts.append(frames['imu']['t'][j])
            acc_x.append(frames['imu']['acc'][j*3+0])
            acc_y.append(frames['imu']['acc'][j*3+1])
            acc_z.append(frames['imu']['acc'][j*3+2])
            rads_x.append(frames['imu']['rads'][j*3+0])
            rads_y.append(frames['imu']['rads'][j*3+1])
            rads_z.append(frames['imu']['rads'][j*3+2])

    mean_acc_x = np.mean(acc_x)
    mean_acc_y = np.mean(acc_y)
    mean_acc_z = np.mean(acc_z)
    mean_rads_x = np.mean(rads_x)
    mean_rads_y = np.mean(rads_y)
    mean_rads_z = np.mean(rads_z)
    print('mean acc x',mean_acc_x)
    print('mean acc x',mean_acc_y)
    print('mean acc x',mean_acc_z)
    print('mean gyro x',mean_rads_x)
    print('mean gyro y',mean_rads_y)
    print('mean gyro z',mean_rads_z)
    
    fname = 'imu_means.txt'
    np.savetxt(fname,[mean_acc_x,mean_acc_y,mean_acc_z,mean_rads_x,mean_rads_y,mean_rads_z])
    print('Saved mean values to: '+fname)

    if 0:
        fig = plt.figure()
        axes = []
        for i in range(1,7):
            axes.append(fig.add_subplot(6, 1, i))
        axes[0].plot(ts, acc_x)
        axes[1].plot(ts, acc_y)
        axes[2].plot(ts, acc_z)
        axes[3].plot(ts, rads_x)
        axes[4].plot(ts, rads_y)
        axes[5].plot(ts, rads_z)
        axes[0].set_ylabel('Acc x') 
        axes[1].set_ylabel('Acc y') 
        axes[2].set_ylabel('Acc z') 
        axes[3].set_ylabel('Rads x') 
        axes[4].set_ylabel('Rads y') 
        axes[5].set_ylabel('Rads z') 


if __name__ == '__main__':
    calibrate()

