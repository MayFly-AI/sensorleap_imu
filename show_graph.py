import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.constants import g
import argparse
import time

from get_data import DataSensorleap, DataRecording
from kalman import KalmanWrapper

class App:
    def __init__(self):
        self.fig = plt.figure('IMU data',figsize=(24,15))
        self.axes = []
        N_figs = 10
        for i in range(1,N_figs):
            self.axes.append(self.fig.add_subplot(N_figs-1, 1, i))
        self.xs = []
        self.acc_x = []
        self.acc_y = []
        self.acc_z = []
        self.rads_x = []
        self.rads_y = []
        self.rads_z = []
        self.a_roll = []
        self.a_pitch = []
        self.g_roll = [0.]
        self.g_pitch = [0.]
        self.g_yaw = [0.]
        self.roll = [0.]
        self.pitch = [0.]
        self.yaw = [0.]

        self.record_file = None
        self.kalman_wrapper = KalmanWrapper()

    def record_to_file(j):
        if self.record_file is None:
            self.record_file = open('record_imu.txt','w')
        self.record_file.write(str(self.xs[-j])+' '+str(self.acc_x[-j])+' '+str(self.acc_y[-j])+' '+str(self.acc_z[-j])+' '+
                            str(self.rads_x[-j])+' '+str(self.rads_y[-j])+' '+str(self.rads_z[-j])+'\n')

    def animate(self, i, means, args, data):
        N = data.get_data(self.xs, self.acc_x, self.acc_y, self.acc_z, self.rads_x, self.rads_y, self.rads_z)
        for j in range(N,0,-1):
            if args.record:
                self.record_to_file(j)

            ax = self.acc_x[-j]-means[0]
            ay = self.acc_y[-j]-means[1] # y is down
            az = self.acc_z[-j]-means[2] 
            gx = self.rads_x[-j]-means[3]
            gy = self.rads_y[-j]-means[4]
            gz = self.rads_z[-j]-means[5]

            self.a_roll.append(np.arcsin(ax / g))
            self.a_pitch.append(-np.arcsin(az / (g * np.cos(self.a_roll[-1]))))
            if len(self.xs) > N or j < N:
                dt = (self.xs[-(j+1)]-self.xs[-j])/1000000. # microseconds to seconds
                self.g_roll.append(self.g_roll[-1] + gz*dt) # pitch and roll 
                self.g_pitch.append(self.g_pitch[-1] + gx*dt)
                self.g_yaw.append(self.g_yaw[-1] + gy*dt)

                ret = self.kalman_wrapper([gz,gx,gy,ax,az,ay], dt)
                self.roll.append(ret[0])
                self.pitch.append(ret[1])
                self.yaw.append(ret[2])

        x_len = 200
        self.xs = self.xs[-x_len:]
        self.acc_x = self.acc_x[-x_len:]
        self.acc_y = self.acc_y[-x_len:]
        self.acc_z = self.acc_z[-x_len:]
        self.rads_x = self.rads_x[-x_len:]
        self.rads_y = self.rads_y[-x_len:]
        self.rads_z = self.rads_z[-x_len:]
        self.a_roll = self.a_roll[-x_len:]
        self.a_pitch = self.a_pitch[-x_len:]
        self.g_roll = self.g_roll[-x_len:]
        self.g_pitch = self.g_pitch[-x_len:]
        self.g_yaw = self.g_yaw[-x_len:]
        self.roll = self.roll[-x_len:]
        self.pitch = self.pitch[-x_len:]
        self.yaw = self.yaw[-x_len:]

        for ax in self.axes:
            ax.clear()
        self.axes[0].plot(self.xs, self.acc_x-means[0],color='b') # why can i subtract scalar from list?
        self.axes[1].plot(self.xs, self.acc_y-means[1],color='b')
        self.axes[2].plot(self.xs, self.acc_z-means[2],color='b')
        self.axes[3].plot(self.xs, self.rads_x-means[3],color='orange')
        self.axes[4].plot(self.xs, self.rads_y-means[4],color='orange')
        self.axes[5].plot(self.xs, self.rads_z-means[5],color='orange')
        self.axes[6].plot(self.xs, self.a_roll,color='blue')
        self.axes[6].plot(self.xs, self.g_roll,color='orange')
        self.axes[6].plot(self.xs, self.roll, color='green')
        self.axes[6].legend(['Accelerometer','Gyroscope','Kalman'],loc='upper left')
        self.axes[7].plot(self.xs, self.a_pitch,color='blue')
        self.axes[7].plot(self.xs, self.g_pitch,color='orange')
        self.axes[7].plot(self.xs, self.pitch, color='green')
        self.axes[7].legend(['Accelerometer','Gyroscope','Kalman'],loc='upper left')
        self.axes[8].plot(self.xs, self.g_yaw,color='orange')
        self.axes[8].plot(self.xs, self.yaw,color='green')
        self.axes[8].legend(['Gyroscope','Kalman'],loc='upper left')

        self.axes[0].set_ylabel('Acc x') 
        self.axes[1].set_ylabel('Acc y') 
        self.axes[2].set_ylabel('Acc z') 
        self.axes[3].set_ylabel('Gyro x') 
        self.axes[4].set_ylabel('Gyro y') 
        self.axes[5].set_ylabel('Gyro z') 
        self.axes[6].set_ylabel('Roll') 
        self.axes[7].set_ylabel('Pitch') 
        self.axes[8].set_ylabel('Yaw') 
        self.axes[8].set_xlabel('Microseconds since epoch')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--means', type=str, required=False)
    parser.add_argument('--record', required=False, action='store_true')
    parser.add_argument('--recording', type=str, required=False)
    args = parser.parse_args()
    means = np.zeros([6])
    if args.means is not None:
        means = np.loadtxt(args.means)
        print('Loaded means',means)

    if args.recording is None:
        data = DataSensorleap()
    else:
        data = DataRecording(args.recording)

    app = App()

    ani = animation.FuncAnimation(app.fig, app.animate, fargs=(means,args,data), interval=1)
    plt.show()


