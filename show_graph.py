import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter
from scipy.constants import g
import argparse
import time
from multiprocessing import Process, Queue
import glm

from get_data import DataSensorleap, DataRecording
from kalman import KalmanWrapper
from conversion import accelerometer_to_attitude

save_gif = False

def capture_and_compute(q_plot, calib, args):
    if args.record:
        record_file = open('record_imu.txt','w')
    if args.recording is None:
        data = DataSensorleap()
    else:
        data = DataRecording(args.recording)
    xs = []
    acc_x = []
    acc_y = []
    acc_z = []
    rads_x = []
    rads_y = []
    rads_z = []
    a_roll = []
    a_pitch = []
    g_roll = [0.]
    g_pitch = [0.]
    g_yaw = [0.]
    roll = [0.]
    pitch = [0.]
    yaw = [0.]

    record_file = None
    kalman_wrapper = KalmanWrapper()
    while True:
        N = data.get_data(xs, acc_x, acc_y, acc_z, rads_x, rads_y, rads_z)
        for j in range(N,0,-1):
            if args.record:
                record_file.write(str(self.xs[-j])+' '+str(self.acc_x[-j])+' '+str(self.acc_y[-j])+' '+
                        str(self.acc_z[-j])+' '+str(self.rads_x[-j])+' '+str(self.rads_y[-j])+
                        ' '+str(self.rads_z[-j])+'\n')

            ax = acc_x[-j]-calib[0]
            ay = acc_y[-j]-calib[1] # y is down
            az = acc_z[-j]-calib[2] 
            gx = rads_x[-j]-calib[3]
            gy = rads_y[-j]-calib[4]
            gz = rads_z[-j]-calib[5]

            a_roll_j, a_pitch_j, _ = accelerometer_to_attitude(ax,az,ay)
            a_roll.append(a_roll_j)
            a_pitch.append(a_pitch_j)
            if len(xs) > N or j < N:
                dt = (xs[-j]-xs[-(j+1)])/1000000. # microseconds to seconds
                g_roll.append(g_roll[-1] + gz*dt) 
                g_pitch.append(g_pitch[-1] + gx*dt)
                g_yaw.append(g_yaw[-1] + gy*dt)

                Q = kalman_wrapper([gx,gy,gz,ax,az,ay], dt)
                Q = glm.quat(*Q) # convert to GLM quat
                Q = glm.normalize(Q)
                euler = glm.eulerAngles(Q) # pitch, yaw, roll
                roll.append(euler.z)
                pitch.append(euler.x)
                yaw.append(euler.y)

        x_len = 200
        xs = xs[-x_len:]
        acc_x = acc_x[-x_len:]
        acc_y = acc_y[-x_len:]
        acc_z = acc_z[-x_len:]
        rads_x = rads_x[-x_len:]
        rads_y = rads_y[-x_len:]
        rads_z = rads_z[-x_len:]
        a_roll = a_roll[-x_len:]
        a_pitch = a_pitch[-x_len:]
        g_roll = g_roll[-x_len:]
        g_pitch = g_pitch[-x_len:]
        g_yaw = g_yaw[-x_len:]
        roll = roll[-x_len:]
        pitch = pitch[-x_len:]
        yaw = yaw[-x_len:]
        
        try:
            q_plot.put([np.array(xs,copy=True),
                        np.array(acc_x,copy=True), np.array(acc_y,copy=True), np.array(acc_z,copy=True),
                        np.array(rads_x,copy=True), np.array(rads_y,copy=True), np.array(rads_z,copy=True),
                        np.array(a_roll,copy=True), np.array(a_pitch,copy=True),
                        np.array(g_roll,copy=True), np.array(g_pitch,copy=True), np.array(g_yaw,copy=True),
                        np.array(roll,copy=True), np.array(pitch,copy=True), np.array(yaw,copy=True)], block=False)
        except:
            pass

class AnimatedPlot:
    def __init__(self):
        self.fig = plt.figure('IMU data',figsize=(15,15))
        self.axes = []
        N_figs = 10
        for i in range(1,N_figs):
            self.axes.append(self.fig.add_subplot(N_figs-1, 1, i))

    def animate(self, i, q_plot, calib, args):
        print('i',i)
        xs, acc_x, acc_y, acc_z, rads_x, rads_y, rads_z,\
                a_roll, a_pitch, g_roll, g_pitch, g_yaw, roll, pitch, yaw = q_plot.get()

        fs = 12
        for ax in self.axes:
            ax.clear()
        self.axes[0].plot(xs, acc_x-calib[0],color='b')
        self.axes[1].plot(xs, acc_y-calib[1],color='b')
        self.axes[2].plot(xs, acc_z-calib[2],color='b')
        self.axes[3].plot(xs, rads_x-calib[3],color='orange')
        self.axes[4].plot(xs, rads_y-calib[4],color='orange')
        self.axes[5].plot(xs, rads_z-calib[5],color='orange')
        self.axes[6].plot(xs, a_roll,color='blue')
        self.axes[6].plot(xs, g_roll,color='orange')
        self.axes[6].plot(xs, roll, color='green')
        self.axes[6].legend(['Accelerometer','Gyroscope','Kalman'],loc='upper left', fontsize=fs, prop=dict(weight='bold'))
        self.axes[7].plot(xs, a_pitch,color='blue')
        self.axes[7].plot(xs, g_pitch,color='orange')
        self.axes[7].plot(xs, pitch, color='green')
        self.axes[7].legend(['Accelerometer','Gyroscope','Kalman'],loc='upper left', fontsize=fs, prop=dict(weight='bold'))
        self.axes[8].plot(xs, g_yaw,color='orange')
        self.axes[8].plot(xs, yaw,color='green')
        self.axes[8].legend(['Gyroscope','Kalman'],loc='upper left', fontsize=fs, prop=dict(weight='bold'))
        self.axes[0].set_ylabel('Acc x', fontsize=fs, weight='bold') 
        self.axes[1].set_ylabel('Acc y', fontsize=fs, weight='bold') 
        self.axes[2].set_ylabel('Acc z', fontsize=fs, weight='bold') 
        self.axes[3].set_ylabel('Gyro x', fontsize=fs, weight='bold') 
        self.axes[4].set_ylabel('Gyro y', fontsize=fs, weight='bold') 
        self.axes[5].set_ylabel('Gyro z', fontsize=fs, weight='bold') 
        #for ax in self.axes[0:6]:
        #    ax.set_ylim([-10,10])

        self.axes[6].set_ylabel('Roll', fontsize=fs, weight='bold') 
        self.axes[7].set_ylabel('Pitch', fontsize=fs, weight='bold') 
        self.axes[8].set_ylabel('Yaw', fontsize=fs, weight='bold') 
        self.axes[8].set_xlabel('Microseconds since epoch', fontsize=fs, weight='bold')
        for ax in self.axes[6:]:
            ax.set_ylim([-1,1])
        for ax in self.axes:
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.tight_layout() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib', type=str, required=False)
    parser.add_argument('--record', required=False, action='store_true')
    parser.add_argument('--recording', type=str, required=False)
    args = parser.parse_args()
    calib = np.zeros([6])
    if args.calib is not None:
        calib = np.loadtxt(args.calib)
        print('Loaded calibration values:',calib)

    q_plot = Queue(1)
    p = Process(target=capture_and_compute, args=(q_plot,calib,args,))
    p.start()

    aniplot = AnimatedPlot()

    frames = None
    if save_gif:
        frames = 530
    ani = animation.FuncAnimation(aniplot.fig, aniplot.animate, fargs=(q_plot,calib,args), interval=1, frames=frames)

    if save_gif:
        ani.save(filename = 'show_graph.gif', writer='pillow', fps=5)
    else:
        plt.show()

