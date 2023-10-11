import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.constants import g

import torch
from mayfly.videocapture import VideoCapture

cap = VideoCapture(list(range(64)),'')

x_len = 200

fig = plt.figure('IMU data',figsize=(24,15))
axes = []
N_figs = 10
for i in range(1,N_figs):
    axes.append(fig.add_subplot(N_figs-1, 1, i))
xs = []
ys = []
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

def animate(i, xs, ys, means):
    global acc_x
    global acc_y
    global acc_z
    global rads_x
    global rads_y
    global rads_z
    global a_roll
    global a_pitch
    global g_roll
    global g_pitch
    global g_yaw

    frames = cap.read()
    if type(frames) is tuple:
        return
    for j in range(len(frames['imu']['t'])):
        xs.append(frames['imu']['t'][j])
        acc_x.append(frames['imu']['acc'][j*3+0])
        acc_y.append(frames['imu']['acc'][j*3+1])
        acc_z.append(frames['imu']['acc'][j*3+2])
        rads_x.append(frames['imu']['rads'][j*3+0])
        rads_y.append(frames['imu']['rads'][j*3+1])
        rads_z.append(frames['imu']['rads'][j*3+2])

        ax = acc_x[-1]-means[0]
        ay = acc_y[-1]-means[1] # y is down
        az = acc_z[-1]-means[2] 
        gx = rads_x[-1]-means[3]
        gy = rads_y[-1]-means[4]
        gz = rads_z[-1]-means[5]

        a_roll.append(np.arcsin(ax / g))
        a_pitch.append(-np.arcsin(az / (g * np.cos(a_roll[-1]))))
        if len(xs) > 1:
            dt = (xs[-2]-xs[-1])/1000000. # microseconds to seconds
            g_roll.append(g_roll[-1] + gz*dt) # pitch and roll 
            g_pitch.append(g_pitch[-1] + gx*dt)
            g_yaw.append(g_yaw[-1] + gy*dt)

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

    for ax in axes:
        ax.clear()
    axes[0].plot(xs, acc_x-means[0],color='b')
    axes[1].plot(xs, acc_y-means[1],color='b')
    axes[2].plot(xs, acc_z-means[2],color='b')
    axes[3].plot(xs, rads_x-means[3],color='orange')
    axes[4].plot(xs, rads_y-means[4],color='orange')
    axes[5].plot(xs, rads_z-means[5],color='orange')
    axes[6].plot(xs, a_roll,color='blue')
    axes[6].plot(xs, g_roll,color='orange')
    axes[6].legend(['Accelerometer','Gyroscope'],loc='upper left')
    axes[7].plot(xs, a_pitch,color='blue')
    axes[7].plot(xs, g_pitch,color='orange')
    axes[7].legend(['Accelerometer','Gyroscope'],loc='upper left')
    axes[8].plot(xs, g_yaw,color='orange')

    axes[0].set_ylabel('Acc x') 
    axes[1].set_ylabel('Acc y') 
    axes[2].set_ylabel('Acc z') 
    axes[3].set_ylabel('Gyro x') 
    axes[4].set_ylabel('Gyro y') 
    axes[5].set_ylabel('Gyro z') 
    axes[6].set_ylabel('Roll') 
    axes[7].set_ylabel('Pitch') 
    axes[8].set_ylabel('Yaw') 
    axes[8].set_xlabel('Microseconds since epoch')

if __name__ == '__main__':
    means = np.zeros([6])
    if len(sys.argv) > 1:
        means = np.loadtxt(sys.argv[1])
        print('Loaded means',means)

    ani = animation.FuncAnimation(fig, animate, fargs=(xs,ys,means), interval=1)
    plt.show()


