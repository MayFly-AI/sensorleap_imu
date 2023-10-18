import numpy as np

from mayfly.videocapture import VideoCapture

class DataSensorleap:
    def __init__(self):
        self.cap = VideoCapture(list(range(64)),'')

    def get_data(self, ts, acc_x, acc_y, acc_z, rads_x, rads_y, rads_z):
        frames = self.cap.read()
        if type(frames) is tuple:
            return 0
        N = len(frames['imu']['t'])
        for j in range(N):
            ts.append(frames['imu']['t'][j])
            acc_x.append(frames['imu']['acc'][j*3+0])
            acc_y.append(frames['imu']['acc'][j*3+1])
            acc_z.append(frames['imu']['acc'][j*3+2])
            rads_x.append(frames['imu']['rads'][j*3+0])
            rads_y.append(frames['imu']['rads'][j*3+1])
            rads_z.append(frames['imu']['rads'][j*3+2])
        return N

class DataRecording:
    def __init__(self, file_path):
        self.recording_file = open(file_path,'r')
        self.lines = self.recording_file.readlines()
        self.lines_idx = 0

    def get_data(self, ts, acc_x, acc_y, acc_z, rads_x, rads_y, rads_z):
        if self.lines_idx == len(self.lines):
            return 0
        line = self.lines[self.lines_idx].split(' ')
        self.lines_idx += 1
        ts.append(int(line[0]))
        acc_x.append(float(line[1]))
        acc_y.append(float(line[2]))
        acc_z.append(float(line[3]))
        rads_x.append(float(line[4]))
        rads_y.append(float(line[5]))
        rads_z.append(float(line[6]))
        #time.sleep(1./50.)
        return 1
