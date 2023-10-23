import numpy as np
import time

from mayfly.sensorcapture import SensorCapture

class DataSensorleap:
    def __init__(self):
        self.cap = SensorCapture(list(range(64)),'')

    def get_data(self, ts, acc_x, acc_y, acc_z, rads_x, rads_y, rads_z):
        capture = self.cap.read()
        if capture['type'] != 'imu':
            return 0
        tstamps = capture['values']['t']
        acc = capture['values']['acc']
        rads = capture['values']['rads']
        N = len(tstamps)
        for j in range(N):
            ts.append(tstamps[j])
            acc_x.append(acc[j*3+0])
            acc_y.append(acc[j*3+1])
            acc_z.append(acc[j*3+2])
            rads_x.append(rads[j*3+0])
            rads_y.append(rads[j*3+1])
            rads_z.append(rads[j*3+2])
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
        time.sleep(1./30.)
        return 1
