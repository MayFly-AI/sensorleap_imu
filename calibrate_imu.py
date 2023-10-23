import numpy as np

from get_data import DataSensorleap

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

    data = DataSensorleap()

    N = 2000
    for i in range(N):
        if i%100==1:
            print(i, ' out of ', N)
        _ = data.get_data(ts, acc_x, acc_y, acc_z, rads_x, rads_y, rads_z)
       
        if i < 100: # probably not necessary
            continue

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

if __name__ == '__main__':
    calibrate()

