# sensorleap_imu

This repository contains a demo to stream live IMU data from a sensorleap sensor and to use this data to
estimate roll, pitch and yaw (assuming constant position). The demo uses a kalman filter to fuse data from the gyroscope
with the data from the accelerometer.

Setup sensorleap sensor with IMU enabled. Guide (TODO)

Let sensor or robot with sensor mounted on still on a level surface, then run calibration: 

```bash
python calibrate_imu.py
```

This will dump a file (imu_means.txt) with mean values for the 6 DoF for the BMI088 IMU.

To stream and visualize the IMU data, run
```bash
python show_graph.py
```

Notice that the calibration is off. Accelerometer is not at (0,-1g,0) and gyroscope is drifting a lot.
To take the IMU calibration into account when visualizing the streamed data, run:

```bash
python show_graph.py --means imu_means.txt
```

To record IMU data to text file, run:
```bash
python show_graph.py --record
```
This will create a file: record_imu.txt.

To use the recorded data instead of live streamed data, do:
```bash
python show_graph.py --means imu_means.txt --recording record_imu.txt
```
To see the IMU data visualized with OpenGL, run:
```bash
python gl_imu.py --means imu_means.txt --recording record_imu.txt
```
and with recorded data:

```bash
python gl_imu.py --means imu_means.txt --recording record_imu.txt
```








