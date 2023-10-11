# sensorleap_imu

Setup sensorleap sensor with IMU enabled. Guide (TODO)

Run calibration:

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
python show_graph.py imu_means.txt
```


