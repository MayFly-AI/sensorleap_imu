import numpy as np
from numpy.linalg import inv as inverse

from conversion import accelerometer_to_attitude, euler_to_quaternion, quaternion_to_euler_angles, gyro_transition_matrix, normalize_quaternion

# Kalman filter implementation from https://github.com/Silverlined/Kalman-Quaternion-Rotation

"""
# x -> state estimate;
# z -> state measurement;
# F -> state-transition model;
# H -> observation model;
# P -> process covariance;
# Q -> covariance of the process noise;
# R -> covariance of the observation noise;
# K -> kalman gain;
"""

class KalmanFilter:
    def __init__(self, x0, F, H, P, Q, R):
        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.P = P
        self.Q = Q
        self.R = R
        self.x = x0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.transpose() + self.Q
        
    def correct(self, z):
        self.K = self.P @ self.H.transpose() @ inverse(self.H @ self.P @ self.H.transpose() + self.R)
        self.x = self.x + self.K @ (z - self.H @ self.x)

        I = np.eye(self.n)
        self.P = (I - self.K @ self.H) @ self.P

        return self.x

    def update_state_transition(self, F):
        self.F = F

    def normalize_x(self, x):
        self.x = x

 
class KalmanWrapper:
    def __init__(self):
        x0 = np.array(euler_to_quaternion(0,0,0))
        F = np.identity(4)
        H = np.identity(4)
        P = np.eye(4)
        # Initialize covariance matrices
        Q = np.array([[10 ** -4, 0, 0, 0],
                    [0, 10 ** -4, 0, 0], 
                    [0, 0, 10 ** -4, 0], 
                    [0, 0, 0, 10 ** -4]])

        R = np.array([[10, 0, 0, 0],
                    [0, 10, 0, 0],
                    [0, 0, 10, 0],
                    [0, 0, 0, 10]])
        self.kalman = KalmanFilter(x0, F, H, P, Q, R)

    def __call__(self, inp, dt):
        self.F = gyro_transition_matrix(inp[0], inp[1], inp[2], dt)
        self.kalman.update_state_transition(self.F)
        self.kalman.predict()

        z = euler_to_quaternion(*accelerometer_to_attitude(inp[3], inp[4], inp[5]))
        x = self.kalman.correct(z)
        x = normalize_quaternion(*x)
        self.kalman.normalize_x(x)

        phi, theta, omega = quaternion_to_euler_angles(*x)

        return phi, theta, omega   
