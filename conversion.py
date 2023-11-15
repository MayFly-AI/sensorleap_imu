import numpy as np
from scipy.constants import g

def accelerometer_to_attitude(ax, ay, az):
    # range: +-pi/2 (+-90 degrees)
    #roll = np.arcsin(np.clip(-ax / g,-0.9999999,0.9999999))
    #pitch = np.arcsin(np.clip(ay / (g * np.cos(roll)),-0.9999999,0.9999999))

    # range +-pi (+-180 degrees)
    # needs 1g (redo calibration code to account for this)
    roll = np.arctan2(-ax, np.sqrt(az**2 + ay**2))
    pitch = np.arctan2(ay, np.sqrt(ax**2 + az**2))
 
    yaw = 0

    return roll, pitch, yaw


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to Quaternion"""

    q_1 = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    q_2 = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    q_3 = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    q_4 = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)

    return q_1, q_2, q_3, q_4

# Body 3-2-1 sequence (yaw -> pitch -> roll)
def quaternion_to_euler_angles(q_1, q_2, q_3, q_4):
    """Convert Quaternion to Euler angles"""

    #phi = np.degrees(np.arctan2(2 * (q_1 * q_2 + q_3 * q_4), 1 - 2 * (q_2 ** 2 + q_3 ** 2)))
    #theta = np.degrees(np.arcsin(2 * (q_1 * q_3 - q_4 * q_2)))
    #omega = np.degrees(np.arctan2(2 * (q_1 * q_4 + q_2 * q_3), 1 - 2 * (q_3 ** 2 + q_4 ** 2)))

    phi = np.arctan2(2 * (q_1 * q_2 + q_3 * q_4), 1 - 2 * (q_2 ** 2 + q_3 ** 2))
    theta = np.arcsin(2 * (q_1 * q_3 - q_4 * q_2))
    omega = np.arctan2(2 * (q_1 * q_4 + q_2 * q_3), 1 - 2 * (q_3 ** 2 + q_4 ** 2))

    return phi, theta, omega

def quaternion_to_rotation_matrix(q0, q1, q2, q3):
    r00 = 1. - 2.*(q2**2 + q3**2)
    r01 = 2.*(q1 * q2 - q0 * q3)
    r02 = 2.*(q1 * q3 + q0 * q2)

    r10 = 2.*(q1 * q2 + q0 * q3)
    r11 = 1. - 2.*(q1**2 + q3**2)
    r12 = 2.*(q2 * q3 - q0 * q1)

    r20 = 2.*(q1 * q3 - q0 * q2)
    r21 = 2.*(q2 * q3 + q0 * q1)
    r22 = 1. - 2.*(q1**2 + q2**2)

    rot_matrix = np.array([[r00, r01, r02, 0.],
                           [r10, r11, r12, 0.],
                           [r20, r21, r22, 0.],
                           [0.,  0.,  0.,  1.]])
    return rot_matrix

def gyro_transition_matrix(gyro_phi, gyro_theta, gyro_omega, delta_t):
    """
    Calculate state transition matrix from gyro readings. (Estimate for Kalman).
    Quaternion Integration for Attitude Estimation.
    """

    A = np.array(
        np.identity(4)
        + (delta_t / 2)
        * np.array(
            [
                [0, -gyro_phi, -gyro_theta, -gyro_omega],
                [gyro_phi, 0, gyro_omega, -gyro_theta],
                [gyro_theta, -gyro_omega, 0, gyro_phi],
                [gyro_omega, gyro_theta, -gyro_phi, 0],
            ]
        )
    )
    return A

def normalize_quaternion(q_1, q_2, q_3, q_4):
    """Normalize a quaternion to get a unit length (important for rotations)"""

    norm = np.sqrt(q_1 ** 2 + q_2 ** 2 + q_3 ** 2 +  q_4 ** 2)

    return np.array([q/norm for q in [q_1, q_2, q_3, q_4]])
