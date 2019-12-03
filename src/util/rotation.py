import numpy as np


def euler_to_matrix(euler):
    """
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    :param euler:
    :return:
    """
    r, p, y = euler

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(r), -np.sin(r)],
                    [0, np.sin(r), np.cos(r)]
                    ])

    R_y = np.array([[np.cos(p), 0, np.sin(p)],
                    [0, 1, 0],
                    [-np.sin(p), 0, np.cos(p)]
                    ])

    R_z = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0],
                    [0, 0, 1]
                    ])

    return np.dot(R_z, np.dot(R_y, R_x))


def matrix_to_euler(R):
    """
    From Mitash:
    - Calculates rotation matrix to euler angles
    - The result is the same as MATLAB except the order of the euler angles ( x and z are swapped )
    :param R:
    :return:
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return [x, y, z]


def euler_to_quaternion(euler):
    """
    From Mitash.
    :param euler:
    :return:
    """
    r, p, y = euler

    cr = np.cos(r * 0.5)
    cp = np.cos(p * 0.5)
    cy = np.cos(y * 0.5)
    sr = np.sin(r * 0.5)
    sp = np.sin(p * 0.5)
    sy = np.sin(y * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp

    return [w, x, y, z]


def quaternion_to_euler(q):
    """
    From Mitash.
    :param q:
    :return:
    """
    q = list((np.matrix(q)/np.linalg.norm(q)).flat)  # normalize
    w, x, y, z = q

    sr = 2 * (w * x + y * z)
    cr = 1 - 2 * (x**2 + y**2)
    r = np.arctan2(sr, cr)

    sp = 2 * (w * y - z * x)
    if np.abs(sp) >= 1:
        p = np.pi/2 * np.sign(sp)  # use 90deg if out-of-range
    else:
        p = np.arcsin(sp)

    sy = 2 * (w * z + x * y)
    cy = 1 - 2 * (y**2 + z**2)
    y = np.arctan2(sy, cy)

    return [r, p, y]


def matrix_to_quaternion(R):
    """

    :param R:
    :return:
    """
    return euler_to_quaternion(matrix_to_euler(R))


def quaternion_to_matrix(q):
    """

    :param q:
    :return:
    """
    return euler_to_matrix(quaternion_to_euler(q))
