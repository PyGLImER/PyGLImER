import numpy as np


def Ra2b(a, b):
    """Gets rotation matrix for R3 vectors that rotates a -> b.
    This is the linear algebra version of rodriquez formula.
    Theory:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Parameters
    ----------
    a : np.ndarray
        vector to rotate towards
    b : np.ndarray
        vector to rotate from

    Returns
    -------
    np.ndarray
        3x3 rotation matrix

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.04.17 00.30

    """

    # compute normalized vectors
    an = a / np.linalg.norm(a)
    bn = b / np.linalg.norm(b)

    # Compute cross and dot product
    v = np.cross(an, bn)
    c = np.dot(an, bn)

    # Compute skew-symmetric cross-product matrix of n
    def S(n):
        Sn = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        return Sn
    Sv = S(v)

    # Compute the rotation matrix
    R = np.eye(3) + Sv + np.dot(Sv, Sv) * 1/(1+c)

    return R
