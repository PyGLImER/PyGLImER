import numpy as np

import pyglimer.utils.even2Dpoints as e2D


def test_distance():


    x = np.array([0, 1, 1, 0])
    y = np.array([0, 0, 1, 1])

    solutions =[]
    solutions.append(np.array([0., 1., 1.41421356, 1.]))
    solutions.append(np.array([1., 0., 1., 1.41421356]))
    solutions.append(np.array([1.41421356, 1., 0., 1.]))
    solutions.append(np.array([1., 1.41421356, 1., 0.]))

    for _x, _y, _sol in zip(x, y, solutions):
        
        # Compute distances
        darray = e2D.distance(x, y, _x, _y)

        # Check if correct
        np.testing.assert_array_almost_equal(darray, _sol)


def test_even2D():

    # Create even 2D points with min distance
    mind = 0.1
    x, y = e2D.even2Dpoints(10, 1, 1, mind)

    for _x, _y in zip(x, y):
        
        # Compute distances
        darray = e2D.distance(x, y, _x, _y)

        # Check Array for large distances 
        # Note that one of the points has 0 distance since we compute
        # distance to itself --> compare where darray > 0.0
        np.testing.assert_array_less(0.1, darray[darray>0.0])   


def test_even2D_seed():

    # Create even 2D points with min distance
    a = e2D.even2Dpoints(10, 1, 1, 0.1, seed=1234)
    b = e2D.even2Dpoints(10, 1, 1, 0.1, seed=1234)
    
    # Check whether a and be are the same
    np.testing.assert_array_almost_equal(a, b)   