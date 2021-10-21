'''
Toolbox for statistical tools to quantify errors in receiver functions.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 27th September 2021 03:14:17 pm
Last Modified: Thursday, 21st October 2021 10:23:01 am
'''


from typing import List
import numpy as np


def stack_case_resampling(data: np.ndarray, b: int = 1000) -> List[np.ndarray]:
    """
    Monte Carlo Case resampling algorithm that can be used with single station
    RF stacks (aka bootstrap).

    :param data: Moveout corrected receiver function data that is used for the
        bootstrap. Given in form of a 2D matrix -> 1 line = 1 RF.
    :type data: np.ndarray
    :param b: Number of iterations that the bootstrap is supposed to run.
        Defaults to 1000
    :type b: int, optional
    :return: A list with each of the boot strapped stacks / averages.
    :rtype: List[np.ndarray]
    """
    data = np.array(data)
    rstack = []
    for _ in range(b):
        resampled = np.empty_like(data)
        for ni, _ in enumerate(resampled):
            n = np.random.randint(0, data.shape[0])
            resampled[ni, :] = data[n, :]
        rstack.append(np.average(resampled, axis=0))
    return rstack
