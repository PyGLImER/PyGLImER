#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Rounds to next 0.5

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Saturday, 02nd May 2020 01:48:03 pm
Last Modified: Thursday, 25th March 2021 04:00:02 pm
'''


def roundhalf(number) -> float:
    """
    Rounds to next half of integer

    Parameters
    ----------
    number : float/int
        number to be rounded.

    Returns
    -------
    float
        Closest half of integer.

    """
    return round(number*2) / 2
