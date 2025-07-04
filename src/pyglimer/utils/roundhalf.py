#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Rounds to next 0.5

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Saturday, 02nd May 2020 01:48:03 pm
Last Modified: Friday, 20th January 2023 03:50:29 pm
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
