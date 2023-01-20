'''
This module will handle file IO. The problematic functions here are the
incoming `.mat` files that are produced by the GLImER database.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 22nd July 2020 10:51:22 am
Last Modified: Friday, 20th January 2023 03:50:29 pm
'''
import os


def finddir():
    """
    Returns the absolute data directory, so stuff can be loaded
    """
    d = os.path.dirname(os.path.realpath(__file__))
    return d
