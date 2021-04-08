'''
This module will handle file IO. The problematic functions here are the
incoming `.mat` files that are produced by the GLImER database.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 22nd July 2020 10:51:22 am
Last Modified: Thursday, 25th March 2021 03:36:30 pm
'''
import os


def finddir():
    """
    Returns the absolute data directory, so stuff can be loaded
    """
    d = os.path.dirname(os.path.realpath(__file__))
    return d
