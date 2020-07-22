'''
Author: Peter Makus (peter.makus@student.uib.no)

Created: Wednesday, 22nd July 2020 10:51:22 am
Last Modified: Wednesday, 22nd July 2020 10:52:36 am
'''
import os

def finddir():
    """
    Returns the absolute data directory, so stuff can be loaded
    """
    d = os.path.dirname(os.path.realpath(__file__))
    return d