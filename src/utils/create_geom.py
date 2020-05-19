#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic creation of raysum geom files

Created on Thu May 14 10:23:39 2020

Author:
    Peter Makus (peter.makus@student.uib.no)

Last updated:
"""

import os
import numpy as np


def geom3D():
    N = 21
    bazv = np.arange(0, 360, 20)
    raypv = np.linspace(5e-5, 1.5e-4, 8)#np.arange(.1e-5, .31e-4, .25e-5)
    shift_max = 55660/2  # 55660
    filename = '3D'
    create_geom(N, bazv, raypv, shift_max, filename)


def geom3D_even():
    N = 100
    bazv = np.arange(0, 360, 20)
    # raypv = np.arange(.1e-5, .11e-4, .25e-5)
    raypv = np.arange(.7e-4, 1e-4, 1e-5)
    shift_max = 55660
    filename = '3D'
    create_geom(N, bazv, raypv, shift_max, filename, shape='even')


def create_geom(N, bazv, raypv, shift_max, filename, shape='cross'):
    """
    Creates geometry files for Raysum.

    Parameters
    ----------
    N : int
        Number of stations. Has to be uneven if shape=cross.
        Else has to be N=M**2. Were M is a natural number
    bazv : np.ndarray(1D)
        1D array containing the backzimuths per station (deg).
    raypv : np.ndarray(1D)
        1D array containing the slownesses in s/m per backzimuth.
    shift_max : int
        Maximum horizontal shift in m.
    filename : str
        Name of the output file.
    shape : str
        shape of the array

    Raises
    ------
    ValueError
        For Even Ns.

    Returns
    -------
    None.

    """
    if shape == 'cross':
        if N/2 == round(N/2):
            raise ValueError('Number of station has to be uneven.')
    
        # create shift vectors
        xshift = np.hstack((np.linspace(-shift_max, shift_max, round((N+1)/2)),
                            np.zeros(round((N+1)/2))))
        yshift = np.hstack((np.zeros(round((N+1)/2)),
                           np.linspace(-shift_max, shift_max, round((N+1)/2))))
    
        coords = np.unique(np.column_stack((yshift, xshift)), axis=0)
    else:
        M = np.sqrt(N)  # Number of stations per line
        d = shift_max/M
        xshift, yshift = np.mgrid[
            -shift_max:shift_max:M, -shift_max:shift_max:M]
        coords = np.column_stack((xshift.ravel(), yshift.ravel()))

    lines = []  # list with text

    # header
    lines.append('# Automatically created geometry file.\n')
    lines.append("""# Note that one file cannot contain more than 200 traces (max for raysum).\n""")

    ntr = 0  # Number of traces counter
    fpi = []  # List with indices to split file

    for i in range(N):
        lines.append('# Station '+str(i)+'\n')

        for j, baz in enumerate(bazv):
            for k, rayp in enumerate(raypv):
                if rayp == 0:
                    rayp = '0.'
                else:
                    rayp = str(round(rayp,6))

                line = ' '.join(
                    [str(int(bazv[j]))+'.', rayp,
                     str(int(coords[i, 0]))+'.', str(int(coords[i, 1]))+'.\n'])
                lines.append(line)
                ntr = ntr + 1

                if ntr == 200:
                    fpi.append(lines.index(line))
                    ntr = 0

    fpi.append(lines.index(line))

    # Write text to file
    # open outfile
    of = os.path.join('data', 'raysum_traces', filename+'.geom')
    with open(of, 'w') as text:
        text.writelines(lines)

    # Write splitted files
    for i, j in enumerate(fpi):
        of = os.path.join('data', 'raysum_traces', filename+str(i)+'.geom')
        with open(of, 'w') as text:
            if i:
                text.writelines(lines[fpi[i-1]+1:j+1])
            else:
                text.writelines(lines[:j+1])
