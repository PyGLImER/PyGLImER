'''
Author: Peter Makus (peter.makus@student.uib.no
Created: Mon May 25 2020 10:40:04
Last Modified: Monday, 25th May 2020 11:52:46 am
'''

import os
import subprocess

from obspy import UTCDateTime
from pathlib import Path

import config
from .roundhalf import roundhalf


def shortennames(phase):
    # %%
    # parent folder
    # import config
    # phase = 'P'
    # os.chdir('/home/pm/Documents/Masters/PyGLImER')
    di = os.path.join(config.waveform[:-1], phase)

    # sub directories
    sudi = os.listdir(di)
    
    for d in sudi:
        x = d.split('_')
        
        # %%
    
        # x[0] = ot_fiss, x[1] = evtlat, x[2] = evtlon
        # precision=-1 -> precise on 10 seconds
        try:
            ot = UTCDateTime(x[0], precision=-1).format_fissures()[:-6]
        except ValueError:
            # When there is a weird file/filename
            continue
        
        evtlat = str(roundhalf(float(x[1])))
        evtlon = str(roundhalf(float(x[2])))
    
        di_old = os.path.join(di, d)    
        di_new = os.path.join(di, ot + '_' + evtlat + '_' + evtlon)
        if di_old == di_new:
            continue

        elif Path(di_new).is_dir():
            # move content
            susudi = os.listdir(di_new)
            for didi in susudi:
                subprocess.call(['mv', os.path.join(di_old, didi), di_new])
            subprocess.call(['rmdir', di_old])
        else:
            # move the whole folder
            subprocess.call(['mv', di_old, di_new])