'''
Author: Peter Makus (peter.makus@student.uib.no)

Created: Monday, 25th May 2020 10:40:03
Last Modified: Tuesday, 16th June 2020 12:01:45 pm
'''


import os
import subprocess

from obspy import UTCDateTime
from pathlib import Path
from joblib import Parallel, delayed, cpu_count

from .roundhalf import roundhalf
from .utils import chunks


def shortennames(phase, rawloc):
    """That was kinda a one-time function to correct a bug.

    Parameters
    ----------
    phase : [type]
        [description]
    """
    # %%
    # parent folder
    di = os.path.join(rawloc, phase)

    # sub directories
    sudi = os.listdir(di)
    
    num_cores = cpu_count()
    
    
    def __multicore__(dd):
        for d in dd:
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
                susudi = os.listdir(di_old)
                for didi in susudi:
                    subprocess.call(['mv', os.path.join(di_old, didi), di_new])
                subprocess.call(['rmdir', di_old])
            else:
                # move the whole folder
                subprocess.call(['mv', di_old, di_new])
    
    Parallel(n_jobs=num_cores)(
        delayed(__multicore__)(d) for d in chunks(sudi, num_cores))