from sys import argv, stdout, stderr, exit
from datetime import datetime

class Logger:

    def __init__(self):
        pass

    def info(self, *args):
        print(f"INFO      [{datetime.now()}]:", *args, file=stdout)

    def debug(self, *args):
        print(f"DEBUG     [{datetime.now()}]:", *args, file=stdout)

    def warning(self, *args):
        print(f"WARNING   [{datetime.now()}]:", *args, file=stdout)

    def error(self, *args):
        print(f"ERROR     [{datetime.now()}]:", *args, file=stderr)

    def exception(self, *args):
        print(f"EXCEPTION [{datetime.now()}]:", *args, file=stderr)


def main():

    logger = Logger()

    logger.info("Starting preprocessh5.py")

    if len(argv) != 11 + 1:
        raise ValueError("Incorrect number of arguments")

    logger.info("Importing modules")

    from ..waveform.preprocessh5 import _preprocessh5_single
    from .. import constants
    from obspy import read_events
    from obspy.taup import TauPyModel

    logger.info("Defining args")

    # Parse arguments
    phase = argv[1]
    rot = argv[2]
    pol =  argv[3]
    taper_perc = float(argv[4])
    model = TauPyModel('iasp91')
    taper = argv[5]
    ta  = 120.0
    tz = constants.onset[phase.upper()]
    rfloc = argv[6]
    deconmeth = argv[7]
    hc_filt = float(argv[8])
    hdf5_file = argv[9]
    evtcat = read_events(argv[10])
    remove_response = argv[11]

    logger.info("Running")

    # Run
    _preprocessh5_single(
        phase,
        rot,
        pol,
        taper_perc,
        model,
        taper,
        tz,
        ta,
        rfloc,
        deconmeth,
        hc_filt,
        logger,
        logger,
        hdf5_file,
        evtcat,
        remove_response
    )

    logger.info("Finished preprocessh5.py")

    exit(0)



