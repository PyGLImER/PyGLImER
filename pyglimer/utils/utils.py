"""

Module containing mini-utilities/convenience functions.

Author:
    Lucas Sawade (lsawade@princeton.edu)


Last Update: November 2019

"""


def dt_string(dt):
    """Returns Time elapsed string depending on how much time has passed.
    After a certain amount of seconds it returns minutes, and after a certain
    amount of minutes it returns the elapsed time in hours."""

    if dt > 500:
        dt = dt / 60

        if dt > 120:
            dt = dt / 60
            tstring = "   Time elapsed: %3.1f h" % dt
        else:
            tstring = "   Time elapsed: %3.1f min" % dt
    else:
        tstring = "   Time elapsed: %3.1f s" % dt

    return tstring


def chunks(lst, n):
    """Yield successive n-sized chunks from lst. Useful for multi-threading"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
