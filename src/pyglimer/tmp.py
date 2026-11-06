'''
Contains variables that are constantly changing and need to be accessed by
several modules.

Author: Peter Makus (peter.makus@student.uib.no)

Created: Tuesday, 16th June 2020 10:39:21 am
Last Modified: Wednesday, 7th September 2022 11:46:47 am
'''

# If download and preprocessing are conducted simultaneously this will
# define where the download is happening
folder = 'undefined'

re_client = []


def set_clients(clients):
    """
    Registers the FDSN client names that the download uses. The error
    handlers in :mod:`pyglimer.waveform.errorhandler` use this list to
    retry failed downloads with other data centres.

    :param clients: FDSN client name(s) (e.g. ``'IRIS'``), a list thereof,
        or ``None`` (leaves the current list untouched).
    :type clients: str, list, or None
    """
    global re_client
    if clients is None:
        return
    if isinstance(clients, str):
        clients = [clients]
    re_client = list(clients)
