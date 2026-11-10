'''
Shared infrastructure for the hdf5 waveform databases (raw waveforms and
receiver functions). Both :mod:`~pyglimer.database.raw` and
:mod:`~pyglimer.database.rfh5` build on the handler and header-conversion
logic defined here so that a bugfix has to be made only once.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
    Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 10th November 2026
'''
import fnmatch
import logging
import os
import re
import warnings

import numpy as np
import h5py
from obspy.core import Stats
from obspy.core.utcdatetime import UTCDateTime


# Filename layout of the per-station hdf5 database files (shared by the
# raw-waveform and receiver-function databases). The dataset hierarchy
# inside the files differs between the two databases and is defined in
# their respective modules.
h5_FMTSTR = os.path.join("{dir}", "{network}.{station}.h5")


class DBHandlerBase(h5py.File):
    """
    Base file handler for the hdf5 waveform databases. Parses the
    compression options; the format-specific database handlers inherit from
    this class.

    .. warning::

        **Should not be accessed directly.** Access the databases via their
        context-manager wrappers (e.g.,
        :class:`~pyglimer.database.rfh5.RFDataBase`) instead.
    """
    def __init__(self, path, mode, compression):
        super(DBHandlerBase, self).__init__(path, mode=mode)
        if isinstance(compression, str):
            self.compression = re.findall(r'(\w+?)(\d+)', compression)[0][0]
            if self.compression != 'gzip':
                raise ValueError(
                    'Compression of type %s is not supported.'
                    % self.compression)
            self.compression_opts = int(
                re.findall(r'(\w+?)(\d+)', compression)[0][1])
            if self.compression_opts not in np.arange(1, 10, 1, dtype=int):
                ii = np.argmin(abs(
                    np.arange(1, 10, 1, dtype=int) - self.compression_opts))
                self.compression_opts = np.arange(1, 10, 1, dtype=int)[ii]
                warnings.warn(
                    'Chosen compression level is not available for %s. \
%s Has been chosen instead (closest)' % (
                        self.compression, str(self.compression_opts)))
        else:
            self.compression = None
            self.compression_opts = None

    def _close(self):
        self.close()


def all_traces_recursive(
    group: h5py._hl.group.Group, stream, pattern: str, trace_cls):
    """
    Recursively, appends all traces in a h5py group to the input stream.
    In addition this will check whether the data matches a certain pattern.

    :param group: group to search through
    :type group: class:`h5py._hl.group.Group`
    :param stream: Stream to append the traces to
    :type stream: Stream
    :param pattern: pattern for the path in the hdf5 file, see fnmatch for
        details.
    :type pattern: str
    :param trace_cls: The Trace subclass to create (e.g., Trace or RFTrace)
    :return: Stream with appended traces
    :rtype: Stream
    """
    for v in group.values():
        if isinstance(v, h5py._hl.group.Group):
            all_traces_recursive(v, stream, pattern, trace_cls)
        elif not fnmatch.fnmatch(v.name, pattern) and v.name not in pattern:
            continue
        else:
            stream.append(
                trace_cls(np.array(v), header=read_hdf5_header(v)))
    return stream


def convert_header_to_hdf5(dataset: h5py.Dataset, header: Stats):
    """
    Convert an :class:`~obspy.core.Stats` object and adds it to the provided
    hdf5 dataset.

    :param dataset: the dataset that the header should be added to
    :type dataset: h5py.Dataset
    :param header: The trace's header
    :type header: Stats
    """
    header = dict(header)
    for key in header:
        try:
            if isinstance(header[key], UTCDateTime):
                # convert time to string
                header[key] = header[key].format_fissures()
            dataset.attrs[key] = header[key]
        except TypeError:
            logging.debug(
                f'The header contains an item of type {type(header[key])}.'
                + 'Information of this type cannot be written to hdf5.')
            continue


def read_hdf5_header(dataset: h5py.Dataset) -> Stats:
    """
    Takes an hdf5 dataset as input and returns the header of the Trace.

    :param dataset: The dataset to be read from
    :type dataset: h5py.Dataset
    :return: The trace's header
    :rtype: Stats
    """
    attrs = dataset.attrs
    time_keys = ['starttime', 'endtime', 'onset', 'event_time']
    header = {}
    for key in attrs:
        if key in time_keys:
            try:
                header[key] = UTCDateTime(attrs[key])
            except ValueError as e:
                # temporary fix of obspy's UTCDateTime issue. SHould be removed
                # as soon as they release version 1.23
                if attrs[key][4:8] == '360T':
                    new = list(attrs[key])
                    new[6] = '1'
                    header[key] = UTCDateTime(''.join(new)) - 86400
                else:
                    raise e
        elif key == 'processing':
            header[key] = list(attrs[key])
        else:
            header[key] = attrs[key]
    return Stats(header)
