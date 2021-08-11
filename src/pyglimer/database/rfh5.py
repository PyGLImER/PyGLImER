'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   `GNU Lesser General Public License, Version 3 <https://www.gnu.org/copyleft/lesser.html>`
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 11th August 2021 03:20:09 pm
Last Modified: Wednesday, 11th August 2021 05:08:16 pm
'''

import ast
import fnmatch
import os
import re
from typing import List
import warnings
from copy import deepcopy

import numpy as np
# from numpy.core.fromnumeric import compress
from obspy.core.utcdatetime import UTCDateTime
from obspy.core import Stats
import h5py

from pyglimer.rf.create import RFStream, RFTrace


hierarchy = "/{tag}/{network}/{station}/{phase}/{pol}/{evt_time}"
h5_FMTSTR = os.path.join("{dir}", "{network}.{station}.h5")


class DBHandler(h5py.File):
    """
    The actual file handler of the hdf5 correlation files.

    .. warning::

        **Should not be accessed directly. Access
        :class:`~seismic.db.corr_hdf5.CorrelationDataBase` instead.**

    Child object of :class:`h5py.File` and inherets all its attributes and
    functions in addition to functions that are particularly useful for noise
    correlations.
    """
    def __init__(self, path, mode, compression):
        super(DBHandler, self).__init__(path, mode=mode)
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

    def add_rf(
            self, data: RFTrace or RFStream, tag: str = 'rf'):
        """
        Add correlation data to the hdf5 file. Can be accessed using the
        :func:`~seismic.db.corr_hdf5.DBHandler.get_data()` method.

        :param data: Data to save. Either a
            :class:`~seismic.correlate.correlate.CorrTrace` object or a
            :class:`~seismic.correlate.correlate.CorrStream` holding one or
            several traces.
        :type data: CorrTrace or CorrStream
        :param tag: The tag that the data should be saved under. By convention,
            unstacked correlations are saved with the tag `'subdivision'`,
            whereas stacks are saved with the tag `stack_$stacklen$`, where
            $stacklen$ is to be replaced by the length of the stack in seconds.
        :raises TypeError: for wrong data type.
        """
        if not isinstance(data, RFTrace) and\
                not isinstance(data, RFStream):
            raise TypeError('Data has to be either a \
:class:`~pyglimer.rf.create.RFTrace` object or a \
:class:`~~pyglimer.rf.create.RFStream` object')

        if isinstance(data, RFTrace):
            data = [data]

        for tr in data:
            st = tr.stats
            path = hierarchy.format(
                tag=tag,
                network=st.network, station=st.station, phase=st.phase,
                pol=st.pol, evt_time=st.event_time.format_fissures())
            try:
                ds = self.create_dataset(
                    path, data=tr.data, compression=self.compression,
                    compression_opts=self.compression_opts)
                convert_header_to_hdf5(ds, st)
            except ValueError as e:
                print(e)
                warnings.warn("The dataset %s is already in file and will be \
omitted." % path, category=UserWarning)

    def get_corr_options(self) -> dict:
        try:
            sco = str(self['co'].attrs['co'])
            co = ast.literal_eval(sco)
        except KeyError:
            raise KeyError('No correlation options in file')
        return co

    def get_data(
        self, network: str, station: str, phase: str, evt_time: UTCDateTime,
            tag: str = 'rf', pol: str = 'v') -> RFStream:
        """
        Returns a :class:`~seismic.correlate.correlate.CorrStream` holding
        all the requested data.

        .. note::

            Wildcards are allowed for all parameters.

        :param network: network (combination), e.g., IU-YP
        :type network: str
        :param station: station (combination), e.g., HRV-BRK
        :type station: str
        :param channel: channel (combination), e.g., BZ-BR
        :type channel: str
        :param corr_start: starttime of the time windows used to computed this
            correlation, defaults to None
        :type corr_start: UTCDateTime, optional
        :param corr_end: endtime of the time windows used to computed this
            correlation, defaults to None
        :type corr_end: UTCDateTime, optional
        :return: a :class:`~seismic.correlate.correlate.CorrStream` holding
            all the requested data.
        :rtype: CorrStream
        """
        if isinstance(evt_time, UTCDateTime):
            evt_time = evt_time.format_fissures()
        else:
            evt_time = '*'

        path = hierarchy.format(
            tag=tag, network=network, station=station, phase=phase,
            pol=pol, evt_time=evt_time)
        # Extremely ugly way of changing the path
        if '*' not in path:
            data = np.array(self[path])
            header = read_hdf5_header(self[path])
            return RFStream(RFTrace(data, header=header))
        # Now, we need to differ between the fnmatch pattern and the actually
        # acessed path
        pattern = path.replace('/*', '*')
        if evt_time == '*':
            if pol == '*':
                if phase == '*':
                    if station == '*':
                        if network == '*':
                            path = tag
                        else:
                            path = '/'.join(path.split('/')[:-4])
                    else:
                        path = '/'.join(path.split('/')[:-3])
                else:
                    path = '/'.join(path.split('/')[:-2])
            else:
                path = '/'.join(path.split('/')[:-1])
        return all_traces_recursive(self[path], RFStream(), pattern)


class RFDataBase(object):
    """
    Base class to handle the hdf5 files that contain noise correlations.
    """
    def __init__(
            self, path: str, mode: str = 'a', compression: str = 'gzip3'):
        """
        Access an hdf5 file holding receiver functions. The resulting file can
        be accessed using all functionalities of
        `h5py <https://www.h5py.org/>`_ (for example as a dict).

        :param path: Full path to the file
        :type path: str
        :param mode: Mode to access the file. Options are: 'a' for all, 'w' for
            write, 'r+' for writing in an already existing file, or 'r' for
            read-only , defaults to 'a'.
        :type mode: str, optional
        :param compression: The compression algorithm and compression level
            that the arrays should be saved with. 'gzip3' tends to perform
            well, else you could choose 'gzipx' where x is a digit between
            1 and 9 (i.e., 9 is the highest compression) or None for fastest
            perfomance, defaults to 'gzip3'.
        :type compression: str, optional

        .. warning::

            **Access only through a context manager (see below):**

            >>> with CorrelationDataBase(myfile.h5) as cdb:
            >>>     type(cdb)  # This is a DBHandler
            <class 'seismic.db.corr_hdf5.DBHandler'>

        Example::

            >>> with CorrelationDataBase(
                        '/path/to/db/XN-XN.NEP06-NEP06.h5') as cdb:
            >>>     # find the available tags for existing db
            >>>     print(list(cdb.keys()))
            ['co', 'recombined', 'stack_86398', 'subdivision']
            >>>     # find available channels with tag subdivision
            >>>     print(cdb.get_available_channels(
            >>>         'subdivision', 'XN-XN', 'NEP06-NEP06'))
            ['HHE-HHN', 'HHE-HHZ', 'HHN-HHZ']
            >>>     # Get Data from all times, specific channel and tag
            >>>     st = cdb.get_data(
            >>>         'XN-XN', 'NEP06-NEP06', 'HHE-HHN', 'subdivision')
            >>> print(st.count())
            250
        """

        # Create / read file
        if not path.split('.')[-1] == 'h5':
            path += '.h5'
        self.path = path
        self.mode = mode
        self.compression = compression

    def __enter__(self) -> DBHandler:
        self.db_handler = DBHandler(
            self.path, self.mode, self.compression)
        return self.db_handler

    def __exit__(self, exc_type, exc_value, tb) -> None or bool:
        self.db_handler._close()
        if exc_type is not None:
            return False


def all_traces_recursive(
    group: h5py._hl.group.Group, stream: RFStream,
        pattern: str) -> RFStream:
    """
    Recursively, appends all traces in a h5py group to the input stream.
    In addition this will check whether the data matches a certain pattern.

    :param group: group to search through
    :type group: class:`h5py._hl.group.Group`
    :param stream: Stream to append the traces to
    :type stream: CorrStream
    :param pattern: pattern for the path in the hdf5 file, see fnmatch for
        details.
    :type pattern: str
    :return: Stream with appended traces
    :rtype: CorrStream
    """
    for v in group.values():
        if not fnmatch.fnmatch(v.name, pattern) and v.name not in pattern:
            continue
        if isinstance(v, h5py._hl.group.Group):
            all_traces_recursive(v, stream, pattern)
        else:
            try:
                stream.append(
                    RFTrace(np.array(v), header=read_hdf5_header(v)))
            except ValueError:
                warnings.warn(
                    'Header could not be converted. Attributes are: %s' % (
                        str(v.attrs)))
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
            warnings.warn(
                'The header contains an item of type %s. Information\
            of this type cannot be written to an hdf5 file.'
                % str(type(header[key])), UserWarning)
            continue


def read_hdf5_header(dataset: h5py.Dataset) -> Stats:
    """
    Takes an hdft5 dataset as input and returns the header of the CorrTrace.

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
            header[key] = UTCDateTime(attrs[key])
        elif key == 'processing':
            header[key] = list(attrs[key])
        else:
            header[key] = attrs[key]
    return Stats(header)
