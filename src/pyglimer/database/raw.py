'''
HDF5 based data format to save raw waveforms and station XMLs. Works similar
to the data format saving receiver functions.

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 6th September 2022 10:37:12 am
Last Modified: Thursday, 13th October 2022 09:50:17 am
'''

import fnmatch
from io import BytesIO
import logging
import os
import re
from typing import Iterable
import warnings
import glob

import numpy as np
import obspy
from obspy.core.utcdatetime import UTCDateTime
from obspy.core import Stats, Stream, Trace, read
from obspy import Inventory, read_inventory
import h5py


hierarchy = "/{tag}/{network}/{station}/{starttime}/{channel}"
hierarchy_xml = '/{tag}/{network}/{station}'
h5_FMTSTR = os.path.join("{dir}", "{network}.{station}.h5")


class DBHandler(h5py.File):
    """
    The actual file handler of the hdf5 receiver function files.

    .. warning::

        **Should not be accessed directly. Access
        :class:`~pyglimer.database.raw.RawDataBase` instead.**

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

    def _define_content(self, content: dict):
        """
        Define the known waveforms per waveform.
        Known waveforms to receive. Is just a dictionary with one key for
        each channel with the values being a list of available waveform
        starttimes as a string in format_fissure until [:-4].

        :param content: dictionary with one key for
            each channel with the values being a list of available waveform
            starttimes
        :type ret: dict

        .. note:: This overwrites the old table of contents, so make sure
            to add everything!
        """
        try:
            ds = self.create_dataset('content', data=np.empty(1))
        except ValueError:
            ds = self['content']
            # Already existing, just change attributes
        ds.attrs['content'] = str(content)

    def add_waveform(
            self, data: Trace or Stream, tag: str = 'raw'):
        """
        Add receiver function to the hdf5 file. The data can later be accessed
        using the :meth:`~pyglimer.database.rfh5.DBHandler.get_data()` method.

        :param data: Data to save.
        :type data: Trace or Stream
        :param tag: The tag that the data should be saved under. Defaults to
            'raw'
        :raises TypeError: for wrong data type.
        """
        if not isinstance(data, Trace) and\
                not isinstance(data, Stream):
            raise TypeError('Data has to be either an obspy Trace or Stream')

        if isinstance(data, Trace):
            data = [data]

        for tr in data:
            st = tr.stats
            path = hierarchy.format(
                tag=tag, network=st.network, station=st.station,
                starttime=st.starttime.format_fissures()[:-4],
                channel=st.channel)
            try:
                ds = self.create_dataset(
                    path, data=tr.data, compression=self.compression,
                    compression_opts=self.compression_opts)
                convert_header_to_hdf5(ds, st)
            except ValueError:
                warnings.warn("The dataset %s is already in file and will be \
omitted." % path, category=UserWarning)

    def add_response(self, inv: Inventory, tag: str = 'response'):
        """
        Add the response information (i.e., stationxml).

        :param inv: inventory containing information of only **one**
            station
        :type inv: Inventory
        :param tag: tag to save under, defaults to 'response'
        :type tag: str, optional

        .. note:: The Inventory `inv` should only contain information of one
            station.
        """
        network = inv[0].code
        station = inv[0][0].code
        path = hierarchy_xml.format(tag=tag, network=network, station=station)
        # Writing as bytes to avoid slow xml in hdf5
        with BytesIO() as b:
            inv.write(b, format='stationxml')
            b.seek(0, 0)
            try:
                self.create_dataset(
                    path, data=np.frombuffer(b.read(), dtype=np.dtype('byte')),
                    compression=self.compression,
                    compression_opts=self.compression_opts)
            except ValueError as e:
                print(e)
                warnings.warn("The dataset %s is already in file and will be \
omitted." % path, category=UserWarning)

    def get_data(
        self, network: str, station: str, starttime: UTCDateTime = None,
            endtime: UTCDateTime = None, tag: str = 'raw') -> Stream:
        """
        Returns an obspy Stream holding the requested data and all
        existing channels.

        .. note::

            `Wildcards are allowed for all parameters`.

        :param network: network code, e.g., IU
        :type network: str
        :param station: station code, e.g., HRV
        :type station: str
        :param starttime: Starttime of the Trace
        :type starttime: UTCDateTime, optional
        :param tag: Data tag (e.g., 'raw'). Defaults to raw.
        :type tag: str, optional
        :return: a :class:`Stream` holding the requested
            data.
        :rtype: Stream
        """
        try:
            search_time = starttime.format_fissures()[:-10] + '*'
            # starttime = UTCDateTime(starttime)
            # starttime = starttime.format_fissures()[:-4]
        except (TypeError, AttributeError):
            search_time = '*'

        path = hierarchy.format(
            tag=tag, network=network, station=station, starttime=search_time,
            channel="*")
        # Now, we need to differ between the fnmatch pattern and the actually
        # acessed path
        pattern = path.replace('/*', '*')
        path = path.split('*')[0]
        path = os.path.dirname(path)
        try:
            st = all_traces_recursive(self[path], Stream(), pattern)
            if isinstance(starttime, UTCDateTime):
                st = st.slice(starttime=starttime, endtime=endtime)
            return st
        except KeyError:
            warnings.warn(
                f'Could not find data from {network}.{station} for'
                + "and time f'{starttime}. Returning empty Stream.")
            return Stream()

    def get_response(
        self, network: str, station: str,
            tag: str = 'response') -> Inventory:
        """
        Get the Response information for the queried station.

        :param network: Network Code
        :type network: str
        :param station: Station Code
        :type station: str
        :param tag: Tag under which the response is saved,
            defaults to 'response'
        :type tag: str, optional
        :return: Obspy Inventory
        :rtype: Inventory
        """
        path = hierarchy_xml.format(tag=tag, network=network, station=station)
        with BytesIO(np.array(self[path], dtype=np.dtype('byte'))) as b:
            return read_inventory(b)

    def _get_table_of_contents(self) -> dict:
        """
        Retrieve the contents of the file.

        :return: A dictionary where the keys are the available channels and the
            values lists of the available starttimes in rounded format fissure
            on 10 seconds.
        :rtype: dict
        """
        try:
            ds = self['content']
            content = eval(ds.attrs['content'])
            return content
        except (KeyError, AttributeError):
            return {}

    def walk(
            self, tag: str, network: str, station: str) -> Iterable[Stream]:
        """
        Iterate over all Streams with the given properties.
        (i.e, all starttimes)

        :param tag: data tag
        :type tag: str
        :param network: Network code
        :type network: str
        :param station: Statio ncode
        :type station: str
        :return: Iterator
        :rtype: Iterable[RFTrace]
        :yield: one Stream per starttime.
        :rtype: Iterator[Iterable[Stream]]

        .. note::

            Does not accept wildcards.
        """
        for w in self[tag][network][station].values():
            st = Stream()
            for v in w.values():
                st.append(Trace(np.array(v), header=read_hdf5_header(v)))
            yield st


class RawDatabase(object):
    """
    Base class to handle the hdf5 files that contain raw data for receiver
    function computaiton.
    """
    def __init__(
            self, path: str, mode: str = 'a', compression: str = 'gzip3'):
        """
        Access an hdf5 file holding raw waveforms. The resulting file can
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

            >>> with RawDataBase('myfile.h5') as rdb:
            >>>     type(rdb)  # This is a DBHandler
            <class 'pyglimer.database.rfh5.DBHandler'>

        Example::

            >>> with RawDataBase(
                        '/path/to/db/XN.NEP06.h5') as rfdb:
            >>>     # find the available tags for existing db
            >>>     print(list(rfdb.keys()))
            ['raw', 'weird']
            >>>     # Get Data from all times and
            >>>     st = rfdb.get_data(
            >>>         'XN', 'NEP06', '*')
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
    group: h5py._hl.group.Group, stream: Stream,
        pattern: str) -> Stream:
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
    :return: Stream with appended traces
    :rtype: Stream
    """
    for v in group.values():
        if isinstance(v, h5py._hl.group.Group):
            all_traces_recursive(v, stream, pattern)
        elif not fnmatch.fnmatch(v.name, pattern) and v.name not in pattern:
            continue
        else:
            stream.append(
                Trace(np.array(v), header=read_hdf5_header(v)))
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


def mseed_to_hdf5(
    rawfolder: str or os.PathLike, save_statxml: bool,
        statloc: str = None):
    """
    Convert a given mseed database in rawfolder to an h5 based database.

    :param rawfolder: directory in which the fodlers for each event are stored
    :type rawfolder: os.PathLike
    :param save_statxml: should stationxmls be read as well? If so, you need
        to provide `statloc` as well.
    :type save_statxml: bool
    :param statloc: location of station xmls only needed if `save_statxml` is
        is set to True, defaults to None
    :type statloc: str, optional
    """
    av_mseed = glob.glob(os.path.join(rawfolder, '*', '*.mseed'))
    # We do that for every single station
    if len(av_mseed) == 0:
        # Delete empty folders
        evt_dirs = glob.glob(os.path.join(rawfolder, '*_*_*'))
        for d in evt_dirs:
            os.rmdir(d)
        if save_statxml:
            statxml_to_hdf5(rawfolder, statloc)
        return
    try:
        st = read(av_mseed[0])
    except obspy.io.mseed.InternalMSEEDError:
        # broken mseed
        logger = logging.getLogger('pyglimer.request')
        logger.warning(f'File {av_mseed[0]} is corrupt. Skipping this file..')
        os.remove(av_mseed[0])
        mseed_to_hdf5(rawfolder, save_statxml, statloc=statloc)
    try:
        if not len(st):
            # broken mseed
            logger = logging.getLogger('pyglimer.request')
            logger.warning(
                f'File {av_mseed[0]} is corrupt. Skipping this file..')
            os.remove(av_mseed[0])
            mseed_to_hdf5(rawfolder, save_statxml, statloc=statloc)
        net = st[0].stats.network
        stat = st[0].stats.station
    except UnboundLocalError as e:
        # broken mseed Don't really understand why this happens..
        logger = logging.getLogger('pyglimer.request')
        logger.error(
            f'File {av_mseed[0]} is corrupt. Skipping this file..\n'
            + f'The original error was {e} on Stream {st}')
        try:
            os.remove(av_mseed[0])
        except FileNotFoundError:
            # This might actually be the reason for hte Unboundlocalerror
            pass
        mseed_to_hdf5(rawfolder, save_statxml, statloc=statloc)

    # Now, read all available files for this station
    mseeds = os.path.join(rawfolder, '*', f'{net}.{stat}.mseed')
    try:
        st = read(mseeds)
    except obspy.io.mseed.InternalMSEEDError:
        # Read each of them and remove the broken files
        st = Stream()
        for mseed in glob.glob(mseeds):
            try:
                st.extend(read(mseed))
            except obspy.io.mseed.InternalMSEEDError:
                logger = logging.getLogger('pyglimer.request')
                logger.warning(
                    f'File {mseed} is corrupt. Skipping this file..')
                os.remove(mseed)
        if not st.count():
            mseed_to_hdf5(rawfolder, save_statxml, statloc=statloc)

    # Create table of new contents
    new_cont = {}
    for tr in st:
        new_cont.setdefault(tr.stats.channel, [])
        new_cont[tr.stats.channel].append(
            tr.stats.starttime.format_fissures()[:-4])

    h5_file = os.path.join(rawfolder, f'{net}.{stat}.h5')
    # Write all of them to the hdf5 file
    with RawDatabase(h5_file) as rdb:
        old_cont = rdb._get_table_of_contents()
        for k, v in old_cont.items():
            try:
                new_cont[k].extend(v)
            except KeyError:
                new_cont[k] = v
        rdb.add_waveform(st)
        if save_statxml:
            statxml = os.path.join(statloc, f'{net}.{stat}.xml')
            rdb.add_response(read_inventory(statxml))
        rdb._define_content(new_cont)
    # Clear RAM
    st.clear()
    # If all of that was successful, we can remove the mseeds
    for f in glob.glob(mseeds):
        os.remove(f)
    if save_statxml:
        os.remove(statxml)
    # next station
    mseed_to_hdf5(rawfolder, save_statxml, statloc=statloc)


def statxml_to_hdf5(
        rawfolder: str or os.PathLike, statloc: str or os.PathLike):
    """
    Write a statxml database to h5 files

    :param rawfolder: location that contains the h5 files
    :type rawfolder: stroros.PathLike
    :param statloc: location where the stationxmls are stored
    :type statloc: stroros.PathLike
    """
    av_xml = glob.glob(os.path.join(statloc, '*.xml'))
    for xml in av_xml:
        try:
            inv = read_inventory(xml)
            net = inv[0].code
            stat = inv[0][0].code
            h5_file = os.path.join(rawfolder, f'{net}.{stat}.h5')
            with RawDatabase(h5_file) as rdb:
                rdb.add_response(inv)
                os.remove(xml)
        except Exception as e:
            logging.warning(f'Station XML could not be added to h5 file. {e}')
