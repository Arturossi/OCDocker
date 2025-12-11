#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to handle I/O operations.

They are imported as:

import OCDocker.Toolbox.IO as ocio
'''

# Imports
###############################################################################
import os
import mmap

from typing import Generator

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

## Public ##
def lazyread_mmap(file_name: str, decode: str = "utf-8") -> Generator[str, None, None]:
    '''Read a file in sequential order using mmap.

    Parameters
    ----------
    file_name : str
        The file to be read.
    decode : str, optional
        The decode to be used, by default "utf-8"

    Returns
    -------
    Generator[str, None, None]
        A generator with the lines of the file in sequential order.
    '''

    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        with mmap.mmap(read_obj.fileno(), 0, access = mmap.ACCESS_READ) as mmap_obj:
            # Read line by line
            for line in iter(mmap_obj.readline, b''):
                yield line.decode(decode)


def lazyread_reverse_order_mmap(file_name: str, decode: str = "utf-8") -> Generator[str, None, None]:
    '''Read a file in reverse order using mmap.

    Parameters
    ----------
    file_name : str
        The file to be read.
    decode : str, optional
        The decode to be used, by default "utf-8"

    Returns
    -------
    Generator[str, None, None]
        A generator with the lines of the file in reverse order.
    '''
    
    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        with mmap.mmap(read_obj.fileno(), 0, access = mmap.ACCESS_READ) as mmap_obj:
            # Move the cursor to the end of the file
            mmap_obj.seek(0, os.SEEK_END)
            # Get the current position of pointer i.e eof
            pointer_location = mmap_obj.tell()
            # Create a buffer to keep the last read line
            buffer = bytearray()
            # Loop till pointer reaches the top of the file
            while pointer_location >= 0:
                # Move the file pointer to the location pointed by pointer_location
                mmap_obj.seek(pointer_location)
                # Shift pointer location by -1
                pointer_location = pointer_location - 1
                # read that byte / character
                new_byte = mmap_obj.read(1)
                # If the read byte is new line character then it means one line is read
                if new_byte == b'\n':
                    # Only yield if there is content accumulated (avoid empty line for trailing newline)
                    if len(buffer) > 0:
                        yield buffer.decode(decode)[::-1]
                        # Reinitialise the byte array to save next line
                        buffer = bytearray()
                else:
                    # If last read character is not eol then add it in buffer
                    if new_byte:
                        buffer.extend(new_byte)
            # As file is read completely, if there is still data in buffer, then its the first line.
            if len(buffer) > 0:
                # Yield the first line too
                yield buffer.decode(decode)[::-1]


def lazyread(file_name: str, decode: str = "utf-8") -> Generator[str, None, None]:
    '''Read a file in sequential order.

    Parameters
    ----------
    file_name : str
        The file to be read.
    decode : str, optional
        The decode to be used, by default "utf-8"

    Returns
    -------
    Generator[str, None, None]
        A generator with the lines of the file in sequential order.
    '''

    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        # Read line by line
        for line in iter(read_obj.readline, b''):
            yield line.decode(decode)


def lazyread_reverse_order(file_name: str, decode: str = "utf-8") -> Generator[str, None, None]:
    '''Read a file in reverse order.

    Parameters
    ----------
    file_name : str
        The file to be read.
    decode : str, optional
        The decode to be used, by default "utf-8"

    Returns
    -------
    Generator[str, None, None]
        A generator with the lines of the file in reverse order.
    '''

    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        # Move the cursor to the end of the file
        read_obj.seek(0, os.SEEK_END)
        # Get the current position of pointer i.e eof
        pointer_location = read_obj.tell()
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Loop till pointer reaches the top of the file
        while pointer_location >= 0:
            # Move the file pointer to the location pointed by pointer_location
            read_obj.seek(pointer_location)
            # Shift pointer location by -1
            pointer_location = pointer_location - 1
            # read that byte / character
            new_byte = read_obj.read(1)
            # If the read byte is new line character then it means one line is read
            if new_byte == b'\n':
                # Only yield if there is content accumulated (avoid empty line for trailing newline)
                if len(buffer) > 0:
                    yield buffer.decode(decode)[::-1]
                    # Reinitialie the byte array to save next line
                    buffer = bytearray()
            else:
                # If last read character is not eol then add it in buffer
                if new_byte:
                    buffer.extend(new_byte)
        # As file is read completely, if there is still data in buffer, then its the first line.
        if len(buffer) > 0:
            # Yield the first line too
            yield buffer.decode(decode)[::-1]
