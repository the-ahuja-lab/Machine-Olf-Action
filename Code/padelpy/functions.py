#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# padelpy/functions.py
# v.0.1.6
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains various functions commonly used with PaDEL-Descriptor
#

# stdlib. imports
from collections import OrderedDict
from csv import DictReader
from datetime import datetime
from os import remove
from re import compile, IGNORECASE
from time import sleep

# PaDELPy imports
from padelpy import padeldescriptor


def from_smiles(smiles: str, output_csv: str=None, descriptors: bool=True,
                fingerprints: bool=False, timeout: int=12, java_path: str=None) -> OrderedDict:
    ''' from_smiles: converts SMILES string to QSPR descriptors/fingerprints

    Args:
        smiles (str): SMILES string for a given molecule
        output_csv (str): if supplied, saves descriptors to this CSV file
        descriptors (bool): if `True`, calculates descriptors
        fingerprints (bool): if `True`, calculates fingerprints
        timeout (int): maximum time, in seconds, for conversion
        java_path(str): custom java path, if None, system java path

    Returns:
        OrderedDict: descriptors/fingerprint labels and values
    '''

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]

    with open('{}.smi'.format(timestamp), 'w') as smi_file:
        smi_file.write(smiles)
    smi_file.close()

    save_csv = True
    if output_csv is None:
        save_csv = False
        output_csv = '{}.csv'.format(timestamp)

    for attempt in range(3):
        try:
            padeldescriptor(
                mol_dir='{}.smi'.format(timestamp),
                d_file=output_csv,
                convert3d=True,
                retain3d=True,
                d_2d=descriptors,
                d_3d=descriptors,
                fingerprints=fingerprints,
                sp_timeout=timeout,
                java_path=java_path
            )
            break
        except RuntimeError as exception:
            if attempt == 2:
                remove('{}.smi'.format(timestamp))
                if not save_csv:
                    sleep(0.5)
                    remove(output_csv)
                raise RuntimeError(exception)
            else:
                continue

    with open(output_csv, 'r', encoding='utf-8') as desc_file:
        reader = DictReader(desc_file)
        rows = [row for row in reader]
    desc_file.close()

    remove('{}.smi'.format(timestamp))
    if not save_csv:
        remove(output_csv)

    del rows[0]['Name']
    return rows[0]


def from_mdl(mdl_file: str, output_csv: str=None, descriptors: bool=True,
             fingerprints: bool=False, timeout: int=12) -> list:
    ''' from_mdl: converts MDL file into QSPR descriptors/fingerprints;
    multiple molecules may be represented in the MDL file

    Args:
        mdl_file (str): path to MDL file
        output_csv (str): if supplied, saves descriptors/fingerprints here
        descriptors (bool): if `True`, calculates descriptors
        fingerprints (bool): if `True`, calculates fingerprints
        timeout (int): maximum time, in seconds, for conversion

    Returns:
        list: list of dicts, where each dict corresponds sequentially to a
            compound in the supplied MDL file
    '''

    is_mdl = compile(r'.*\.mdl$', IGNORECASE)
    if is_mdl.match(mdl_file) is None:
        raise ValueError('MDL file must have a `.mdl` extension: {}'.format(
            mdl_file
        ))

    save_csv = True
    if output_csv is None:
        save_csv = False
        output_csv = '{}.csv'.format(
            datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        )

    for attempt in range(3):
        try:
            padeldescriptor(
                mol_dir=mdl_file,
                d_file=output_csv,
                convert3d=True,
                retain3d=True,
                retainorder=True,
                d_2d=descriptors,
                d_3d=descriptors,
                fingerprints=fingerprints,
                sp_timeout=timeout
            )
            break
        except RuntimeError as exception:
            if attempt == 2:
                if not save_csv:
                    sleep(0.5)
                    remove(output_csv)
                raise RuntimeError(exception)
            else:
                continue

    with open(output_csv, 'r', encoding='utf-8') as desc_file:
        reader = DictReader(desc_file)
        rows = [row for row in reader]
    desc_file.close()
    if not save_csv:
        remove(output_csv)
    for row in rows:
        del row['Name']
    return rows
