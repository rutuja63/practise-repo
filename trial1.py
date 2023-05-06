# Import packages
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import random

# Read specific packages from PhoREAL
# from phoreal.reader import get_atl03_struct
# from phoreal.reader import get_atl08_struct
# from phoreal.reader import get_atl_alongtrack
# from phoreal.binner import rebin_atl08
# from phoreal.binner import rebin_truth
# from phoreal.binner import match_truth_fields

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 08:26:42 2021
@author: eguenther
original file accessed at: https://github.com/icesat-2UT/PhoREAL/blob/master/source_code/background_noise_correction.py
Edits by Hazel Davies commented in code with "HD"
"""


import numpy as np
import pandas as pd
import h5py
import os
import argparse

def getCmdArgs():
    '''HD - command line arguments to run file through Linux terminal'''
    parser = argparse.ArgumentParser()

    parser.add_argument("dir03", help = "Folder directory containing ATL03 .h5 files")
    parser.add_argument("dir08", help = "Folder directory containing ATL08 .h5 files")
    parser.add_argument("outfile", help = "file path directory for output files folder")

    args = parser.parse_args()

    return args

def get_len_unique(series):
    try:
        length = len(np.unique(series))
    except:
        length = np.nan
    return length

def get_len(series):
    try:
        length = len(series)
    except:
        length = np.nan
    return length

def ismember(a_vec, b_vec, methodType = 'normal'):

    """ MATLAB equivalent ismember function """

    # Combine multi column arrays into a 1-D array of strings if necessary
    # This will ensure unique rows when using np.isin below
    if(methodType.lower() == 'rows'):

        # Turn a_vec into an array of strings
        a_str = a_vec.astype('str')
        b_str = b_vec.astype('str')

        # Concatenate each column of strings with commas into a 1-D array
        for i in range(0,np.shape(a_str)[1]):
            a_char = np.char.array(a_str[:,i])
            b_char = np.char.array(b_str[:,i])
            if(i==0):
                a_vec = a_char
                b_vec = b_char
            else:
                a_vec = a_vec + ',' + a_char
                b_vec = b_vec + ',' + b_char

    # Find which values in a_vec are present in b_vec
    matchingTF = np.isin(a_vec,b_vec)
    common = a_vec[matchingTF]
    common_unique, common_inv  = np.unique(common, return_inverse=True)
    b_unique, b_ind = np.unique(b_vec, return_index=True)
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    matchingInds = common_ind[common_inv]

    return matchingTF, matchingInds

def gtx_radiometry(file_03, file_08, gt, d, outfile_name):
    ''' HD change function name and function arguments to incorporate command line arguments
    and ancillary data needed for investigation'''

    #identify weak/strong and night values from 08 file for single groundtrack
    #sc_orient ~ Backward = 0 (strong leading weak), Forward = 1 (weak leading strong)
    s = np.asarray(file_03['/orbit_info/sc_orient'], dtype = bool)
    #connect s to groundtrack direction for strong/weak
    # s ~ True = 1 (FORWARD), False = 0 (BACKWARD)
    #gtf 3 == equiv to l track. if not, -1 = r track.
    gtf = gt.find('l')
    if (s == False and gtf == 3) or (s == True and gtf == -1):
        s_bool = True
    else:
        s_bool = False

    #read 03 and 08 files for necessary data

    #03 data
    segment_ph_count = np.array(file_03[gt + '/geolocation/segment_ph_cnt'])
    atl03_segment_id = np.array(file_03[gt + '/geolocation/segment_id'])
    atl03_ph_index_beg = np.array(file_03[gt + '/geolocation/ph_index_beg'])
    bihr = np.asarray(file_03[gt + '/bckgrd_atlas/bckgrd_int_height_reduced'])
    bcr = np.asarray(file_03[gt + '/bckgrd_atlas/bckgrd_counts_reduced'])
    bapmc = np.asarray(file_03[gt + '/bckgrd_atlas/pce_mframe_cnt'])
    h_ph = np.asarray(file_03[gt + '/heights/h_ph'])
    hpmc = np.asarray(file_03[gt + '/heights/pce_mframe_cnt'])
    delta_time = np.asarray(file_03[gt + '/heights/delta_time'])

    #08 data
    seg_beg = np.asarray(file_08[gt + '/land_segments/segment_id_beg']).astype(int)
    seg_end = np.asarray(file_08[gt + '/land_segments/segment_id_end']).astype(int)
    latitude = np.asarray(file_08[gt + '/land_segments/latitude'])
    longitude = np.asarray(file_08[gt + '/land_segments/longitude'])
    n = np.asarray(file_08[gt + '/land_segments/night_flag'], dtype=bool)
    msw = np.asarray(file_08[gt +'/land_segments/msw_flag'])

    # Get segment ID to photon level
    h_seg = np.zeros(len(h_ph))
    for i in range(0,len(atl03_segment_id)):
        if atl03_ph_index_beg[i] > 0:
            h_seg[atl03_ph_index_beg[i]-1:atl03_ph_index_beg[i]-1 + segment_ph_count[i]] = atl03_segment_id[i]
    h_seg = np.int32(h_seg)

    # Calculate rate and assign to the photon level
    rate = bcr / bihr

    # Assign bckgrd_atlas attributes to photon level
    tf, inds = ismember(hpmc, bapmc)
    inds.resize(((len(h_ph)),),refcheck=False)
    ph_bihr = bihr[inds]
    ph_bcr = bcr[inds]
    ph_rate = rate[inds]

    # Create ATL03 indices (based on segment_id)
    h_ind = np.zeros(len(h_seg))
    h_ind[:] = -1

    # Link ATL03 segments to ATL08 segments
    for i in range(0,len(seg_beg)):
        h_ind[np.logical_and(h_seg >= seg_beg[i], h_seg <= seg_end[i])] = i

    # Put everything in ATL03 into pandas df
    df = pd.DataFrame({'h_ind': h_ind,
                       'delta_time': delta_time,
                       'bckgrd_int_height_reduced':ph_bihr,
                       'bckgrd_counts_reduces': ph_bcr,
                       'bckgrd_rate': ph_rate})

    # Remove ATL03 files that do not fit into ATL08 segment
    df = df[df['h_ind'] >= 0]

    # Average Background Rate to ATL08
    zgroup = df.groupby('h_ind')
    zout = zgroup.aggregate(np.nanmean)
    f08_rate = np.asarray(zout['bckgrd_rate'])
    h_max_canopy = np.asarray(file_08[gt + '/land_segments/canopy/h_max_canopy'])
    h_max_canopy[h_max_canopy > 100000] = np.nan
    canopy_noise_count = (f08_rate) * h_max_canopy

    # Count the number of photons at the ATL08 segment
    n_ca_photons = np.asarray(file_08[gt + '/land_segments/canopy/n_ca_photons'])
    n_toc_photons = np.asarray(file_08[gt + '/land_segments/canopy/n_toc_photons'])
    n_te_photons = np.asarray(file_08[gt + '/land_segments/terrain/n_te_photons'])

    # Calculate 'noise adjusted' number of canopy photons at ATL08 segment
    n_ca_photons_nr = (n_ca_photons + n_toc_photons) - canopy_noise_count

    # Calculate photon rate (radiometry)
    zgroup = df.groupby('h_ind')
    zout = zgroup.aggregate(get_len_unique)
    f08_shots = np.asarray(zout['delta_time'])

    zgroup = df.groupby('h_ind')
    zout = zgroup.aggregate(get_len)

    photon_rate_can_nr = n_ca_photons_nr / f08_shots
    photon_rate_can = (n_ca_photons + n_toc_photons) / f08_shots
    photon_rate_can[np.isnan(photon_rate_can_nr)] = np.nan
    photon_rate_signal = (n_te_photons + n_ca_photons + n_toc_photons)        / f08_shots
    photon_rate_ground = n_te_photons / f08_shots

    #create dataframe csv output of all counts and one for input into matts code
    output = pd.DataFrame({'date': d,
                          'gtx': gt,
                          'latitude': latitude,
                          'longitude': longitude,
                          'beam strength': s_bool,
                          'night flag': n,
                          'msw' : msw,
                          'noise reduced canopy photon rate': photon_rate_can_nr,
                          'canopy photon rate': photon_rate_can,
                          'ground photon rate': photon_rate_ground,
                          'signal photon rate': photon_rate_signal,
                            'f08 shots': f08_shots,
                           'canopy noise count' : canopy_noise_count
                          })

    print(output)
    output.to_csv(outfile_name + '_all' + '.csv')


if __name__ == '__main__':
    ''' HD '''
    cmd = getCmdArgs()

    dir03 = cmd.dir03
    dir08 = cmd.dir08
    outfile = cmd.outfile

    gtx = ['gt1r','gt1l','gt2r', 'gt2l', 'gt3r', 'gt3l']

    dir_array = np.asarray(os.listdir(dir03))
    dir_array = dir_array[23:31]


    #for i in gtx:
    gt = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']

    for index in dir_array:
        atl03file = index

        atl08file = 'ATL08' + atl03file.split('ATL03')[1]
        atl03filepath = os.path.join(dir03, atl03file)
        atl08filepath = os.path.join(dir08, atl08file)

        date = os.path.basename(atl03filepath)

        date_split = date.split("_")

        d = date_split[1][0:8]

        outfile_path = os.path.join(outfile + "/ph_cnt")
        if not os.path.exists(outfile_path):
            os.mkdir(outfile_path)

        outfile_name = os.path.join(outfile_path, d + '_' + gt)


        file_03 = h5py.File(atl03filepath, 'r')
        file_08 = h5py.File(atl08filepath, 'r')


        gtx_radiometry(file_03, file_08, gt, d, outfile_name)