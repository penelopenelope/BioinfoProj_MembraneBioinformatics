#!/usr/bin/env python

import pandas as pd
import numpy as np
import os   
from joblib import Parallel, delayed

def Dist2CMap(dist_map, cmap_thresh):

    dist_map = dist_map.to_numpy()
    cmap = np.double(dist_map < cmap_thresh).astype(int)
    
    return cmap

def CMap2SumCMap(func, df_contact, null_value=0):

# creates a subset for every combination of amino acid types, then applies func to all numbers in the subset
# the result of func has to be a single number
    records = []
    for aa1 in aa20:
        for aa2 in aa20:
            if aa1 not in df_contact.columns or aa2 not in df_contact.columns:
                records.append([aa1, aa2, null_value])
                continue
            contacts = df_contact.loc[aa1, aa2]
            print(aa1, aa2)
            print(contacts)
            # contacts can be an int, a series or a dataframe
            if isinstance(contacts, (int, np.integer)):
                records.append([aa1, aa2, (contacts < distance_threshold) * 1])
                #print('int!!!!!!!!!!!!!')
            else:  # series or a dataframe
                np_array = contacts.to_numpy()
                print(np_array)
                if isinstance(contacts, pd.DataFrame):
                    if aa1 == aa2: 
                        temp_mat = np.full((np_array.shape), 10000)
                        np_array = np.triu(np_array) + np.tril(temp_mat, -1)
                    np_array = np_array.flatten()
                    #print('DataFrame')

                # check if array is really 1d
                assert len(np.shape(np_array)) == 1

                np_array_transformed = func(np_array)
                print('np_array_transformed - ', np_array_transformed)
                records.append([aa1, aa2, np_array_transformed])

    df_result_long = pd.DataFrame.from_records(records, columns=["aa1", "aa2", "value"])
    df_result = df_result_long.pivot(index="aa1", columns="aa2", values="value")
    return df_result

def create_features(entry_name, dist_map_pd, seq_info, distance_threshold, contact_map_folder_name, sum_contact_aa_folder_name):

    print('entry name - ', entry_name)
    dist_map_pd.columns = list(seq_info)
    dist_map_pd.index = list(seq_info)
    print('seq information - ', seq_info)

    # generate CMap from distance maps
    contact_map = Dist2CMap(dist_map_pd, distance_threshold)
    contact_map_pd = pd.DataFrame(contact_map)
    contact_map_pd.columns = list(seq_info)
    contact_map_pd.index = list(seq_info)
    #print(contact_map_pd)

    # save the contact map
    np.savetxt(contact_map_folder_name + entry_name + '_CMap', contact_map, fmt='%d')
    
    # generate sum of contact aa map from the CMap
    sum_CMap = CMap2SumCMap(lambda a: ((a < distance_threshold) * 1).sum(), dist_map_pd)
    print(sum_CMap)
    
    # save the sum contact aa map to *.txt file
    np.savetxt(sum_contact_aa_folder_name + entry_name + '_SUM_CMap', sum_CMap, fmt='%d')

    return True

#Check if folder exists, if not create folder
def check_folder_exists(folder_name):
  if not os.path.isdir(folder_name):
    os.makedirs(folder_name)

if __name__ == "__main__":
  

    # Parameters - 

    # read distance maps in folder
    dist_maps_dir = '../Datasets/TestDistMaps/'
    #dist_maps_dir = '../Datasets/CA_dist_maps/'
    directory = os.fsdecode(dist_maps_dir)
    #print(directory)

    # extract entry and sequence columns from helix_sheet dataset
    hf_pd = pd.read_csv('../Datasets/AF_helix_sheet.tsv', sep='\t', header=0)
    entry = hf_pd[["Entry"]]
    seq = hf_pd[["Sequence"]]
    entry_seq_pd = pd.concat([entry, seq], axis=1)
    entry_seq_pd = entry_seq_pd.set_index('Entry')
    #print(entry_seq_pd)

    aa20 =['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    # TODO - decide the exact contact threshold after finding the best performance
    distance_threshold = 20

    # create folder to save the contact maps 
    contact_map_folder_name = 'contact_maps_thresh_' + str(distance_threshold) + '/'
    check_folder_exists(contact_map_folder_name)

    # create folder to save the sum contact aa maps 
    sum_contact_aa_folder_name = 'SUM_contact_aa_maps_' + str(distance_threshold) + '/'
    check_folder_exists(sum_contact_aa_folder_name)

    notMatch = []
    
    # create list of "create_features" parameter values for the individual threads
    # important: no two elements of worker_inputs can be the identical, otherwise unexpected behaviour
    worker_inputs = []
    for dist_map_filename in os.listdir(directory):

        # read sequence by entry names info from AF_helix_sheet.tsv 
        entry_name = dist_map_filename[:-12]
        print(entry_name)
        seq_info = entry_seq_pd.at[entry_name, 'Sequence']
        print(seq_info)

        # read the distance map and conbine it with sequence info to generate distance matrix
        #print(dist_map_filename)
        d_map_dir = os.fsdecode(dist_maps_dir + dist_map_filename)
        #print(os.path.abspath(d_map_dir))        
        dist_map = pd.read_csv(d_map_dir, delim_whitespace=True, header=None)
        dist_map_pd = pd.DataFrame(dist_map)

        # check if the length of seq_info is the same as in the pdb file
        print(len(seq_info), len(dist_map_pd))
        if len(seq_info) != len(dist_map_pd):
            notMatch.append(entry_name)
            print('Not match between hlix_sheet file and pdb file - ', notMatch)
            continue

        worker_inputs.append(
            [
                entry_name,
                dist_map_pd,
                seq_info,
                distance_threshold,
                contact_map_folder_name,
                sum_contact_aa_folder_name,
            ]
        )


    # https://joblib.readthedocs.io/en/latest/parallel.html#parallel
    r = Parallel(n_jobs=-1)(
        delayed(create_features)(*worker_input) for worker_input in worker_inputs
    )

    np.savetxt('notMatch', notMatch, fmt="%s")

