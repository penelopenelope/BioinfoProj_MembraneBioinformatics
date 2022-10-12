#!/usr/bin/env python

import pandas as pd
import numpy as np
import os

def DistMap2AvgDistMap(func, df_contact, null_value=0):

    records = []
    for aa1 in aa20:
        for aa2 in aa20:
            if aa1 not in df_contact.columns or aa2 not in df_contact.columns:
                records.append([aa1, aa2, null_value])
                continue
            contacts = df_contact.loc[aa1, aa2]
            #print(aa1, aa2)
            #print(contacts)
            # contacts can be an int, a series or a dataframe
            if isinstance(contacts, (int, np.integer)):
                records.append([aa1, aa2, (contacts < threshold) * 1])
                #print('int!!!!!!!!!!!!!')
            else:  # series or a dataframe
                np_array = contacts.to_numpy()
                #print(np_array)
                if isinstance(contacts, pd.DataFrame):
                    if aa1 == aa2: 
                        temp_mat = np.full((np_array.shape), 10000)
                        np_array = np.triu(np_array) + np.tril(temp_mat, -1)
                    np_array = np_array.flatten()
                    #print('DataFrame')

                # check if array is really 1d
                assert len(np.shape(np_array)) == 1

                np_array_transformed = func(np_array)
                #print('np_array_transformed - ', np_array_transformed)
                records.append([aa1, aa2, np_array_transformed])

    df_result_long = pd.DataFrame.from_records(records, columns=["aa1", "aa2", "value"])
    df_result = df_result_long.pivot(index="aa1", columns="aa2", values="value")
    return df_result


#Check if folder exists, if not create folder
def check_folder_exists(folder_name):
  if not os.path.isdir(folder_name):
    os.makedirs(folder_name)

aa20 =['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


if __name__ == "__main__":
  
    # read distance maps in folder
    #dist_maps_dir = 'TestDistMaps/'
    dist_maps_dir = '../Datasets/CA_dist_maps/'
    directory = os.fsdecode(dist_maps_dir)
    #print(directory)

    # extract entry and sequence columns from helix_sheet dataset
    hf_pd = pd.read_csv('../Datasets/AF_helix_sheet.tsv', sep='\t', header=0)
    entry = hf_pd[["Entry"]]
    seq = hf_pd[["Sequence"]]
    entry_seq_pd = pd.concat([entry, seq], axis=1)
    entry_seq_pd = entry_seq_pd.set_index('Entry')
    #print(entry_seq_pd)

    # create folder to save the average distance aa maps 
    average_aa_distance_folder_name = 'AVG_aa_dist_maps/'
    check_folder_exists(average_aa_distance_folder_name)

    for dist_map_filename in os.listdir(directory):

        # read sequence by entry names info from AF_helix_sheet.tsv 
        entry_name = dist_map_filename[:-12]
        #print(entry_name)
        seq_info = entry_seq_pd.at[entry_name, 'Sequence']
        #print(seq_info)

        # generate average distance maps according to the entry name and sequence information
        #print(dist_map_filename)
        d_map_dir = os.fsdecode(dist_maps_dir + dist_map_filename)
        #print(os.path.abspath(d_map_dir))
        df_contact = pd.read_table(d_map_dir, header=None, sep=" ")
        avg_aa_map = DistMap2AvgDistMap(np.mean, df_contact, null_value=None) # uncheck yet
        
        # save the average_aa_mat to *.txt file
        np.savetxt(average_aa_distance_folder_name + entry_name + '_AVG_CA_dist_map', avg_aa_map, fmt='%1.2f')


