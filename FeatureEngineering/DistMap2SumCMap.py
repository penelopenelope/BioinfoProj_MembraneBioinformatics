#!/usr/bin/env python

import pandas as pd
import numpy as np
import os   

def Dist2CMap(dist_map, cmap_thresh):

    dist_map = dist_map.to_numpy()
    cmap = np.double(dist_map < cmap_thresh).astype(int)
    
    return cmap

def CMap2SumCMap(sequence, contact_map):

    # create a zero-filled sum of contact aa matrix - https://www.cup.uni-muenchen.de/ch/compchem/tink/as.html 
    aa20 =['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    sum_contact_aa_mat = pd.DataFrame(0, columns=aa20, index=aa20)

    # iterate the contact map to fill in the sum of contact aa matrix - 
    contact_map_pd = pd.DataFrame(contact_map)
    contact_map_pd.columns = list(sequence)
    contact_map_pd.index = list(sequence)
    up_triangle_contact_map_pd = contact_map_pd.where(np.triu(np.ones(contact_map_pd.shape)).astype(bool))
    contact_map_dict = up_triangle_contact_map_pd.stack().reset_index()
    contact_map_dict.columns = ['C1','C2','Contact']
    for index, row in contact_map_dict.iterrows(): # including the contact info of the atom to itself
        #print(row['C1'], row['C2'], row['Contact'])
        sum_contact_aa_mat.at[row['C1'], row['C2']] = sum_contact_aa_mat.at[row['C1'], row['C2']] + row['Contact']
    sum_contact_aa_mat = sum_contact_aa_mat.to_numpy()
    sum_contact_aa_mat = np.triu(sum_contact_aa_mat) + np.tril(sum_contact_aa_mat, -1).transpose()
    sum_contact_aa_mat = pd.DataFrame(sum_contact_aa_mat, columns=aa20, index=aa20)

    return sum_contact_aa_mat

#Check if folder exists, if not create folder
def check_folder_exists(folder_name):
  if not os.path.isdir(folder_name):
    os.makedirs(folder_name)

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

    # TODO - decide the exact contact threshold after finding the best performance
    distance_threshold = 10

    # create folder to save the contact maps 
    contact_map_folder_name = 'contact_maps_thresh_' + str(distance_threshold) + '/'
    check_folder_exists(contact_map_folder_name)

    # create folder to save the sum contact aa maps 
    sum_contact_aa_folder_name = 'SUM_contact_aa_maps/'
    check_folder_exists(sum_contact_aa_folder_name)

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
        dist_map_pd.columns = list(seq_info)
        dist_map_pd.index = list(seq_info)

        # generate CMap from distance maps
        contact_map = Dist2CMap(dist_map_pd, distance_threshold)
        contact_map_pd = pd.DataFrame(contact_map)
        contact_map_pd.columns = list(seq_info)
        contact_map_pd.index = list(seq_info)
        print(contact_map_pd)

        # save the contact map
        np.savetxt(contact_map_folder_name + entry_name + '_CMap', contact_map, fmt='%d')
        
        # generate sum of contact aa map from the CMap
        sum_CMap = CMap2SumCMap(seq_info, contact_map)
        print(sum_CMap)
        
        # save the sum contact aa map to *.txt file
        np.savetxt(sum_contact_aa_folder_name + entry_name + '_SUM_CMap', sum_CMap, fmt='%d')


