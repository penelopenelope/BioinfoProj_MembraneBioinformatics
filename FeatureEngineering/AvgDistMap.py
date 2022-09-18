#!/usr/bin/env python

import pandas as pd
import numpy as np
import os

def DistMap2AvgDistMap(sequence, distance_map_dir):

    # read the distance map and conbine it with sequence info to generate distance matrix
    dist_map = pd.read_csv(distance_map_dir, delim_whitespace=True, header=None)
    dist_map_pd = pd.DataFrame(dist_map)
    dist_map_pd.columns = list(sequence)
    dist_map_pd.index = list(sequence)

    # create a zero-filled sum of aa distance matrix - https://www.cup.uni-muenchen.de/ch/compchem/tink/as.html 
    aa20 =['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    sum_dist_aa_mat = pd.DataFrame(0, columns=aa20, index=aa20)

    # create a zero-filled matrix to store the number of each pairs
    nr_aa_pairs = pd.DataFrame(0, columns=aa20, index=aa20)

    # iterate the distance map to fill in the sum of aa distance matrix - 
    # https://stackoverflow.com/questions/34417685/melt-the-upper-triangular-matrix-of-a-pandas-dataframe 
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    # https://stackoverflow.com/questions/46342953/is-it-possible-to-divide-one-dataframe-from-another-dataframe
    up_triangle_dist_map_pd = dist_map_pd.where(np.triu(np.ones(dist_map_pd.shape)).astype(bool))
    dist_map_dict = up_triangle_dist_map_pd.stack().reset_index()
    dist_map_dict.columns = ['C1','C2','Dist']
    for index, row in dist_map_dict.iterrows():
        #print(row['C1'], row['C2'], row['Dist'])
        sum_dist_aa_mat.at[row['C1'], row['C2']] = sum_dist_aa_mat.at[row['C1'], row['C2']] + row['Dist']
        nr_aa_pairs.at[row['C1'], row['C2']] += 1      
    sum_dist_aa_mat_np = sum_dist_aa_mat.to_numpy()
    sum_dist_aa_mat_np = np.triu(sum_dist_aa_mat_np) + np.tril(sum_dist_aa_mat_np, -1).transpose()
    sum_dist_aa_mat_np = sum_dist_aa_mat_np.astype(float)
    sum_dist_aa_mat = pd.DataFrame(sum_dist_aa_mat_np, columns=aa20, index=aa20)
    nr_aa_pairs_np = nr_aa_pairs.to_numpy()
    nr_aa_pairs_np = np.triu(nr_aa_pairs_np) + np.tril(nr_aa_pairs_np, -1).transpose()
    nr_aa_pairs_np = nr_aa_pairs_np.astype(float)
    nr_aa_pairs = pd.DataFrame(nr_aa_pairs_np, columns=aa20, index=aa20)

    # divide the sum of aa distance matrix by the number of each pairs matrix
    average_aa_mat_np = np.nan_to_num(np.divide(sum_dist_aa_mat_np, nr_aa_pairs_np, out=np.zeros_like(sum_dist_aa_mat_np), where=nr_aa_pairs_np!=0)).round(2)
    average_aa_mat_np = np.triu(average_aa_mat_np) + np.triu(average_aa_mat_np, 1).transpose()
    average_aa_mat = pd.DataFrame(average_aa_mat_np, columns=aa20, index=aa20)

    return average_aa_mat

#Check if folder exists, if not create folder
def check_folder_exists(folder_name):
  if not os.path.isdir(folder_name):
    os.makedirs(folder_name)

if __name__ == "__main__":
  
    # read distance maps in folder
    dist_maps_dir = 'TestDistMaps/'
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
        avg_aa_map = DistMap2AvgDistMap(seq_info, d_map_dir)
        
        # save the average_aa_mat to *.txt file
        np.savetxt(average_aa_distance_folder_name + entry_name + '_AVG_CA_dist_map', avg_aa_map, fmt='%1.2f')


