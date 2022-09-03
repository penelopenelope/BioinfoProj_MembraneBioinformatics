#!/usr/bin/env python

import numpy as np
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

import urllib
import urllib.request
import os
import pandas as pd
import requests

#Check if folder exists, if not create folder
def check_folder_exists(folder_name):
  if not os.path.isdir(folder_name):
    os.makedirs(folder_name)
  

if __name__ == "__main__":

    # generate the protein entry list from the helix-sheet dataset 
    hs_df = pd.read_csv('helix_sheet.tsv', sep='\t', header=0)
    prot_entry = hs_df.iloc[:,0]
    print(prot_entry)

    # download structure files (*.pdb) from Alphafold for the helix-sheet dataset\
    url_AF = 'https://alphafold.ebi.ac.uk/files/'
    folder_name = 'pdbfiles/'
    check_folder_exists(folder_name)
    not_exist_prot = open('Not_exist_AF_prot.txt', 'w+')
    for prot in prot_entry:
        print(prot)
        AFname = 'AF-' + prot + '-F1-model_v3.pdb'
        response = requests.get(url_AF + AFname)
        if response.status_code == 200:
            urllib.request.urlretrieve(url_AF + AFname, folder_name + prot + '.pdb')
        else:
            print('Protein does not exist in AlphaFold database.') 
            not_exist_prot.write(prot + '\n')
    not_exist_prot.close()
        