#!/usr/bin/env python

import numpy as np
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

import urllib.request
import os
import pandas as pd
import requests

def load_predicted_PDB_alphaC(pdbfile):
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues()]
    #print(residues[108])    # ['CB'].get_coord()) - <Residue GLY het=  resseq=109 icode= >

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one-two)

    return distances, seqs[0]

def load_predicted_PDB_betaC(pdbfile):
    # Generate (diagonalized) C_beta distance matrix from a pdbfile
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CB"].get_coord()
            two = residues[y]["CB"].get_coord()
            distances[x, y] = np.linalg.norm(one-two)

    return distances, seqs[0]

#Check if folder exists, if not create folder
def check_folder_exists(folder_name):
  if not os.path.isdir(folder_name):
    os.makedirs(folder_name)
  

if __name__ == "__main__":

    pdb_files_dir = 'pdbfiles/'
    directory = os.fsencode(pdb_files_dir)
    
    cmap_thresh = 10

    for pdbfile_dir in os.listdir(directory):
        filename = os.fsdecode(pdbfile_dir)
        #print(filename)
        #print(os.path.abspath(pdb_files_dir + filename))

        # Generate (diagonalized) C_alpha distance and contact matrix from a pdbfile
        dist_mat_alphaC, seq = load_predicted_PDB_alphaC(os.path.abspath(pdb_files_dir + filename))
        contact_map_alphaC = np.double(dist_mat_alphaC < cmap_thresh)
        #print(dist_mat_alphaC)
        #print(contact_map_alphaC)

        # Generate (diagonalized) C_beta distance and contact matrix from a pdbfile
        #dist_mat_betaC, _ = load_predicted_PDB_betaC(pdbfile_dir) # what to do with CB for Glycine?
        #contact_map_betaC = np.double(dist_mat_betaC < cmap_thresh)
        #print(dist_mat_betaC) # glycine??
        #print(contact_mat_betaC)

        #print(seq)

        # save the distance matrix to *.npz files 
        folder_name = 'CA_cmaps/'
        check_folder_exists(folder_name)
        np.save(folder_name + filename[:-4:] + '_CA_cmap', contact_map_alphaC)

        # # check the saved file
        # data = np.load(folder_name + 'AF_CA_cmap.npy', mmap_mode='r')
        # print(len(seq))
        # print((data == contact_map_alphaC).sum())

        # save figs? -- https://birdlet.github.io/2017/08/08/trj_contact_map/ 

    # # """
    # # OR default pdf file from AlphaFold 
    # # """
    # pdbfile_dir = '/home/pen/Desktop/HiWi_Helms/0-Workflow_Github/pdbfiles/B6I5Q0.pdb'

    # # Generate (diagonalized) C_alpha distance and contact matrix from a pdbfile
    # dist_mat_alphaC, seq = load_predicted_PDB_alphaC(pdbfile_dir)
    # contact_map_alphaC = np.double(dist_mat_alphaC < cmap_thresh)
    # #print(dist_mat_alphaC)
    # #print(contact_map_alphaC)

    # # Generate (diagonalized) C_beta distance and contact matrix from a pdbfile
    # #dist_mat_betaC, _ = load_predicted_PDB_betaC(pdbfile_dir) # what to do with CB for Glycine?
    # #contact_map_betaC = np.double(dist_mat_betaC < cmap_thresh)
    # #print(dist_mat_betaC) # glycine??
    # #print(contact_mat_betaC)

    # #print(seq)

    # # save the distance matrix to *.npz files 
    # folder_name = 'CA_cmaps/'
    # check_folder_exists(folder_name)
    # np.save(folder_name + 'B6I5Q0_CA_cmap', contact_map_alphaC)

    # # # check the saved file
    # # data = np.load(folder_name + 'AF_CA_cmap.npy', mmap_mode='r')
    # # print(len(seq))
    # # print((data == contact_map_alphaC).sum())

    # # save figs? -- https://birdlet.github.io/2017/08/08/trj_contact_map/ 

