#!/usr/bin/env python

from email import header
import pandas as pd
import numpy as np

hs_df = pd.read_csv('helix_sheet.tsv', sep='\t', header=0)
nonAFprot = pd.read_csv('Not_exist_AF_prot.txt', decimal="\t", header=None).iloc[:,0]

# Check for non-AF virus proteins 
virus_prot = hs_df[hs_df['Organism'].str.contains('virus')]
# print(virus_prot)
virus_prot.to_csv('virus_helix_sheet.tsv', sep="\t", index=False)
virus_entry = virus_prot.iloc[:,0]
# print(nonAFprot)
print("{first} of {second} virus proteins in helix-sheet dataset are not included in AlphaFold database.". \
    format(first=nonAFprot.isin(virus_entry).sum(), second=virus_entry.size)) # 389 viral prots

# Check for non-AF prots with non-standard amino acids
rest_nonAFprots = nonAFprot[~nonAFprot.isin(virus_entry)].values # 282 rest prots
print(rest_nonAFprots)
rest_nonAFprots_df = hs_df[hs_df['Entry'].isin(rest_nonAFprots)]
print(rest_nonAFprots_df)
non_std_aa = ['X', 'U', 'O']
non_std_aa_prot = rest_nonAFprots_df[rest_nonAFprots_df['Sequence'].isin(non_std_aa)]
print(non_std_aa_prot) # 0

# Check for non-AF prots by length
long_prot = rest_nonAFprots_df[rest_nonAFprots_df['Sequence'].map(len) > 1280]
print(long_prot) # 45
short_prot = rest_nonAFprots_df[rest_nonAFprots_df['Sequence'].map(len) < 16]
print(short_prot) # 0
long_prot_hs = hs_df[hs_df['Sequence'].map(len) > 1280]
print(long_prot_hs) # 802
short_prot_hs = hs_df[hs_df['Sequence'].map(len) < 16]
print(short_prot_hs) # 0

# P15484 - several accessions - none exist in AlphaFold database - search by proteomic name - https://alphafold.ebi.ac.uk/search/text/CS3%20pili%20synthesis%20104%20kDa%20protein?organismScientificName=Escherichia%20coli
# P44523 - found some by gene names - how to identify which one is the desired? - https://alphafold.ebi.ac.uk/search/text/tdhA%20HI_0113?organismScientificName=Haemophilus%20influenzae
# K7ZP88 - not really the same but will the similar entry

# POSSIBLE SOLUTION - https://alphafold.ebi.ac.uk/faq
    # Prediction via AlphaFold source code
    # using the EBI Protein Similarity Search tool 