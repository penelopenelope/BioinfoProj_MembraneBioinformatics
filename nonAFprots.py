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


# 
rest_nonAFprots = nonAFprot[~nonAFprot.isin(virus_entry)].values # 282 rest prots
# print(rest_nonAFprots)