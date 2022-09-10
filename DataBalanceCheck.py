#!/usr/bin/env python 

import pandas as pd

hs_df = pd.read_csv('helix_sheet.tsv', sep='\t', header=0)
nonAFprot = pd.read_csv('Not_exist_AF_prot.txt', decimal="\t", header=None).iloc[:,0]

# generate the helix-sheet dataset excluding all 671 nonAF proteins
AF_hs_df = hs_df[~hs_df['Entry'].isin(nonAFprot)]
print(AF_hs_df['label'].value_counts())
AF_hs_df.to_csv('AF_helix_sheet.tsv', sep="\t", index=False)

# Alpha helix    30659
# Beta strand      638
# Name: label, dtype: int64

# generate the helix-sheet dataset excluding only 389 viral proteins 
virus_df = pd.read_csv('virus_helix_sheet.tsv', sep='\t', header=0)
nonVirus_hs_df = hs_df[~hs_df['Entry'].isin(virus_df['Entry'])]
nonVirus_hs_df.to_csv('nonVirus_helix_sheet.tsv', sep="\t", index=False)
