# BioinfoProj_MembraneBioinformatics

## Progress Record

### ~~2022-Aug-3 Meeting 1 TO DO LIST~~
- [x] Download PDB files according to the helix-sheet dataset 

### ~~2022-Aug-24 Meeting 2 TO DO LIST~~
- [x] Generate CA-CA distance maps according to the PDB files 
- [x] Generate CA-CA contact maps 

### 2022-Sep-01 Meeting 3 TO DO LIST
- [x] Dataset cleansing for viral sequences 
- [ ] Still 282 nonAF proteins - generate the PDB files by prediction tools if low accuracy later

### 2022-Sep-12 Meeting 4 TO DO LIST 
- [x] Generate the averaged 20-amino acid distance matrix
- [x] Embed the averaged 20-amino acid distance matrix into SVM
- [x] Generate the accumulated contact aa matrix.
- [x] Embed the accumulated contact aa matrix into SVM
- [ ] Other feature engineering from literature reviews - DL-based feature pre-processing etc.

### 2022-Oct-5 Meeting 5 TO DO LIST
- [x] Update Cmap generation codes according to Andreas' script
- [x] Generate all the required datasets on the server - cmap thresholds 10, 15, 20, 25, 30
- [x] Prediction results of SVM - prediction results, best-performance parameters, trained models 
- [ ] CNN trial - VGG19
  - [ ] seperate cmaps into train and test folders, alpha and beta folders
  - [ ] VGG19 model generation
  - [ ] cross validation for model performance
