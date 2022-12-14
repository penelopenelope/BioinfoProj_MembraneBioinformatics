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
- [x] CNN trial - InceptionV3
  - [x] seperate cmaps into train and test folders, alpha and beta folders
  - [ ] Inception V3 ~~model generation and~~ prediction
  - [ ] Simple CNN ~~model generation and~~ prediction
    - [x] only 2* FC layers
    - [x] 2* Conv layers + 2* FC layers 

### 2022-Nov-02 Meeting 6 TO DO LIST
- [x]  Group seminar slides 
  - [x]  done
  - [x]  plan
    - [ ]  GNN
    - [ ]  Representation Learning - feature can be extracted by GNN
    - [ ]  text classification and no-pretrained model
- [ ]  Compare SVM and CNN model performances 
- [x]  lower cmap threshold for SVM - 5
- [x]  other features by Representation Learning
- [ ]  GPU test
- [ ]  new dataset for transporter substrate
- [ ]  dataset imbalance!!!
- [ ]  multimodal encoder-decoder
