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
    - [x]  GNN
    - [x]  Representation Learning - feature can be extracted by GNN
    - [x]  text classification and no-pretrained model
- [x]  Compare SVM and CNN model performances 
- [x]  lower cmap threshold for SVM - 5
- [x]  other features by Representation Learning
- [ ]  GPU test
- [ ]  new dataset for transporter substrate
- [ ]  dataset imbalance!!!
- [ ]  multimodal encoder-decoder


### 2023-Mar-17 Meeting 7 TO DO LIST
- [ ] New dataset - Transmembrane transporters VS other membrane proteins
  - [ ] positive - 
    - [ ] Transmembrane AND Transport AND transmembrane transport (GO), NOT virus - 11215126 results
    - [ ] Transmembrane (kw) AND Transport (kw) AND transmembrane transport (GO), NOT virus - 8281273 results
  - [ ] negative - 
    - [ ] membrane, NOT transport, NOT transmembrane transport (GO), NOT virus - 35404263 results
    - [ ] membrane, NOT transport, NOT transmembrane transport (GO), AND signal, NOT virus - 5939064 results
    - [ ] membrane (kw), NOT transport (kw), NOT transmembrane transport (GO), NOT virus - 32833742 results
    - [ ] membrane (kw), NOT transport (kw), NOT transmembrane transport (GO), AND signal (kw), NOT virus - 204347 results
    - [ ] Questions - intracellular protein transport? (Q07418)
- [ ] Old dataset - Alpha-helix VS Beta-sheet
  - [ ] oversampling on minor class - imbalanced-learn package
    - [ ] GNN
    - [ ] AutoEncoder
- [ ] Representation Learning on new dataset
  - [ ] GNN
  - [ ] AutoEncoder
  - [ ] Causal Representation Learning