# PoseFGS
Source code for paper "PoseFGS: Improving and Interpreting Ligand Pose Prediction with Fine-Grained Supervision"

<div><img width=300 src=https://github.com/tiantz17/PoseFGS/blob/main/fig/PoseFGS-novelty.png></div>

# Requirements

    pytorch=1.11.0
    pyg=2.0.4



# How to score ligand poses using PoseFGS?

## 1. Dataset preparation
PoseFGS takes protein atom elements, protein atom coordinates, ligand atom elements, and ligand atom coordinates as inputs.

An example dataset of CASF-2016 docking power is provided and can be obtained by

    unzip data.zip

### 1.1 Protein (pockets)
The protein atom elements and coordinates are obtained from the [PDBbind](http://www.pdbbind.org.cn/) database.
Please download and unzip the files of the PDBbind database into the following folder

    ./data/pdbbind/

and orgainze as

    ./data/pdbbind/[PDBID]

The ```./data/pdbbind/[PDBID]/[PDBID]_pocket.pdb``` file will be used for extracting the protein atom elements and coordinates.

### 1.2 Ligand (poses)
Prepare ligand files (e.g., from docking softwares) and convert them to *.sdf format.
Typically, the ligand poses generated by docking softwares are in *.pdbqt format.
If needed, the format conversion can be done using script (openbabel 3.1.0 required):

    python convert_pdbqt_to_sdf.py ./data/[DATASET]

### 1.3 Label (for training)
PoseFGS is trained by predicting the atom deviations of the input ligand pose to those of the native pose.
The atom deviations can be obtained using the following script:

    python generate_atom_deviations.py ./data/[DATASET]

The above script will create a pickle object ```dict_rmsd.pk``` at ```./data/[DATASET]/```.


## 2. Make predictions
The PoseFGS models trained on the general set of PDBbind-2016 are provided in ```./trained_models/``` with five repeats.
Run the following script to generate scores for individual ligand poses:

    python runPrediction.py --dataset [DATASET] --gpu [GPU]

If use cpu, please specify ```--gpu -1```.

For example:

    python runPrediction.py --dataset casf2016 --gpu 0

PoseFGS will create a folder for each prediction at ```./predictions/[FOLDER]```.
For example, the PoseFGS scores generated by repeat 0 will be stored at ```./predictions/[FOLDER]/repeat0/[PDBID]_score.dat```.
PoseFGS also provide atom-level predictions for each ligand pose at ```./predictions/[FOLDER]/repeat0/[PDBID]_fgs.dat```.

<div><img width=300 src=https://github.com/tiantz17/PoseFGS/blob/main/fig/PoseFGS-case-new.png></div>


## 3. Train PoseFGS
You can train PoseFGS using your custom dataset.
After generating dataset, run 

    python runTrain.py --dataset [DATASET] --gpu [GPU]

for training.
