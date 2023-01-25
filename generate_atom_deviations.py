import os
import sys
import pickle
import numpy as np
from rdkit import Chem
from sklearn.metrics import pairwise_distances

def RMSD(pdbid, docking_file):
    ###########################
    # specify the correct file names
    raw_file = './data/pdbbind/{}/{}_ligand.sdf'.format(pdbid, pdbid)
    ###########################
    assert os.path.exists(raw_file), 'please check the file name'
    assert os.path.exists(docking_file), 'please check the file name'
    
    ligand_mol = next(Chem.SDMolSupplier(raw_file, sanitize=False))
    ligand_coords = ligand_mol.GetConformer().GetPositions()[:,:,None]
    ligand_elems = np.array([atom.GetSymbol() for atom in ligand_mol.GetAtoms()])

    docking_mols = Chem.SDMolSupplier(docking_file, sanitize=False)
    docking_coords = []
    docking_names = []
    for docking_mol in docking_mols:
        docking_coords.append(np.array(docking_mol.GetConformer().GetPositions())[:,:,None])
        docking_elems = np.array([atom.GetSymbol() for atom in docking_mol.GetAtoms()])
        docking_names.append(docking_mol.GetProp('_Name'))
    docking_coords = np.concatenate(docking_coords, 2)
    
    m1, _, n1 = ligand_coords.shape
    m2, _, n2 = docking_coords.shape
    
    # find atoms with the same elements
    mask = np.zeros((m1, m2), dtype=bool)
    for i in range(m1):
        if ligand_elems[i] == 'H':
            continue
        for j in range(m2):
            if docking_elems[j] == 'H':
                continue
            if ligand_elems[i] == docking_elems[j]:
                mask[i, j] = 1
                
    # find rmsd
    results = np.zeros((n1, n2))
    results_raw = np.zeros((n1, n2, m2))
    for i in range(n1):
        for j in range(n2):
            dist = pairwise_distances(ligand_coords[:,:,i], docking_coords[:,:,j])
            dist[mask==False] = np.inf
            d1 = np.nanmin(dist, 0)
            valid_d1 = d1[np.isinf(d1) == False]
            rmsd1 = np.sqrt(np.mean(valid_d1 ** 2))
            results[i, j] = rmsd1
            results_raw[i, j] = d1
    return results, results_raw#, docking_names

folder = sys.argv[1]
dict_rmsd = {}
for item in os.listdir(folder):
    pdbid = item[:4]
    docking_file = '{}/{}'.format(folder, item)
    dict_rmsd[pdbid] = RMSD(pdbid, docking_file)
    
pickle.dump(dict_rmsd, folder + '/dict_rmsd.pk')
