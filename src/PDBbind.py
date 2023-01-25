import os
import pickle
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

from rdkit import Chem
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 
             'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 
             'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 
             'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'Sr', 'unknown']
DATASET_PARAMS = {
    "pos_rate": 0.01,
}

def batch_data_process_PoseFGS(data):
    # re-organize
    data = list(zip(*data))
    pocketGraph, ligandGraph, complexGraph, pdbid, name, atom_name = [], [], [], [], [], []
    for a, b, c, d, e in zip(*data):
        if a is None:
            continue
        pocketGraph.append(a)
        ligandGraph.append(b)
        complexGraph.append(c)
        pdbid.append(d)
        name.append(e)
        atom_name.append([e+"_"+str(i) for i in range(b.num_nodes)])


    # from list to batch
    pocketGraphBatch = Batch.from_data_list(pocketGraph)
    ligandGraphBatch = Batch.from_data_list(ligandGraph)
    complexGraphBatch = Batch.from_data_list(complexGraph)

    # labels
    atom_label = ligandGraphBatch.y[ligandGraphBatch.ymask]
    atom_label = atom_label.reshape(-1, 1)

    pocket_label = complexGraphBatch.y[complexGraphBatch.ymask]
    pocket_label = pocket_label.reshape(-1, 1)

 
    return (pocketGraphBatch, ligandGraphBatch, complexGraphBatch), \
        {"Pocket": pocket_label, "Atom": atom_label, "PDBID": pdbid, "Name": name, "AtomName": atom_name}
    

class DataSet(object):
    """
    Dataset for poseFGS
    """
    def __init__(self, path, kwargs):
        self.path = path
        self.kwargs = kwargs
        self.pdbbind_path = ''
        self.ligand_path = ''

        self.all_data = []
        self.dict_file = {}
        self.dict_conformer = {}
        self.list_pdbid = []
        for item in os.listdir(self.path + self.kwargs['dataset']):
            if not item.endswith('.sdf'):
                continue
            pdbid = item[:4]
            self.list_pdbid.append(pdbid)
            self.dict_conformer[pdbid] = []
            self.dict_file[pdbid] = item
            mol_file = self.path + self.kwargs['dataset'] + '/' + item
            mols = Chem.SDMolSupplier(mol_file, sanitize=False)
            for i, _ in enumerate(mols):
                self.dict_conformer[pdbid].append(i)
                self.all_data.append([pdbid, i])
        np.random.shuffle(self.all_data)
        '''
        dict of pocket elements and coordinates:    
            pocket_elems, pocket_coords = dict_pocket[pdbid]
        ''' 
        # self.dict_pocket = pickle.load(open(self.path + self.kwargs['dataset'] + '/dict_pocket.pk', 'rb'))
        self.dict_pocket = {}
        for pdbid in self.list_pdbid:
            list_elems = []
            list_coords = []
            pocket_file = self.path + 'pdbbind/{}/{}_pocket.pdb'.format(pdbid, pdbid)
            with open(pocket_file, 'r') as f:
                line = f.readline()
                while line:
                    if line.startswith('ATOM'):
                        line = line.strip()
                        coord = line[32:56].strip().split()
                        elem = line[75:].strip()
                        if elem != 'H':
                            list_elems.append(elem)
                            list_coords.append(coord)
                    line = f.readline()
            list_elems = np.array(list_elems)
            list_coords = np.array(list_coords, dtype=float)
            self.dict_pocket[pdbid] = [list_elems, list_coords]
        '''
        dict of label, global rmsd and atom deviations (rmsd_raw):
            rmsd, rmsd_raw = self.dict_rmsd[pdbid]
        '''
        self.dict_rmsd = {}
        if os.path.exists(self.path + self.kwargs['dataset'] + '/dict_rmsd.pk'):
            self.dict_rmsd = pickle.load(open(self.path + self.kwargs['dataset'] + '/dict_rmsd.pk', 'rb'))

        # for train only
        self.index = np.arange(len(self.list_pdbid))
        self.train = False

        self.prepare_constant()

    def __getitem__(self, idx):
        """
        load data for a single instance
        """
        # get pdbid 
        if self.train:
            pdbid = None
            conf = None
        else:
            pdbid, conf = self.all_data[idx]
        # get graph of pocket
        pocketGraph, ligandGraph, complexGraph, name = self.get_pocket_graph(pdbid, conf)

        return pocketGraph, ligandGraph, complexGraph, pdbid, name
                
    def __len__(self):
        if self.train:
            return len(self.index)
        else:
            return len(self.all_data)
    
    def reset_index(self, index, train=False):
        self.index = index
        self.train = train
        self.all_data = []
        for pdbid in self.list_pdbid[self.index]:
            if self.train:
                conf = np.random.choice(self.dict_conformer[pdbid])
                self.all_data.append([pdbid, conf])
            else:
                for conf in self.dict_conformer[pdbid]:
                    self.all_data.append([pdbid, conf])
        np.random.shuffle(self.all_data)

    def prepare_constant(self):
        # Feature lists
        self.list_element = np.array(ELEM_LIST)
        # Constant parameters
        self.atom_feature_dim = len(self.list_element)
        self.bond_feature_dim = 4
        
    def get_mol(self, pdbid, conf=None):
        # for train only
        if self.train and (np.random.random() < self.pos_rate):
            # path to pdbbind ligand *.sdf file
            ligand_file = self.path + 'pdbbind/{}/{}_ligand.sdf'.format(pdbid, pdbid)
            mol = next(Chem.SDMolSupplier(ligand_file, sanitize=False))
            if mol is not None:
                list_elems = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
                list_coords = np.array(mol.GetConformer().GetPositions(), dtype=float)
                rmsd, rmsd_raw = 0.0, np.zeros(len(list_elems))
                name = pdbid + '_ligand'
                return list_elems, list_coords, rmsd, rmsd_raw, name
        if conf is None:
            conf = np.random.choice(self.dict_conformer[pdbid])
        # native pose
        if conf == -1:
            ligand_file = self.path + 'pdbbind/{}/{}_ligand.sdf'.format(pdbid, pdbid)
            mol = next(Chem.SDMolSupplier(ligand_file, sanitize=False))
            list_elems = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
            list_coords = np.array(mol.GetConformer().GetPositions(), dtype=float)
        # decoy pose
        else:
            mols_file = self.path + self.kwargs['dataset'] + '/' + self.dict_file[pdbid]
            mols = Chem.SDMolSupplier(mols_file, sanitize=False)
            for i, mol in enumerate(mols):
                if i == conf:
                    break
            list_elems = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
            list_coords = np.array(mol.GetConformer().GetPositions(), dtype=float)
        if pdbid in self.dict_rmsd:
            rmsd, rmsd_raw = self.dict_rmsd[pdbid]
        else:
            rmsd, rmsd_raw = 0.0, np.zeros(len(list_elems))
        name = mol.GetProp('_Name')

        return list_elems, list_coords, rmsd, rmsd_raw, name

    def get_pocket_graph(self, pdbid, conf=None):
        pocket_elems, pocket_coords = self.dict_pocket[pdbid]
        if not self.train:
            ligand_elems, ligand_coords, rmsd, rmsd_raw, name = self.get_mol(pdbid, conf)
            _, mindist = pairwise_distances_argmin_min(pocket_coords, ligand_coords)
            index = mindist < 6
        else:
            for _ in range(10):
                ligand_elems, ligand_coords, rmsd, rmsd_raw, name = self.get_mol(pdbid, conf)
                _, mindist = pairwise_distances_argmin_min(pocket_coords, ligand_coords)
                index = mindist < 6
                if sum(index) > 0:
                    break

        pocket_coords = pocket_coords[index]
        pocket_elems = pocket_elems[index]

        # labels
        atom_mask = torch.zeros(len(ligand_elems), 1).bool()
        atom_label = torch.zeros(len(ligand_elems), 1)
        for idx, item in enumerate(rmsd_raw):
            if np.isnan(item):
                continue
            if item < 2:
                label = 1
            else:
                label = 0
            atom_mask[idx] = True
            atom_label[idx] = label
        graph_mask = torch.zeros(1, 1).bool()
        graph_label = torch.zeros(1, 1)
        if (not np.isnan(rmsd)) and (not np.isinf(rmsd)):
            if rmsd < 2:
                label = 1
            else:
                label = 0
            graph_mask[0] = True
            graph_label[0] = label

        # remove Hs
        index = ligand_elems != 'H'
        ligand_elems = ligand_elems[index]
        ligand_coords = ligand_coords[index]
        atom_mask = atom_mask[index]
        atom_label = atom_label[index]

        # atom features
        get_one_atom_feature = lambda atom: self.onehot(atom, self.list_element)
        pocket_features = list(map(get_one_atom_feature, pocket_elems))
        ligand_features = list(map(get_one_atom_feature, ligand_elems))
        pocket_features = torch.FloatTensor(pocket_features)
        ligand_features = torch.FloatTensor(ligand_features)

        # edges
        if len(pocket_coords) == 0:
            pocket_edges = torch.zeros(2, 0).long()
            pocket_weight = torch.zeros(0, 4)
        else:
            pocket_distance = pairwise_distances(pocket_coords, pocket_coords)
            pocket_edges = np.array(np.where((pocket_distance > 0) & (pocket_distance < 4)))
            temp = pocket_distance[pocket_edges[0], pocket_edges[1]]
            pocket_weight = np.vstack([np.exp(-temp), np.exp(1-temp), np.exp(2-temp), np.exp(3-temp)]).T
        
        ligand_distance = pairwise_distances(ligand_coords, ligand_coords)
        ligand_edges = np.array(np.where((ligand_distance > 0) & (ligand_distance < 4)))
        temp = ligand_distance[ligand_edges[0], ligand_edges[1]]
        ligand_weight = np.vstack([np.exp(-temp), np.exp(1-temp), np.exp(2-temp), np.exp(3-temp)]).T
        
        pocket_edges = torch.LongTensor(pocket_edges)
        ligand_edges = torch.LongTensor(ligand_edges)
        pocket_weight = torch.FloatTensor(pocket_weight)
        ligand_weight = torch.FloatTensor(ligand_weight)

        complex_mask = [0] * len(pocket_elems) + [1] * len(ligand_elems)
        complex_mask = torch.BoolTensor(complex_mask)
        
        if len(pocket_coords) == 0:
            complex_edges = torch.zeros(2, 0).long()
            complex_weight = torch.zeros(0, 4)
        else:
            complex_distance = pairwise_distances(pocket_coords, ligand_coords)
            complex_edges = np.array(np.where((complex_distance > 0) & (complex_distance < 6)))
            temp = complex_distance[complex_edges[0], complex_edges[1]]
            complex_weight = np.vstack([np.exp(-temp), np.exp(1-temp), np.exp(2-temp), np.exp(3-temp)]).T
            complex_edges[1] += len(pocket_elems)

        complex_edges = torch.LongTensor(complex_edges)
        complex_weight = torch.FloatTensor(complex_weight)

        complex_edges = torch.cat([complex_edges, torch.cat([complex_edges[1:2], complex_edges[0:1]], 0)], 1)
        complex_weight = torch.cat([complex_weight, complex_weight], 0)

        pocket_coords = torch.FloatTensor(pocket_coords)
        ligand_coords = torch.FloatTensor(ligand_coords)

        pocketGraph = Data(
            x=pocket_features, 
            pos=pocket_coords,
            edge_index=pocket_edges, 
            edge_attr=pocket_weight,
        )

        ligandGraph = Data(
            x=ligand_features, 
            pos=ligand_coords,
            edge_index=ligand_edges, 
            edge_attr=ligand_weight,
            ymask=atom_mask,
            y=atom_label,
        )

        complexGraph = Data(
            pos=torch.cat([pocket_coords, ligand_coords], 0),
            edge_index=complex_edges,
            edge_attr=complex_weight,
            mask=complex_mask,
            ymask=graph_mask,
            y=graph_label,
        )
        complexGraph.num_nodes = len(pocket_elems) + len(ligand_elems)

        return pocketGraph, ligandGraph, complexGraph, name
        
    def onehot(self, code, list_code):
        # if code not in list_code and "unknown" in list_code:
        #     return list(list_code=="unknown")

        # assert code in list_code, "{} not in list {}".format(code, list_code)
        return list(list_code==code)