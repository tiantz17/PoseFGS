import time
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch_scatter import scatter_mean
from torch_geometric.nn import GINEConv

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


MODEL_PARAMS = {
    'input_dim': 64,
    'edge_dim': 4,
    'hidden_dim': 256, 
    'pocket_depth': 4, 
    'ligand_depth': 4, 
    'complex_depth': 4, 
}


TRAIN_PARAMS = {
    'num_repeat': 1,
    'num_fold': 5,
    'batch_size': 32,
    'max_epoch': 100,
    'early_stop': 20,
    'learning_rate': 1e-3,
    'weight_decay': 1e-6,
    'step_size': 20,
    'gamma': 0.5,
    'list_task': ['Pocket', 'Atom'],
    'loss_weight': {'Pocket': 0.0, 'Atom': 1.0},
    'task_eval': {'Pocket': 'cls', 'Atom': 'cls'},
    'task': 'Atom',
    'goal': 'auroc',
}


class Model(nn.Module):
    """
    PocketPoint for pocket scoring
    """
    def __init__(self, params):
        super(Model, self).__init__()  
        """hyper part"""
        self.input_dim = int(params['input_dim'])
        self.edge_dim = int(params['edge_dim'])
        self.hidden_dim = int(params['hidden_dim'])
        self.pocket_depth = int(params['pocket_depth'])
        self.ligand_depth = int(params['ligand_depth'])
        self.complex_depth  = int(params['complex_depth'])
        
        """model part"""
        self.PocketGNN = nn.ModuleList()
        self.PocketBN = nn.ModuleList()
        input_dim = self.input_dim
        for _ in range(self.pocket_depth):
            pocketNN = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim), 
                nn.LeakyReLU(0.1), 
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.PocketGNN.append(GINEConv(pocketNN, edge_dim=self.edge_dim))
            self.PocketBN.append(nn.LayerNorm(self.hidden_dim))
            input_dim = self.hidden_dim
            
        self.LigandGNN = nn.ModuleList()
        self.LigandBN = nn.ModuleList()
        input_dim = self.input_dim
        for _ in range(self.ligand_depth):
            ligandNN = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim), 
                nn.LeakyReLU(0.1), 
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.LigandGNN.append(GINEConv(ligandNN, edge_dim=self.edge_dim))
            self.LigandBN.append(nn.LayerNorm(self.hidden_dim))
            input_dim = self.hidden_dim
            
        self.ComplexGNN = nn.ModuleList()
        self.ComplexBN = nn.ModuleList()
        for _ in range(self.complex_depth):
            complexNN = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim), 
                nn.LeakyReLU(0.1), 
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.ComplexGNN.append(GINEConv(complexNN, edge_dim=self.edge_dim))
            self.ComplexBN.append(nn.LayerNorm(self.hidden_dim))

        self.leakyrelu = nn.LeakyReLU(0.1)

        # Classifier
        self.W_pocket = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_dim, 1),
        )

        self.W_atom = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_dim, 1),
        )

        self.apply(weights_init)

    def load_optimizer(self, train_params):
        self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, self.parameters())), 
                                    lr=train_params["learning_rate"], 
                                    weight_decay=train_params["weight_decay"])

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                   step_size=train_params["step_size"], 
                                                   gamma=train_params["gamma"])

        self.loss = {
            "Pocket": nn.BCEWithLogitsLoss(), 
            "Atom": nn.BCEWithLogitsLoss(),
            }
        self.loss_weight = train_params['loss_weight']
        self.task_eval = train_params['task_eval']

    def get_loss(self, dict_pred, dict_label):
        loss = 0.0
        for task in dict_pred:
            loss = loss + self.loss[task](dict_pred[task], dict_label[task]) * self.loss_weight[task]
        return loss

    def Pocket_pred_module(self, anch_feature):
        pocket_pred = self.W_pocket(anch_feature)
        return pocket_pred

    def Atom_pred_module(self, atom_feature):
        atom_pred = self.W_atom(atom_feature)
        return atom_pred

    def forward(self, pocketGraphBatch, ligandGraphBatch, complexGraphBatch):
        calcTime = False
        list_time = []
        device = pocketGraphBatch.x.device

        # Step 1. Pocket GNN
        """
        Graph message passing on proteins
        """
        # if calcTime: list_time.append(time.time())
        pocket_feature0 = pocketGraphBatch.x
        pocket_edge = pocketGraphBatch.edge_index
        pocket_weight = pocketGraphBatch.edge_attr
        for depth in range(self.pocket_depth):
            pocket_feature = self.PocketGNN[depth](pocket_feature0, pocket_edge, pocket_weight)
            pocket_feature = self.PocketBN[depth](pocket_feature)
            if depth > 0:
                pocket_feature = self.leakyrelu(pocket_feature + pocket_feature0)
            else:
                pocket_feature = self.leakyrelu(pocket_feature)
            pocket_feature0 = pocket_feature

        if calcTime: list_time.append(time.time())

        # Step 2. Ligand GNN
        """
        Graph message passing on ligands
        """
        # if calcTime: list_time.append(time.time())
        ligand_feature0 = ligandGraphBatch.x
        ligand_edge = ligandGraphBatch.edge_index
        ligand_weight = ligandGraphBatch.edge_attr
        for depth in range(self.ligand_depth):
            ligand_feature = self.LigandGNN[depth](ligand_feature0, ligand_edge, ligand_weight)
            ligand_feature = self.LigandBN[depth](ligand_feature)
            if depth > 0:
                ligand_feature = self.leakyrelu(ligand_feature + ligand_feature0)
            else:
                ligand_feature = self.leakyrelu(ligand_feature)
            ligand_feature0 = ligand_feature

        if calcTime: list_time.append(time.time())

        # Step 3. Complex GNN
        """
        Graph message passing on complex
        """
        # if calcTime: list_time.append(time.time())
        complex_feature0 = torch.zeros(complexGraphBatch.num_nodes, pocket_feature.shape[1], device=device)
        complex_feature0[complexGraphBatch.mask==False] = pocket_feature
        complex_feature0[complexGraphBatch.mask] = ligand_feature

        complex_edge = complexGraphBatch.edge_index
        complex_weight = complexGraphBatch.edge_attr
        for depth in range(self.complex_depth):
            complex_feature = self.ComplexGNN[depth](complex_feature0, complex_edge, complex_weight)
            complex_feature = self.ComplexBN[depth](complex_feature)
            complex_feature = self.leakyrelu(complex_feature + complex_feature0)
            complex_feature0 = complex_feature

        if calcTime: list_time.append(time.time())

        
        # Step 4. Prediction
        """
        Predict atom deviations
        """
        if calcTime: list_time.append(time.time())
        ligand_feature = complex_feature[complexGraphBatch.mask]
        atom_pred = self.Atom_pred_module(ligand_feature)
        atom_pred = atom_pred[ligandGraphBatch.ymask].reshape(-1, 1)

        graph_feature = scatter_mean(ligand_feature, ligandGraphBatch.batch, 0)
        pocket_pred = self.Pocket_pred_module(graph_feature)
        pocket_pred = pocket_pred[complexGraphBatch.ymask].reshape(-1, 1)
        
        # Step 5. Output
        if calcTime: list_time.append(time.time())
        if calcTime: 
            list_time = np.array(list_time)
            time_elapsed = list_time[1:]-list_time[:-1]
            print(("{:.4f}\t"*(len(list_time)-1)).format(*time_elapsed))

        return {"Pocket": pocket_pred, "Atom": atom_pred}
