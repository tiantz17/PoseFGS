import os
import sys
import time
import json
import pickle
import socket
import logging
import argparse
from importlib import import_module

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class PoseFGSPrediction(object):
    """
    PocketAnchor prediction using trained model
    """
    def __init__(self, args):
        """ common parameters """
        self.seed = args.seed
        self.info = args.info
        self.gpu = args.gpu
        self.use_cuda = args.gpu != "-1"
        self.path = args.path
        self.num_workers = args.num_workers

        """ special parameters """
        self.dataset = args.dataset
        self.model = args.model
        self.model_path = args.model_path

        """ modules """
        self.DATASET = import_module("src.PDBbind")
        self.MODEL = import_module("src.PoseFGS")

        """ training parameters """      
        self.dataset_params = self.DATASET.DATASET_PARAMS
        self.model_params = self.MODEL.MODEL_PARAMS
        self.train_params = self.MODEL.TRAIN_PARAMS

        if len(args.dataset_params) > 0:
            update_params = {item.split(':')[0]:item.split(':')[1] for item in args.dataset_params.split(',')}
        else:
            update_params = {}
        self.dataset_params.update(update_params)
        self.dataset_params['dataset'] = self.dataset

        if len(args.model_params) > 0:
            update_params = {item.split(':')[0]:item.split(':')[1] for item in args.model_params.split(',')}
        else:
            update_params = {}
        self.model_params.update(update_params)

        if len(args.train_params) > 0:
            update_params = {item.split(':')[0]:item.split(':')[1] for item in args.train_params.split(',')}
            if 'list_task' in update_params:
                update_params['list_task'] = eval(update_params['list_task'])
        else:
            update_params = {}
        self.train_params.update(update_params)

        """ update common parameters"""
        self.list_task = self.train_params["list_task"]

        """ local directory """
        file_folder = "Prediction_dataset_{}_model_{}_info_{}_{}_cuda{}"
        file_folder = file_folder.format(self.dataset, self.model, \
            self.info, socket.gethostname(), self.gpu)
        file_folder += time.strftime("_%Y%m%d_%H%M%S/", time.localtime())
        self.save_path = self.path + "predictions/" + file_folder
        self.prediction_path = self.path + "predictions/" + file_folder
        self.valid_log_file = self.save_path + "validation.log"
        self.test_log_file = self.save_path + "test.log"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.prediction_path):
            os.makedirs(self.prediction_path)
        self.define_logging()
        logging.info("Local folder created: {}".format(self.save_path))

        """ save hyperparameters """
        self.save_hyperparameter(args)

    def define_logging(self):
        # Create a logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %A %H:%M:%S',
            filename=self.save_path + "logging.log",
            filemode='w')
        # Define a Handler and set a format which output to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

    def save_hyperparameter(self, args):
        args.dataset_params = self.dataset_params
        args.model_params = self.model_params
        args.train_params = self.train_params
        json.dump(dict(args._get_kwargs()), open(self.save_path + "config", "w+"), indent=4, sort_keys=True)

    def load_data(self):
        logging.info("Loading data...")
        # load data       
        self.Dataset = self.DATASET.DataSet(self.path + "data/", self.dataset_params)

        self.Dataloader = DataLoader(self.Dataset, 
                                     batch_size=int(self.train_params["batch_size"]), 
                                     shuffle=False, 
                                     collate_fn=eval("self.DATASET.batch_data_process_"+self.model), 
                                     num_workers=self.num_workers, 
                                     drop_last=False, 
                                     pin_memory=self.use_cuda)
        
    def get_data_batch(self, batch_items):
        if self.use_cuda: 
            batch_items = [item.to(next(self.Model.parameters()).device) if item is not None and not isinstance(item, list) else \
                [it.to(next(self.Model.parameters()).device) for it in item] if isinstance(item, list) else \
                None for item in batch_items]

        return batch_items  

    def get_label_batch(self, batch_items):
        if self.use_cuda: 
            for key in batch_items.keys():
                if key in self.list_task:
                    batch_items[key] = batch_items[key].to(next(self.Model.parameters()).device)

        return batch_items

    def load_model(self, model_file):
        logging.info("Loading model...")
        # load model
        if self.use_cuda:
            device = torch.device("cuda:"+self.gpu)
        else:
            device = torch.device("cpu")
        self.Model = self.MODEL.Model(self.model_params)
        self.Model.load_state_dict(torch.load(model_file, map_location=device))
        # self.Model = self.Model.to(device)
        
        # load optimizer
        self.Model.load_optimizer(self.train_params)

    def predict(self):
        logging.info("Start prediction")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        self.load_data()
        list_models = []
        for i in os.listdir(self.model_path):
            if "best_model" not in i:
                continue
            if i[-2:] != "pt":
                continue
            list_models.append(i)
        list_models = sorted(list_models)

        for self.repeat, model_file in enumerate(list_models):
            logging.info("="*60)
            logging.info("Repeat: {}, model: {}".format(self.repeat, model_file))
            self.load_model(self.model_path + model_file)
            self.evaluate()

    def evaluate(self):
        self.Model.eval()
        dict_collect = self.get_results_template()
        with torch.no_grad():
            logging.info("Number of samples: " + str(self.Dataset.__len__()))
            for batch, data in enumerate(self.Dataloader):
                data_tuple, label_dict = data
                data_tuple = self.get_data_batch(data_tuple)
                label_dict = self.get_label_batch(label_dict)
                pred_dict = self.Model(*data_tuple)

                for task in dict_collect:
                    dict_collect[task]["pred"].extend(pred_dict[task].cpu().data.numpy())
                    if task in label_dict:
                        dict_collect[task]["label"].extend(label_dict[task].cpu().data.numpy())
                        if "name" not in dict_collect[task]:
                            dict_collect[task]["name"] = []
                            dict_collect[task]["atomname"] = []
                            dict_collect[task]["pdbid"] = []
                        dict_collect[task]["name"].extend(label_dict['Name'])
                        dict_collect[task]["atomname"].extend(label_dict['AtomName'])
                        dict_collect[task]["pdbid"].extend(label_dict['PDBID'])

                if self.info == "debug":
                    break  
        # save to file
        list_atom_raw = np.array(dict_collect['Atom']['pred'])
        list_name = np.array(dict_collect['Pocket']['name'])
        list_pdbid = np.array([item.split('_')[0] for item in list_name])
        list_atom_name = np.array(dict_collect['Atom']['atomname'], dtype=object)
        list_atom_pred = []
        list_atom_pred_fgs = []
        total = 0
        for i in list_atom_name:
            temp = list_atom_raw[total:total+len(i)]
            list_atom_pred.append(temp.mean())
            list_atom_pred_fgs.append(temp)
            total += len(i)
        list_atom_pred = np.array(list_atom_pred)
        if not os.path.exists(self.prediction_path + 'repeat{}'.format(self.repeat)):
            os.mkdir(self.prediction_path + 'repeat{}'.format(self.repeat))
        for pdbid in np.unique(list_pdbid):
            pred = list_atom_pred[list_pdbid == pdbid].reshape(-1)
            name = list_name[list_pdbid == pdbid]
            pred_fgs = [item for (item, jtem) in zip(list_atom_pred_fgs, list_pdbid) if jtem == pdbid]
            with open(self.prediction_path + 'repeat{}/{}_score.dat'.format(self.repeat, pdbid), 'w') as f:
                f.write("{:13s}{:8s}\n".format('#code', "score"))
                for p, n in zip(pred, name):
                    f.write("{:13s}{:8f}\n".format(n, p))
            with open(self.prediction_path + 'repeat{}/{}_fgs.dat'.format(self.repeat, pdbid), 'w') as f:
                f.write("{:13s} {}\n".format('#code', "fgs"))
                for n, fgs in zip(name, pred_fgs):
                    fgs = np.array(fgs).reshape(-1)
                    f.write("{:13s} {}\n".format(n, ','.join([str(item) for item in fgs])))

    def get_results_template(self):
        results = {}
        for task in self.list_task:
            results[task] = {"pred":[], "label":[]}
        return results

    def save(self, dict_collect, results):
        pickle.dump(dict_collect, open(self.prediction_path + "dict_collect", "wb"))
        json.dump(results, open(self.prediction_path + "results", "w"), indent=4, sort_keys=True)
        logging.info("Prediction results saved at " + self.prediction_path)


def main():
    parser = argparse.ArgumentParser()
    # define environment
    parser.add_argument("--gpu", default="0", help="which GPU to use", type=str)
    parser.add_argument("--seed", default=1234, help="random seed", type=int)
    parser.add_argument("--info", default="test", help="output folder special marker", type=str)
    parser.add_argument("--path", default="./", help="data path", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)

    # define task
    parser.add_argument("--dataset", default='casf2016', help="dataset", type=str)
    parser.add_argument("--model", default="PoseFGS", help="model", type=str)
    parser.add_argument("--model_path", default="./trained_models/", help="path to saved model", type=str)

    # define parameters
    parser.add_argument("--dataset_params", default="", help="dict of dataset parameters", type=str)
    parser.add_argument("--model_params", default="", help="dict of model parameters", type=str)
    parser.add_argument("--train_params", default="", help="dict of training parameters", type=str)
    args = parser.parse_args()

    """ claim class instance """
    tblo = PoseFGSPrediction(args)
        
    """ Train """
    tblo.predict()


if __name__ == "__main__":
    main()

