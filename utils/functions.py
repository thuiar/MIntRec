import os
import torch
import numpy as np
import pandas as pd
import random
import logging
import copy
from .metrics import Metrics

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, args, delta=1e-6):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.         
        """
        self.patience = args.wait_patience
        self.logger = logging.getLogger(args.logger_name)
        self.monitor = args.eval_monitor
        self.counter = 0
        self.best_score = 1e8 if self.monitor == 'loss' else 1e-6
        self.early_stop = False
        self.delta = delta
        self.best_model = None

    def __call__(self, score, model):
        
        better_flag = score <= (self.best_score - self.delta) if self.monitor == 'loss' else score >= (self.best_score + self.delta) 

        if better_flag:
            self.counter = 0
            self.best_model = copy.deepcopy(model)
            self.best_score = score 

        else:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')  

            if self.counter >= self.patience:
                self.early_stop = True
            
def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_output_path(args):
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    pred_output_path = os.path.join(args.output_path, args.logger_name)
    if not os.path.exists(pred_output_path):
        os.makedirs(pred_output_path)

    model_path = os.path.join(pred_output_path, args.model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return pred_output_path, model_path

def save_npy(npy_file, path, file_name):
    npy_path = os.path.join(path, file_name)
    np.save(npy_path, npy_file)

def load_npy(path, file_name):
    npy_path = os.path.join(path, file_name)
    npy_file = np.load(npy_path)
    return npy_file

def save_model(model, model_dir):

    save_model = model.module if hasattr(model, 'module') else model 
    model_file = os.path.join(model_dir, 'pytorch_model.bin')
    model_config_file = os.path.join(model_dir, 'config.json')
    torch.save(save_model.state_dict(), model_file)
    with open(model_config_file, "w") as f:
        f.write(save_model.config.to_json_string())

def restore_model(model, model_dir):
    output_model_file = os.path.join(model_dir, 'pytorch_model.bin')
    model.load_state_dict(torch.load(output_model_file))
    return model

def save_results(args, test_results, debug_args = None):

    pred_labels_path = os.path.join(args.pred_output_path, 'y_pred.npy')
    np.save(pred_labels_path, test_results['y_pred'])
    
    true_labels_path = os.path.join(args.pred_output_path, 'y_true.npy')
    np.save(true_labels_path, test_results['y_true'])

    if 'features' in test_results.keys():
        features_path = os.path.join(args.pred_output_path, 'features.npy')
        np.save(features_path, test_results['features'])

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    results = {}
    metrics = Metrics(args)
    for key in metrics.eval_metrics:
        results[key] = round(test_results[key] * 100, 2)

    eval_key = 'eval_' + args.eval_monitor
    results.update({eval_key: test_results['best_eval_score']})

    _vars = [args.dataset, args.method, args.text_backbone, args.seed, args.logger_name]
    _names = ['dataset',  'method', 'text_backbone', 'seed', 'logger_name']

    if debug_args is not None:
        _vars.extend([args[key] for key in debug_args.keys()])
        _names.extend(debug_args.keys())

    vars_dict = {k:v for k,v in zip(_names, _vars)}
    results = dict(results,**vars_dict)

    keys = list(results.keys())
    values = list(results.values())
    
    results_path = os.path.join(args.results_path, args.results_file_name)
    
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        df1 = df1.append(new,ignore_index=True)
        df1.to_csv(results_path,index=False)
    data_diagram = pd.read_csv(results_path)
    
    print('test_results', data_diagram)

