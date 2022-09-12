from configs.base import ParamManager
from data.base import DataManager
from methods import method_map
from backbones.base import ModelManager
from utils.functions import set_torch_seed, save_results, set_output_path

import argparse
import logging
import os
import datetime
import itertools
import warnings

def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logger_name', type=str, default='mia', help="Logger name for multimodal intent analysis.")

    parser.add_argument('--dataset', type=str, default='MIntRec', help="The name of the used dataset.")

    parser.add_argument('--data_mode', type=str, default='multi-class', help="The task mode (multi-class or binary-class).")

    parser.add_argument('--method', type=str, default='text', help="which method to use (text, mult, misa, mag_bert).")

    parser.add_argument("--text_backbone", type=str, default='bert', help="which backbone to use for the text modality.")

    parser.add_argument('--seed', type=int, default=0, help="The random seed for initialization.")

    parser.add_argument('--num_workers', type=int, default=8, help="The number of workers to load data.")

    parser.add_argument('--gpu_id', type=str, default='0', help="The used gpu index of your device.")

    parser.add_argument("--data_path", default = 'datasets', type=str,
                        help="The input data dir. Should contain text, video and audio data for the task.")

    parser.add_argument("--train", action="store_true", help="Whether to train the model.")

    parser.add_argument("--tune", action="store_true", help="Whether to tune the model with a series of hyper-parameters.")

    parser.add_argument("--save_model", action="store_true", help="whether to save trained-model for multimodal intent recognition.")

    parser.add_argument("--save_results", action="store_true", help="whether to save final results for multimodal intent recognition.")

    parser.add_argument('--log_path', type=str, default='logs', help="Logger directory.")
    
    parser.add_argument('--cache_path', type=str, default='cache', help="The caching directory for pre-trained models.")   

    parser.add_argument('--video_data_path', type=str, default='video_data', help="The directory of the video data.")

    parser.add_argument('--audio_data_path', type=str, default='audio_data', help="The directory of the audio data.")

    parser.add_argument('--video_feats_path', type=str, default='video_feats.pkl', help="The directory of the video features.")

    parser.add_argument('--audio_feats_path', type=str, default='audio_feats.pkl', help="The directory of the audio features.")

    parser.add_argument('--results_path', type=str, default='results', help="The path to save results.")

    parser.add_argument("--output_path", default= 'outputs', type=str, 
                        help="The output directory where all train data will be written.") 

    parser.add_argument("--model_path", default='models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 

    parser.add_argument("--config_file_name", type=str, default='text_bert_tune(.py)', help = "The name of the config file.")

    parser.add_argument("--results_file_name", type=str, default = 'results.csv', help="The file name of all the experimental results.")    

    args = parser.parse_args()

    return args

def set_logger(args):
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.logger_name =  f"{args.method}_{args.dataset}_{args.data_mode}_{time}"

    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(args.log_path, args.logger_name + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    return logger

def run(args, debug_args=None):
    
    logger = set_logger(args)
    args.pred_output_path, args.model_output_path = set_output_path(args)
    
    set_torch_seed(args.seed)

    logger.info("="*30+" Params "+"="*30)
    for k in args.keys():
        logger.info(f"{k}: {args[k]}")
    logger.info("="*30+" End Params "+"="*30)
    
    data = DataManager(args)
    method_manager = method_map[args.method]
    
    if args.method == 'text':
        method = method_manager(args, data)
    else:
        model = ModelManager(args)
        method = method_manager(args, data, model)
        
    logger.info('Multimodal intent recognition begins...')

    if args.train:

        logger.info('Training begins...')
        method._train(args)
        logger.info('Training is finished...')

    logger.info('Testing begins...')
    outputs = method._test(args)
    logger.info('Testing is finished...')
    
    logger.info('Multimodal intent recognition is finished...')

    if args.save_results:
        
        logger.info('Results are saved in %s', str(os.path.join(args.results_path, args.results_file_name)))
        save_results(args, outputs, debug_args=debug_args)
    
if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    
    args = parse_arguments()
    param = ParamManager(args)
    args = param.args

    if args.tune:
        debug_args = {}

        for k,v in args.items():
            if isinstance(v, list):
                debug_args[k] = v

        for result in itertools.product(*debug_args.values()):
            for i, key in enumerate(debug_args.keys()):
                args[key] = result[i]         
            
            run(args, debug_args=debug_args)

    else:
        run(args)
    

