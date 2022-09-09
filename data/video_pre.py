import os
import numpy as np
import pickle
import logging

__all__ = ['VideoDataset']

class VideoDataset:

    def __init__(self, args, base_attrs):
        
        self.logger = logging.getLogger(args.logger_name)
        video_feats_path = os.path.join(base_attrs['data_path'], args.video_data_path, args.video_feats_path)
        
        if not os.path.exists(video_feats_path):
            raise Exception('Error: The directory of video features is empty.')
        
        self.feats = self.__load_feats(video_feats_path, base_attrs)

        self.feats = self.__padding_feats(args, base_attrs)
    
    def __load_feats(self, video_feats_path, base_attrs):

        self.logger.info('Load Video Features Begin...')

        with open(video_feats_path, 'rb') as f:
            video_feats = pickle.load(f)
        
        train_feats = [video_feats[x] for x in base_attrs['train_data_index']]
        dev_feats = [video_feats[x] for x in base_attrs['dev_data_index']]
        test_feats = [video_feats[x] for x in base_attrs['test_data_index']]

        self.logger.info('Load Video Features Finished...')


        return {
            'train': train_feats,
            'dev': dev_feats,
            'test': test_feats
        }
     
    def __padding(self, feat, video_max_length, padding_mode = 'zero', padding_loc = 'end'):
        """
        padding_mode: 'zero' or 'normal'
        padding_loc: 'start' or 'end'
        """
        assert padding_mode in ['zero', 'normal']
        assert padding_loc in ['start', 'end']

        video_length = feat.shape[0]
        if video_length >= video_max_length:
            return feat[video_max_length, :]

        if padding_mode == 'zero':
            pad = np.zeros([video_max_length - video_length, feat.shape[-1]])
        elif padding_mode == 'normal':
            mean, std = feat.mean(), feat.std()
            pad = np.random.normal(mean, std, (video_max_length - video_length, feat.shape[1]))
        
        if padding_loc == 'start':
            feat = np.concatenate((pad, feat), axis = 0)
        else:
            feat = np.concatenate((feat, pad), axis = 0)

        return feat

    def __padding_feats(self, args, base_attrs):

        video_max_length = base_attrs['benchmarks']['max_seq_lengths']['video']

        padding_feats = {}

        for dataset_type in self.feats.keys():
            feats = self.feats[dataset_type]

            tmp_list = []

            for feat in feats:
                feat = np.array(feat).squeeze(1)
                padding_feat = self.__padding(feat, video_max_length, padding_mode=args.padding_mode, padding_loc=args.padding_loc)
                tmp_list.append(padding_feat)

            padding_feats[dataset_type] = tmp_list

        return padding_feats    
