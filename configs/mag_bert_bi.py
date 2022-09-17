class Param():
    
    def __init__(self, args):
        
        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        
        common_parameters = {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': True,
            'eval_monitor': 'f1',
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            beta_shift (float): The coefficient for nonverbal displacement to create the multimodal vector.
            dropout_prob (float): The embedding dropout probability.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr (float): The learning rate of backbone.
            aligned_method (str): The method for aligning different modalities. ('ctc', 'conv1d', 'avg_pool')
            weight_decay (float): The coefficient for L2 regularization. 
        """
        hyper_parameters = {
            'num_train_epochs': 100,
            'beta_shift': 0.04,
            'dropout_prob': 0.5,
            'warmup_proportion': 0.1,
            'lr': 0.00001,
            'aligned_method': 'ctc',
            'weight_decay': 0.1
            
        }
        return hyper_parameters