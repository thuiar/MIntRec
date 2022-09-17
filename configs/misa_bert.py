class Param():
    
    def __init__(self, args):
        
        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        """
            padding_mode (str): The mode for sequence padding ('zero' or 'normal').
            padding_loc (str): The location for sequence padding ('start' or 'end'). 
            eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').  
            need_aligned: (bool): Whether to perform data alignment between different modalities.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patience (int): Patient steps for Early Stop.
        """
        common_parameters = {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': False,
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
            rnn_cell (str): The cell for the recurrent neural network ('lstm' or 'gru').
            use_cmd_sim (bool): Whether to use the cmd loss as the similarity loss.
            reverse_grad_weight (float): The gradient weight of the reverse layer.
            hidden_size (int): The hidden layer size.
            dropout_rate (float): The dropout rate for fusion layer or discriminator layer.
            diff_weight (float): The weight for the difference loss.
            sim_weight (float): The weight for the similarity loss.
            recon_weight (float): The weight for the reconstruction loss.
            lr (float): The learning rate of backbone.
            grad_clip (float): The gradient clip value.
            gamma (float): The base of the exponential learning rate scheduler.
        """
        hyper_parameters = {
            'num_train_epochs': 100,
            'rnncell': 'lstm',
            'use_cmd_sim': False,
            'reverse_grad_weight': 0.8,
            'hidden_size': 256,
            'dropout_rate': 0.1,
            'diff_weight': 0.7,
            'sim_weight': 0.7,
            'recon_weight': 0.6,
            'lr': 0.00003,
            'grad_clip': -1.0,
            'gamma': 0.5,
        }
        return hyper_parameters