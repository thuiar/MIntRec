import torch
import torch.nn.functional as F
import logging
from torch import nn, optim
from tqdm import trange, tqdm
from utils.functions import restore_model, save_model, EarlyStopping
from utils.metrics import AverageMeter, Metrics
from .utils import MSE, CMD, DiffLoss

__all__ = ['MISA']


class MISA:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.gamma)

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']

        self.args = args
        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)

    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):
                    
                    outputs = self.model(text_feats, video_feats, audio_feats)

                    cls_loss = self.criterion(outputs, label_ids)
                    diff_loss = self._get_diff_loss()
                    domain_loss = self._get_domain_loss()
                    recon_loss = self._get_recon_loss()
                    cmd_loss = self._get_cmd_loss()
                    
                    if self.args.use_cmd_sim:
                        similarity_loss = cmd_loss
                    else:
                        similarity_loss = domain_loss

                    loss = cls_loss + \
                    args.diff_weight * diff_loss + \
                    args.sim_weight * similarity_loss + \
                    args.recon_weight * recon_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    
                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    self.optimizer.step()

            outputs = self._get_outputs(args, mode = 'eval')
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4)
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            early_stopping(eval_score, self.model)
            if early_stopping.counter != 0:
                self.lr_scheduler.step()

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break
        
          
        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)     
        
    def _get_outputs(self, args, mode = 'eval', return_sample_results = False, show_results = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)

        loss_record = AverageMeter()   

        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                
                logits = self.model(text_feats, video_feats, audio_feats)

                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))

                loss = self.criterion(logits, label_ids)
                loss_record.update(loss.item(), label_ids.size(0))

        
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        
        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs.update({'loss': loss_record.avg})

        if return_sample_results:

            outputs.update(
                {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            )

        return outputs

    def _get_domain_loss(self):

        if self.args.use_cmd_sim:
            return 0.0
        
        # Predicted domain labels
        domain_pred_t = self.model.model.domain_label_t
        domain_pred_v = self.model.model.domain_label_v
        domain_pred_a = self.model.model.domain_label_a

        # True domain labels
        domain_true_t = torch.LongTensor([0]*domain_pred_t.size(0)).to(self.device)
        domain_true_v = torch.LongTensor([1]*domain_pred_v.size(0)).to(self.device)
        domain_true_a = torch.LongTensor([2]*domain_pred_a.size(0)).to(self.device)

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def _get_cmd_loss(self):

        if not self.args.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.model.utt_shared_t, self.model.model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.model.utt_shared_t, self.model.model.utt_shared_a, 5)
        loss += self.loss_cmd(self.model.model.utt_shared_a, self.model.model.utt_shared_v, 5)
        loss = loss / 3.0

        return loss

    def _get_diff_loss(self):

        shared_t = self.model.model.utt_shared_t
        shared_v = self.model.model.utt_shared_v
        shared_a = self.model.model.utt_shared_a
        private_t = self.model.model.utt_private_t
        private_v = self.model.model.utt_private_v
        private_a = self.model.model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss

    def _get_recon_loss(self, ):

        loss = self.loss_recon(self.model.model.utt_t_recon, self.model.model.utt_t_orig)
        loss += self.loss_recon(self.model.model.utt_v_recon, self.model.model.utt_v_orig)
        loss += self.loss_recon(self.model.model.utt_a_recon, self.model.model.utt_a_orig)
        loss = loss / 3.0
        return loss

    def _test(self, args):

        test_results = self._get_outputs(args, mode = 'test', return_sample_results=True, show_results = True)
        test_results['best_eval_score'] = round(self.best_eval_score, 4)
    
        return test_results