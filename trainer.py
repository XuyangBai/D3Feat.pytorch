import time, os
import numpy as np
from tensorboardX import SummaryWriter
import torch
from utils.timer import Timer, AverageMeter
from utils.metrics import calculate_acc, calculate_iou


class Trainer(object):
    def __init__(self, args):
        self.config = args
        # parameters
        self.start_epoch = 0
        self.max_epoch = args.max_epoch
        self.training_max_iter = args.training_max_iter
        self.val_max_iter = args.val_max_iter
        self.save_dir = args.save_dir
        self.device = args.device
        self.verbose = args.verbose
        self.best_acc = 0
        self.best_loss = 10000000

        self.model = args.model.to(self.device)
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.evaluation_metric = args.evaluation_metric
        self.metric_weight = args.metric_weight
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        # self.writer = SummaryWriter(logdir=args.tboard_dir)

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

        self.train_loader = args.train_loader
        self.val_loader = args.val_loader
        self.test_loader = args.test_loader
        

    def train(self):

        self.model.train()
        # res = self.evaluate(self.start_epoch)
        # for k,v in res.items():
            # self.writer.add_scalar(f'val/{k}', v, 0)
        for epoch in range(self.start_epoch, self.max_epoch):
            self.train_epoch(epoch + 1)

            if (epoch + 1) % 1 == 0:
                res = self.evaluate(epoch + 1)
                if res['desc_loss'] < self.best_loss:
                    self.best_loss = res['desc_loss']
                    self._snapshot(epoch + 1, 'best_loss')
                if res['accuracy'] > self.best_acc:
                    self.best_acc = res['accuracy']
                    self._snapshot(epoch + 1, 'best_acc')

            for k,v in res.items():
                self.writer.add_scalar(f'val/{k}', v, epoch + 1)
                    
            if (epoch + 1) % self.scheduler_interval == 0:
                self.scheduler.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)


        # finish all epoch
        print("Training finish!... save training results")

    def train_epoch(self, epoch):
        data_timer, model_timer = Timer(), Timer()
        desc_loss_meter, det_loss_meter, acc_meter, d_pos_meter, d_neg_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        num_iter = int(len(self.train_loader.dataset) // self.train_loader.batch_size)
        num_iter = min(self.training_max_iter, num_iter)
        train_loader_iter = self.train_loader.__iter__()
        # for iter, inputs in enumerate(self.train_loader):
        for iter in range(num_iter):
            data_timer.tic()
            inputs = train_loader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                else:
                    inputs[k] = v.to(self.device)
            data_timer.toc()

            model_timer.tic()
            # forward
            self.optimizer.zero_grad()
            features, scores = self.model(inputs)
            anc_features = features[inputs["corr"][:, 0].long()]
            pos_features = features[inputs["corr"][:, 1].long() + inputs['stack_lengths'][0][0]]
            anc_scores = scores[inputs["corr"][:, 0].long()]
            pos_scores = scores[inputs["corr"][:, 1].long() + inputs['stack_lengths'][0][0]]
            
            desc_loss, acc, d_pos, d_neg, _, dist = self.evaluation_metric["desc_loss"](anc_features, pos_features, inputs['dist_keypts'])
            det_loss = self.evaluation_metric['det_loss'](dist, anc_scores, pos_scores)
            loss = desc_loss * self.metric_weight['desc_loss'] + det_loss * self.metric_weight['det_loss']
            d_pos = np.mean(d_pos)
            d_neg = np.mean(d_neg)

            # backward
            loss.backward()
            do_step = True
            for param in self.model.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        break
            if do_step is True:
                self.optimizer.step()
            # if self.config.grad_clip_norm > 0:
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.grad_clip_norm)
            model_timer.toc()
            desc_loss_meter.update(float(desc_loss))
            det_loss_meter.update(float(det_loss))
            d_pos_meter.update(float(d_pos))
            d_neg_meter.update(float(d_neg))
            acc_meter.update(float(acc))

            if (iter + 1) % 100 == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + iter
                self.writer.add_scalar('train/Desc_Loss', float(desc_loss_meter.avg), curr_iter)
                self.writer.add_scalar('train/Det_Loss', float(det_loss_meter.avg), curr_iter)
                self.writer.add_scalar('train/D_pos', float(d_pos_meter.avg), curr_iter)
                self.writer.add_scalar('train/D_neg', float(d_neg_meter.avg), curr_iter)
                self.writer.add_scalar('train/Accuracy', float(acc_meter.avg), curr_iter)
                print(f"Epoch: {epoch} [{iter+1:4d}/{num_iter}] "
                      f"desc loss: {desc_loss_meter.avg:.2f} "
                      f"det loss: {det_loss_meter.avg:.2f} "
                      f"acc:  {acc_meter.avg:.2f} "
                      f"d_pos: {d_pos_meter.avg:.2f} "
                      f"d_neg: {d_neg_meter.avg:.2f} "
                      f"data time: {data_timer.avg:.2f}s "
                      f"model time: {model_timer.avg:.2f}s")
        # finish one epoch
        epoch_time = model_timer.total_time + data_timer.total_time
        print(f'Epoch {epoch}: Desc Loss: {desc_loss_meter.avg:.2f}, Det Loss : {det_loss_meter.avg:.2f}, Accuracy: {acc_meter.avg:.2f}, D_pos: {d_pos_meter.avg:.2f}, D_neg: {d_neg_meter.avg:.2f}, time {epoch_time:.2f}s')

    def evaluate(self, epoch):
        self.model.eval()
        data_timer, model_timer = Timer(), Timer()
        desc_loss_meter, det_loss_meter, acc_meter, d_pos_meter, d_neg_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        num_iter = int(len(self.val_loader.dataset) // self.val_loader.batch_size)
        num_iter = min(self.val_max_iter, num_iter)
        test_loader_iter = self.val_loader.__iter__()
        for iter in range(num_iter):
            data_timer.tic()
            inputs = test_loader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                else:
                    inputs[k] = v.to(self.device)
            data_timer.toc()

            model_timer.tic()
            features, scores = self.model(inputs)
            anc_features = features[inputs["corr"][:, 0].long()]
            pos_features = features[inputs["corr"][:, 1].long() + inputs['stack_lengths'][0][0]]
            anc_scores = scores[inputs["corr"][:, 0].long()]
            pos_scores = scores[inputs["corr"][:, 1].long() + inputs['stack_lengths'][0][0]]
            
            desc_loss, acc, d_pos, d_neg, _, dist = self.evaluation_metric['desc_loss'](anc_features, pos_features, inputs['dist_keypts'])
            det_loss = self.evaluation_metric['det_loss'](dist, anc_scores, pos_scores)
            loss = desc_loss * self.metric_weight['desc_loss'] + det_loss * self.metric_weight['det_loss']
            d_pos = np.mean(d_pos)
            d_neg = np.mean(d_neg)

            model_timer.toc()
            desc_loss_meter.update(float(desc_loss))
            det_loss_meter.update(float(det_loss))
            d_pos_meter.update(float(d_pos))
            d_neg_meter.update(float(d_neg))
            acc_meter.update(float(acc))

            if (iter + 1) % 100 == 0 and self.verbose:
                print(f"Eval epoch: {epoch+1} [{iter+1:4d}/{num_iter}] "
                      f"desc loss: {desc_loss_meter.avg:.2f} "
                      f"det loss: {det_loss_meter.avg:.2f} "
                      f"acc:  {acc_meter.avg:.2f} "
                      f"d_pos: {d_pos_meter.avg:.2f} "
                      f"d_neg: {d_neg_meter.avg:.2f} "
                      f"data time: {data_timer.avg:.2f}s "
                      f"model time: {model_timer.avg:.2f}s")
        self.model.train()
        res = {
            'desc_loss': desc_loss_meter.avg,
            'det_loss': det_loss_meter.avg,
            'accuracy': acc_meter.avg,
            'd_pos': d_pos_meter.avg,
            'd_neg': d_neg_meter.avg,
        }
        print(f'Evaluation: Epoch {epoch}: Desc Loss {res["desc_loss"]}, Det Loss {res["det_loss"]}, Accuracy {res["accuracy"]}')
        return res

    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        print(f"Save model to {filename}")
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        if os.path.isfile(resume):
            print(f"=> loading checkpoint {resume}")
            state = torch.load(resume)
            self.start_epoch = state['epoch']
            self.model.load_state_dict(state['state_dict'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_loss = state['best_loss']
            import pdb
            pdb.set_trace()
            # self.best_acc = state['best_acc']
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
