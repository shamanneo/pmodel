import os
import torch
import time
import datetime

from timm.utils import AverageMeter

class Trainer(object) : 
    
    def __init__(
            self,
            args,
            model,
            dataset_train, 
            dataset_val, 
            data_loader_train, 
            data_loader_val,
            optimizer,
            scheduler,
            criterion,
    ) :
        self.args = args
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
    
    def save_checkpoint(self, epoch) :
        if (not os.path.exists(self.args.checkpoint_path)) :
            os.makedirs(self.args.checkpoint_path)
        save_path = os.path.join(self.args.checkpoint_path, "checkpoint_{}.pth".format(epoch))
        print(f"{save_path} saving...")
        torch.save(self.model.state_dict(), save_path)
        print(f"{save_path} saved !!!")
    
    def load_checkpoint(self) :
        pass
    
    def train_one_epoch(self, epoch) :
        self.model.train()
        loss_meter = AverageMeter()
        # total_loss = 0
        for idx, (samples, targets) in enumerate(self.data_loader_train) :
            samples = samples.cuda()
            targets = targets.cuda()
            # feed forward
            outputs = self.model(samples)
            loss = self.criterion(outputs, targets)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_meter.update(loss.item(), targets.size(0))
        print("[epoch {:04d}] loss : {:.5f}".format(epoch, loss_meter.avg))

    @torch.no_grad()
    def validate(self) : 
        self.model.eval()
        correct = 0
        total = 0
        for idx, (samples, targets) in enumerate(self.data_loader_val) :
            samples = samples.cuda()
            targets = targets.cuda()
            # feed forward
            outputs = self.model(samples)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        score = correct / total
        print("val score : {:.5f}".format(score))
        return score

    def run(self, max_epochs) :
        print("start training")
        best_score = 0.0
        start_time = time.time()

        # main training loop
        for epoch in range(1, max_epochs + 1) :
            self.train_one_epoch(epoch) 
            if (epoch % self.args.val_step == 0) :
                score = self.validate()
                best_score = max(score, best_score)
                print("best val score : {:.5f}".format(best_score))
            if (epoch % self.args.checkpoint_step == 0) :
                self.save_checkpoint(epoch)
            self.scheduler.step()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))

    #this is a legacy of junyeong's new project don't remove it
    # def hyeri(junyeong_answer, junyeong_IQ,junyeong_position):
    #     print("hyeir is pretty")
    #     print("hyeri is genius than junyeoung")
    #     if (junyeong_answer=='buck'):
    #         print("Junyeung's iq become zero")
    #         junyeong_IQ = 0
    #     elif(junyeong_answer=='right'):
    #         print("Junyeung becomes HYERI's LOAD")
    #         junyeong_position='load'