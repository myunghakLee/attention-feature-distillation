# Attention-based Feature-level Distillation
# Original Source : https://github.com/HobbitLong/RepDistiller

import argparse
from utils import *
from dataset import *
from distill import *
import models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from time import time
from torch.optim.lr_scheduler import _LRScheduler
import time
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def str2bool(s):
    if s not in {'F', 'T'}:
        raise ValueError('Not a valid boolean string')
    return s == 'T'

def train(epoch, device, model, cifar100_training_loader, args, optimizer):

    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
        
        if batch_index % 50 == 0:

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.batch_size + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))

        #update training loss for each iteration

        if epoch <= args.warm:
            warmup_scheduler.step()


    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
def eval_training(epoch, device, model, cifar100_test_loader, tb=True):

    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    return correct.float() / len(cifar100_test_loader.dataset)
    
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    
    
# def main():
EPOCH = 200
MILESTONES = [60, 120, 160]
SAVE_EPOCH = 10

parser = argparse.ArgumentParser()
# data
parser.add_argument('--data_dir', default=r'D:\Image\Image classification')
parser.add_argument('--data', default='CIFAR100')
#     parser.add_argument('--trained_dir', default='trained/wrn40x2/model.pth')

parser.add_argument('--epoch', default=240, type=int)
parser.add_argument('--schedule', default=[150, 180, 210], type=int, nargs='+')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--lr_decay', default=0.1, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

parser.add_argument('--alpha', default=0.9, type=float, help='weight for KD (Hinton)')
parser.add_argument('--temperature', default=4, type=float)

parser.add_argument('--model', default='wrn16x2', type=str)

parser.add_argument('--beta', default=200, type=float)
parser.add_argument('--qk_dim', default=128, type=int)

parser.add_argument('--use_vanilla', action='store_true', default=False)
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')

args = parser.parse_args()

for param in sorted(vars(args).keys()):
    print('--{0} {1}'.format(param, vars(args)[param]))

cifar100_training_loader, cifar100_test_loader, args.num_classes, args.image_size = create_loader(args.batch_size, args.data_dir, args.data, args.use_vanilla)

print(models.__dict__.keys())
model = models.__dict__[args.model](num_classes=args.num_classes)
device = torch.device('cuda')
model = model.to(device)

loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2) #learning rate decay
iter_per_epoch = len(cifar100_training_loader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

checkpoint_path = f"saved_models/{args.model}/"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

best_acc = 0.0
accs = []

for epoch in range(1, EPOCH + 1):
    if epoch > args.warm:
        train_scheduler.step(epoch)
        print(f"lr : {train_scheduler.get_last_lr()}")

    train(epoch, device, model, cifar100_training_loader, args, optimizer)
    acc = eval_training(epoch, device, model, cifar100_test_loader)
    accs.append(acc.item())
    print("==================================================test acc==================================================")
    print("acc : ", acc.item())
    #start to save best performance model after learning rate decay to 0.01
    if epoch > MILESTONES[1] and best_acc < acc:
        torch.save(model, checkpoint_path + "best.pth")
        best_acc = acc
        continue

    if not epoch % SAVE_EPOCH:
        torch.save(model, checkpoint_path + f"{str(epoch).zfill(3)}.pth")

    with open(checkpoint_path + "acc.json" , "w") as f:
        json.dump(accs, f)



