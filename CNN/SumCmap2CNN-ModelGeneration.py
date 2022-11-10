# https://www.kaggle.com/code/sevenc/mnist-digit-recognition-using-pytorch/edit

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
import torchvision.models as models 
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader
from torch.utils.data import TensorDataset, DataLoader

import os
import numpy as np
import argparse
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.image import imread
import random
import copy

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input cmap threshold for SVM training.')
    parser.add_argument('cmap_threshold', help='an integer to select cmap folder for SVM training', type=int)
    args = parser.parse_args()

    # classes and training set
    classes = os.listdir('./train' + str(args.cmap_threshold) + '/')
    alpha_helix_files = os.listdir('./train' + str(args.cmap_threshold) + '/alpha_helix/')
    beta_strand_files = os.listdir('./train' + str(args.cmap_threshold) + '/beta_strand/')

    # transform training cmaps to tensors 
    train_dataset = []
    for ah_mat in alpha_helix_files:
        ah_cmap_mat = pd.read_csv('./train' + str(args.cmap_threshold) + '/alpha_helix/' + ah_mat, delim_whitespace=True, header=None)
        ah_cmap_pd = pd.DataFrame(ah_cmap_mat)
        ah_cmap_array = np.array(ah_cmap_pd)
        ah_cmap = torch.tensor(ah_cmap_array)
        ah_cmap_w_label = (ah_cmap, 0)
        train_dataset.append(ah_cmap_w_label)
    for bs_mat in beta_strand_files:
        bs_cmap_mat = pd.read_csv('./train' + str(args.cmap_threshold) + '/beta_strand/' + bs_mat, delim_whitespace=True, header=None)
        bs_cmap_pd = pd.DataFrame(bs_cmap_mat)
        bs_cmap_array = np.array(bs_cmap_pd)
        bs_cmap = torch.tensor(bs_cmap_array)
        bs_cmap_w_label = (bs_cmap, 1)
        train_dataset.append(bs_cmap_w_label)
    
    # train and validation datasets preparation
    random_seed = 2022
    torch.manual_seed(random_seed);
    train_size = round(len(train_dataset)*0.7)
    val_size = len(train_dataset) - train_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
    batch_size = 16
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds
        

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
            self.conv_block = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            
            self.linear_block = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(64*10*10, 10),
                nn.ReLU(inplace=True),
                nn.Linear(10, 2),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            x = self.conv_block(x)
            x = x.view(x.size(0), -1)
            x = self.linear_block(x)
            
            return x    

        def validation_step(self, batch):
            images,labels = batch
            out = self(images.float())                                      # generate predictions
            loss = F.cross_entropy(out, labels)                     # compute loss
            acc,preds = accuracy(out, labels)                       # calculate acc & get preds
            
            return {'val_loss': loss.detach(), 'val_acc':acc.detach(), 
                    'preds':preds.detach(), 'labels':labels.detach()}

        # this is for using on the test set, it outputs the average loss and acc, 
        # and outputs the predictions
        def test_prediction(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()           # combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies
            # combine predictions
            batch_preds = [pred for x in outputs for pred in x['preds'].tolist()] 
            # combine labels
            batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]  
            
            return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(),
                    'test_preds': batch_preds, 'test_labels': batch_labels}
        

    conv_model = Net()

    optimizer = optim.Adam(params=conv_model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if torch.cuda.is_available():
        conv_model = conv_model.cuda()
        criterion = criterion.cuda()

    def train_model(num_epoch):
        conv_model.train()
        exp_lr_scheduler.step()
        
        for batch_idx, (data, target) in enumerate(train_dl):
            data = data.unsqueeze(1)
            data, target = data.float(), target
            # print(data.type(), target.type())
            # print(data.size(), target.size())
            # print(batch_idx)
        
            
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                
            optimizer.zero_grad()
            output = conv_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1)% 5 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    num_epoch, (batch_idx + 1) * len(data), len(train_dl.dataset),
                    100. * (batch_idx + 1) / len(train_dl), loss.data))
                
    def evaluate(data_loader):
        conv_model.eval()
        loss = 0
        correct = 0
        
        for data, target in data_loader:
            data = data.unsqueeze(1)
            data, target = data.float(), target
            
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            
            output = conv_model(data)
            
            loss += F.cross_entropy(output, target, size_average=False).data

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
        loss /= len(data_loader.dataset)
            
        print('\nAverage Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
            loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))

    num_epochs = 10

    for n in range(num_epochs):
        train_model(n)
        evaluate(val_dl)

    alpha_helix_testfiles = os.listdir('./test' + str(args.cmap_threshold) + '/alpha_helix/')
    beta_strand_testfiles = os.listdir('./test' + str(args.cmap_threshold) + '/beta_strand/')
    test_dataset = []
    for test_ah_mat in alpha_helix_testfiles:
        test_ah_cmap_mat = pd.read_csv('./test' + str(args.cmap_threshold) + '/alpha_helix/' + test_ah_mat, delim_whitespace=True, header=None)
        test_ah_cmap_pd = pd.DataFrame(test_ah_cmap_mat)
        test_ah_cmap_array = np.array(test_ah_cmap_pd)
        test_ah_cmap = torch.tensor([test_ah_cmap_array])
        test_ah_cmap_w_label = (test_ah_cmap, 0)
        test_dataset.append(test_ah_cmap_w_label)
    for test_bs_mat in beta_strand_testfiles:
        test_bs_cmap_mat = pd.read_csv('./test' + str(args.cmap_threshold) + '/beta_strand/' + test_bs_mat, delim_whitespace=True, header=None)
        test_bs_cmap_pd = pd.DataFrame(test_bs_cmap_mat)
        test_bs_cmap_array = np.array(test_bs_cmap_pd)
        test_bs_cmap = torch.tensor([test_bs_cmap_array])
        test_bs_cmap_w_label = (test_bs_cmap, 1)
        test_dataset.append(test_bs_cmap_w_label)
    test_dl = DataLoader(test_dataset, batch_size*2, num_workers=0, pin_memory=True)
    
    best_model = {
        'model': Net(),
        'state_dict': conv_model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(best_model, 'CNN-' + str(args.cmap_threshold) + '.pth')

     # predicting on test set
    @torch.no_grad()
    def test_predict(model, test_loader):
        model.eval()
        # perform testing for each batch
        outputs = [model.validation_step(batch) for batch in test_loader] 
        results = model.test_prediction(outputs)                          
        print('test_loss: {:.4f}, test_acc: {:.4f}'
            .format(results['test_loss'], results['test_acc']))
        
        return results['test_preds'], results['test_labels']

     # Evaluate test set
    test_dl = DataLoader(test_dataset, batch_size=16)
    preds,labels = test_predict(conv_model, test_dl)

    # plot confusion matrix 
    cm = confusion_matrix(labels, preds)
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8),cmap=plt.cm.Blues)
    plt.xticks(range(2), ['alpha_helix', 'beta_strand'], fontsize=16)
    plt.yticks(range(2), ['alpha_helix', 'beta_strand'], fontsize=16)
    plt.xlabel('Predicted Label',fontsize=18)
    plt.ylabel('True Label',fontsize=18)
    plt.savefig('CNN-' + str(args.cmap_threshold) + '-confusion_matrix.png')

    # compute performance metrics
    tn, fp, fn, tp = cm.ravel()

    acc = (np.array(preds) == np.array(labels)).sum() / len(preds)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))

    print("Accuracy of the model is {:.2f}".format(acc))
    print("Recall of the model is {:.2f}".format(recall))
    print("Precision of the model is {:.2f}".format(precision))
    print("F1 Score of the model is {:.2f}".format(f1))
