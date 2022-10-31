#!/usr/bin/env python

import torch
import torch.nn as nn
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
        ah_cmap_array = ah_cmap_pd.to_numpy()
        ah_cmap = torch.tensor(ah_cmap_array)
        ah_pad = nn.ZeroPad2d((140, 139, 140, 139)) # InceptionV3 - 299; VGG19 - 32.
        ah_output = ah_pad(ah_cmap)
        #print(ah_output)
        ah_output_3c = torch.stack((ah_output, ah_output, ah_output))
        ah_cmap_w_label = (ah_output_3c, 0)
        train_dataset.append(ah_cmap_w_label)
    for bs_mat in beta_strand_files:
        bs_cmap_mat = pd.read_csv('./train' + str(args.cmap_threshold) + '/beta_strand/' + bs_mat, delim_whitespace=True, header=None)
        bs_cmap_pd = pd.DataFrame(bs_cmap_mat)
        bs_cmap_array = bs_cmap_pd.to_numpy()
        bs_cmap = torch.tensor(bs_cmap_array)
        bs_pad = nn.ZeroPad2d((140, 139, 140, 139))
        bs_output = bs_pad(bs_cmap)
        #print(bs_output)
        bs_output_3c = torch.stack((bs_output, bs_output, bs_output))
        bs_cmap_w_label = (bs_output_3c, 1)
        train_dataset.append(bs_cmap_w_label)
    
    # train and validation datasets preparation
    random_seed = 2022
    torch.manual_seed(random_seed);
    train_size = round(len(train_dataset)*0.7)
    val_size = len(train_dataset) - train_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
    batch_size = 128
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)

    # GPU setting
    def get_default_device():
        """take GPU if available, else CPU"""
        if torch.cuda.is_available():
            print('Using GPU!')
            return torch.device('cuda')
        else:
            print('Using CPU!')
            return torch.device('cpu')
    device = get_default_device()

    def to_device(data, device):
        """move tensors to chosen device"""
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
    
    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)

    # Model creation 
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds
    
    def F1_score(outputs, labels):
        _, preds = torch.max(outputs, dim=1)

        # precision, recall and F1
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*((precision*recall)/(precision+recall))

        return precision, recall, f1, preds

    class InceptionV3ModelBase(nn.Module):
        
        def training_step(self, batch, weight):
            images,labels = batch
            out = self(images.float())                                      # generate predictions
            loss = F.cross_entropy(out, labels, weight=weight)      # weighted compute loss
            acc,preds = accuracy(out, labels)                       # calculate accuracy
            
            return {'train_loss': loss, 'train_acc':acc}
       
        # this is for computing the train average loss and acc for each epoch
        def train_epoch_end(self, outputs):
            batch_losses = [x['train_loss'] for x in outputs]       # get all the batches loss
            epoch_loss = torch.stack(batch_losses).mean()           # combine losses
            batch_accs = [x['train_acc'] for x in outputs]          # get all the batches acc
            epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies
            
            return {'train_loss': epoch_loss.item(), 'train_acc': epoch_acc.item()}
        
        # this is for loading the batch of val/test image and outputting its loss, accuracy, 
        # predictions & labels
        def validation_step(self, batch):
            images,labels = batch
            out = self(images.float())                                      # generate predictions
            loss = F.cross_entropy(out, labels)                     # compute loss
            acc,preds = accuracy(out, labels)                       # calculate acc & get preds
            
            return {'val_loss': loss.detach(), 'val_acc':acc.detach(), 
                    'preds':preds.detach(), 'labels':labels.detach()}
        # detach extracts only the needed number, or other numbers will crowd memory
        
        # this is for computing the validation average loss and acc for each epoch
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]         # get all the batches loss
            epoch_loss = torch.stack(batch_losses).mean()           # combine losses
            batch_accs = [x['val_acc'] for x in outputs]            # get all the batches acc
            epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies
            
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

        # this is for printing out the results after each epoch
        def epoch_end(self, epoch, train_result, val_result):
            print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.
                format(epoch+1, train_result['train_loss'], train_result['train_acc'],
                        val_result['val_loss'], val_result['val_acc']))
        
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

    # ATTENTION - in fact the original model pretrained by images is not transferable to our current matrix input
    inception_v3 = models.inception_v3(init_weights=True,aux_logits=False,transform_input=False)
    
    class InceptionV3(InceptionV3ModelBase):
        def __init__(self):
            super().__init__()
            # Use a pretrained model
            self.network = models.inception_v3(init_weights=True,aux_logits=False,transform_input=False)
            # Freeze training for all layers before classifier
            for param in self.network.fc.parameters():
                param.require_grad = False  
            num_features = self.network.fc.in_features # get number of in features of last layer
            self.network.fc = nn.Linear(num_features, 2) # replace model classifier
        
        def forward(self, xb):
            return self.network(xb)

    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def fit(epochs, lr, model, train_loader, val_loader, weight, 
                    weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
        torch.cuda.empty_cache() # release all the GPU memory cache
        history = {}
        
        optimizer = opt_func(model.parameters(), lr)

        best_loss = 1 # initialize best loss, which will be replaced with lower better loss
        for epoch in range(epochs):
            
            # Training Phase 
            model.train() 
            train_outputs = []      
            lrs = []
            
            for batch in train_loader:
                outputs = model.training_step(batch, weight)
                loss = outputs['train_loss']                          # get the loss
                train_outputs.append(outputs)
                # get the train average loss and acc for each epoch
                train_results = model.train_epoch_end(train_outputs)                        
                loss.backward()                                       # compute gradients
                
                # Gradient clipping
                if grad_clip: 
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
                optimizer.step()                                      # update weights
                optimizer.zero_grad()                                 # reset gradients  
            
            # Validation phase
            val_results = evaluate(model, val_loader)
            
            # Save best loss
            if val_results['val_loss'] < best_loss and epoch + 1 > 15:
                best_loss = min(best_loss, val_results['val_loss'])
                best_model_wts = copy.deepcopy(model.state_dict())
                #torch.save(model.state_dict(), 'best_model.pth')
            
            # print results
            model.epoch_end(epoch, train_results, val_results)
            
            # save results to dictionary
            to_add = {'train_loss': train_results['train_loss'],
                    'train_acc': train_results['train_acc'],
                    'val_loss': val_results['val_loss'],
                    'val_acc': val_results['val_acc'], 'lrs':lrs}
            
            # update performance dictionary
            for key,val in to_add.items():
                if key in history:
                    history[key].append(val)
                else:
                    history[key] = [val]
        
        model.load_state_dict(best_model_wts)                         # load best model
        
        return history, optimizer, best_loss

    # train and evaluate model
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    model = to_device(InceptionV3(), device)

    epochs = 20
    lr = 0.0001
    grad_clip = None
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    weight = torch.FloatTensor([0.5, 0.5]).to(device)

    history, optimizer, best_loss = fit(epochs, lr, model, train_dl, val_dl, weight, 
                                    grad_clip=grad_clip, 
                                    weight_decay=weight_decay, 
                                    opt_func=opt_func)

    print('Best loss is - ', best_loss)

    # save model
    bestmodel = {
        'model': InceptionV3(),
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(bestmodel, 'InceptionV3.pth')

    # plot acc and loss
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle('Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    epoch_list = list(range(1,epochs+1))
    ax1.plot(epoch_list, history['train_acc'], label='Train Accuracy')
    ax1.plot(epoch_list, history['val_acc'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(0, epochs+1, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, history['train_loss'], label='Train Loss')
    ax2.plot(epoch_list, history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, epochs+1, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")

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

    
    # transform test cmaps to tensors 
    test_alpha_helix_files = os.listdir('./test' + str(args.cmap_threshold) + '/alpha_helix/')
    test_beta_strand_files = os.listdir('./test' + str(args.cmap_threshold) + '/beta_strand/')
    test_dataset = []
    for test_ah_mat in test_alpha_helix_files:
        test_ah_cmap_mat = pd.read_csv('./test' + str(args.cmap_threshold) + '/alpha_helix/' + test_ah_mat, delim_whitespace=True, header=None)
        test_ah_cmap_pd = pd.DataFrame(test_ah_cmap_mat)
        test_ah_cmap_array = test_ah_cmap_pd.to_numpy()
        test_ah_cmap = torch.tensor(test_ah_cmap_array)
        test_ah_pad = nn.ZeroPad2d((140, 139, 140, 139)) # InceptionV3 - 299; VGG19 - 32.
        test_ah_output = test_ah_pad(test_ah_cmap)
        #print(test_ah_output)
        test_ah_output_3c = torch.stack((test_ah_output, test_ah_output, test_ah_output))
        test_ah_cmap_w_label = (test_ah_output_3c, 0)
        test_dataset.append(test_ah_cmap_w_label)
    for test_bs_mat in test_beta_strand_files:
        test_bs_cmap_mat = pd.read_csv('./test' + str(args.cmap_threshold) + '/beta_strand/' + test_bs_mat, delim_whitespace=True, header=None)
        test_bs_cmap_pd = pd.DataFrame(test_bs_cmap_mat)
        test_bs_cmap_array = test_bs_cmap_pd.to_numpy()
        test_bs_cmap = torch.tensor(test_bs_cmap_array)
        test_bs_pad = nn.ZeroPad2d((140, 139, 140, 139))
        test_bs_output = test_bs_pad(test_bs_cmap)
        #print(test_bs_output)
        test_bs_output_3c = torch.stack((test_bs_output, test_bs_output, test_bs_output))
        test_bs_cmap_w_label = (test_bs_output_3c, 1)
        test_dataset.append(test_bs_cmap_w_label)
    
    # Evaluate test set
    test_dl = DataLoader(test_dataset, batch_size=256)
    test_dl = DeviceDataLoader(test_dl, device)
    preds,labels = test_predict(model, test_dl)

    # plot confusion matrix 
    cm = confusion_matrix(labels, preds)
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8),cmap=plt.cm.Blues)
    plt.xticks(range(2), ['alpha_helix', 'beta_strand'], fontsize=16)
    plt.yticks(range(2), ['alpha_helix', 'beta_strand'], fontsize=16)
    plt.xlabel('Predicted Label',fontsize=18)
    plt.ylabel('True Label',fontsize=18)
    plt.show()

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
