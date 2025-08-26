from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from tqdm import tqdm
import pickle

from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer
from util.misc import AverageMeter, accuracy, update_json
from models.aconvnets import Network
from models.cnn_encoder import ParallelChannelCNN
from mcr2 import MaximalCodingRateReduction
from dataset.data_loader import PolarimetricSARDataset, SARDatasetFromFeatures, create_sar_dataloaders, create_feature_dataloaders, get_sar_transforms


def train_model(model, dataloaders, num_epochs=10, lr=0.001, criterion=None, device='cuda'):
    """Standalone training function for external use"""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(dataloaders, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Check if criterion is MCR² loss by checking its class name
            if hasattr(criterion, '__class__') and 'MaximalCodingRateReduction' in str(criterion.__class__):
                # For MCR² loss, we need features and labels
                loss, _, _ = criterion(outputs, labels)
            else:
                # For CrossEntropy loss
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Calculate accuracy after each batch
            batch_accuracy = 100. * correct / total
            loop.set_postfix(loss=loss.item(), acc=batch_accuracy)

        epoch_loss = running_loss / len(dataloaders)
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser('argument for SAR target recognition training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='train_sar',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    
    # MCR² specific argument
    parser.add_argument('--mcr2', action='store_true',
                        help='use MCR² loss and ParallelChannelCNN model for feature extraction')
    parser.add_argument('--features_dir', type=str, default='./features',
                        help='directory to store extracted features')
    parser.add_argument('--extract_features', action='store_true',
                        help='extract features using trained ParallelChannelCNN model')
    parser.add_argument('--train_on_features', action='store_true',
                        help='train Network model on extracted features')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', default=True,
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')

    # dataset
    parser.add_argument('--dataset', type=str, default='sar')
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--n_cls', type=int, default=7,
                        help='number of classes for classification')

    # model parameters
    parser.add_argument('--model', type=str, default='network',
                        help='model selection: network or parallelcnn')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained model')

    args = parser.parse_args()
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    
    args.model_name = '{}_{}'.format(args.dataset, args.model)
    if args.mcr2:
        args.model_name += '_mcr2'
    if args.tag:
        args.model_name += '_{}'.format(args.tag)

    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    
    # Create features directory
    if not os.path.isdir(args.features_dir):
        os.makedirs(args.features_dir)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    return args


def set_loader(args):
    """Set up data loaders for SAR data"""
    
    # Check if we're using real SAR data or dummy data
    if os.path.exists(args.data_folder) and len(os.listdir(args.data_folder)) > 0:
        # Use real SAR data
        print(f"Loading SAR data from: {args.data_folder}")
        
        # Create transforms for SAR data
        train_transform = get_sar_transforms(
            input_size=(224, 224),  # Adjust size as needed
            normalize=True,
            augment=True  # Apply augmentation for training
        )
        
        val_transform = get_sar_transforms(
            input_size=(224, 224),
            normalize=True,
            augment=False  # No augmentation for validation
        )
        
        # Create datasets
        train_dataset = PolarimetricSARDataset(
            root_dir=args.data_folder,
            transform=train_transform,
            train=True,
            train_split=0.8,
            max_samples_per_class=None,  # Use all samples
            random_seed=args.seed
        )
        
        val_dataset = PolarimetricSARDataset(
            root_dir=args.data_folder,
            transform=val_transform,
            train=False,
            train_split=0.8,
            max_samples_per_class=None,
            random_seed=args.seed
        )
        
        # Update number of classes based on actual data
        args.n_cls = len(train_dataset.target_classes)
        print(f"Number of classes detected: {args.n_cls}")
        
 

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader, args


def set_model(args, stage='mcr2'):    
    if stage == 'mcr2' or args.mcr2:
        # Use ParallelChannelCNN with MCR² loss for feature extraction
        model = ParallelChannelCNN(num_classes=args.n_cls)
        criterion = MaximalCodingRateReduction(gam1=1.0, gam2=1.0, eps=0.01)
        print("Using ParallelChannelCNN with MCR² loss for feature extraction")
    else:
        # Use Network with CrossEntropy loss for classification on features
        model = Network(classes=args.n_cls, channels=4, dropout_rate=0.5)
        criterion = nn.CrossEntropyLoss()
        print("Using Network with CrossEntropy loss for classification")

    # Load pretrained checkpoint if specified
    if args.pretrained and args.pretrained_ckpt is not None:
        if os.path.isfile(args.pretrained_ckpt):
            print("=> loading checkpoint '{}'".format(args.pretrained_ckpt))
            checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            print("=> loaded checkpoint '{}'".format(args.pretrained_ckpt))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained_ckpt))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.cuda()
    if not args.mcr2:
        criterion.cuda()
    
    optimizer = set_optimizer(args, model.parameters())

    return model, criterion, optimizer


def extract_features(model, dataloader, args, split='train'):
    """Extract features from all images using the trained ParallelChannelCNN model"""
    model.eval()
    features_list = []
    labels_list = []
    
    print(f"Extracting features for {split} split...")
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Extracting {split} features")):
            images = images.cuda(non_blocking=True)
            
            # Get features from the model (before the final classification layer)
            # For ParallelChannelCNN, we need to get features before the final fc3 layer
            outputs = model(images)
            
            # Store features and labels
            features_list.append(outputs.cpu())
            labels_list.append(labels)
    
    # Concatenate all features and labels
    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    # Save features to disk
    features_file = os.path.join(args.features_dir, f'{split}_features.pkl')
    with open(features_file, 'wb') as f:
        pickle.dump({
            'features': all_features.numpy(),
            'labels': all_labels.numpy()
        }, f)
    
    print(f"Saved {split} features to {features_file}")
    print(f"Features shape: {all_features.shape}, Labels shape: {all_labels.shape}")
    
    return all_features, all_labels





def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            
            if args.mcr2:
                # MCR² loss
                loss, _, _ = criterion(outputs, labels)
            else:
                # CrossEntropy loss
                loss = criterion(outputs, labels)

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(outputs, labels, topk=(1,))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            with torch.cuda.amp.autocast():
                outputs = model(images)
                
                if args.mcr2:
                    # MCR² loss
                    loss, _, _ = criterion(outputs, labels)
                else:
                    # CrossEntropy loss
                    loss = criterion(outputs, labels)

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(outputs, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    
    if top1.avg > best_acc:
        save_bool = True
        best_acc = top1.avg
        best_model = deepcopy(model.state_dict())

    print(' * Acc@1 {top1.avg:.2f} (Best: {best_acc:.2f})'.format(top1=top1, best_acc=best_acc))

    return best_acc, best_model, save_bool


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    if args.mcr2:
        # Stage 1: Train ParallelChannelCNN with MCR² loss and extract features
        print("=" * 50)
        print("STAGE 1: Training ParallelChannelCNN with MCR² loss")
        print("=" * 50)
        
        train_loader, val_loader, args = set_loader(args)
        model, criterion, optimizer = set_model(args, stage='mcr2')
        
        best_model = None
        best_acc = 0.0

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch'] 
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch += 1
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        else:
            args.start_epoch = 1
        
        # Train the ParallelChannelCNN model
        print('Training ParallelChannelCNN for {} epochs'.format(args.epochs))
        
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model, criterion, optimizer, epoch, args)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2-time1, acc))
            
            # eval for one epoch
            best_acc, best_model, save_bool = validate(val_loader, model, criterion, args, best_acc, best_model)
            
            # save a checkpoint when the best accuracy is updated
            if save_bool:            
                save_file = os.path.join(args.save_folder, 'best_parallelcnn_epoch_{}.pth'.format(epoch))
                print('Best ParallelChannelCNN ckpt is modified with Acc = {:.2f} when Epoch = {}'.format(best_acc, epoch))
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, save_file)
                
            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'parallelcnn_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, save_file)

        # save final ParallelChannelCNN model
        save_file = os.path.join(args.save_folder, 'parallelcnn_final.pth')
        model.load_state_dict(best_model)
        torch.save({
            'epoch': args.epochs,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
        }, save_file)
        
        # Extract features using the trained ParallelChannelCNN model
        print("\n" + "=" * 50)
        print("Extracting features using trained ParallelChannelCNN model")
        print("=" * 50)
        
        # Load the best model for feature extraction
        model.load_state_dict(best_model)
        
        # Extract features from train and validation sets
        train_features, train_labels = extract_features(model, train_loader, args, split='train')
        val_features, val_labels = extract_features(model, val_loader, args, split='val')
        
        print("Feature extraction completed!")
        
    else:
        # Stage 2: Train Network model on extracted features
        print("=" * 50)
        print("STAGE 2: Training Network model on extracted features")
        print("=" * 50)
        
        # Load extracted features
        train_features_file = os.path.join(args.features_dir, 'train_features.pkl')
        val_features_file = os.path.join(args.features_dir, 'val_features.pkl')
        
        if not os.path.exists(train_features_file) or not os.path.exists(val_features_file):
            print("Error: Feature files not found. Please run with --mcr2 first to extract features.")
            return
        
        # Load features using the proper dataset class
        train_dataset = SARDatasetFromFeatures(train_features_file)
        val_dataset = SARDatasetFromFeatures(val_features_file)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=True
        )
        
        # Set up Network model for classification on features
        model, criterion, optimizer = set_model(args, stage='network')
        
        best_model = None
        best_acc = 0.0
        
        # Train the Network model on features
        print('Training Network on features for {} epochs'.format(args.epochs))
        
        for epoch in range(1, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model, criterion, optimizer, epoch, args)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2-time1, acc))
            
            # eval for one epoch
            best_acc, best_model, save_bool = validate(val_loader, model, criterion, args, best_acc, best_model)
            
            # save a checkpoint when the best accuracy is updated
            if save_bool:            
                save_file = os.path.join(args.save_folder, 'best_network_epoch_{}.pth'.format(epoch))
                print('Best Network ckpt is modified with Acc = {:.2f} when Epoch = {}'.format(best_acc, epoch))
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, save_file)
                
            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'network_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, save_file)

        # save final Network model
        save_file = os.path.join(args.save_folder, 'network_final.pth')
        model.load_state_dict(best_model)
        torch.save({
            'epoch': args.epochs,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
        }, save_file)
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()