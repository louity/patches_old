import argparse
import ast
import hashlib
import json
import numpy as np
import os
import time

import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from imagenet import Imagenet32
from utils import progress_bar, grab_patches, normalize_patches, grab_patches_2, update_batch_norms, update_classifiers, grab_patches_from_loader, correct_topk, preprocess, normalize_patches_2

print('simple_logistic.py')
parser = argparse.ArgumentParser('simple logistic regression with convolutional random features model')
# parameters for the patches
parser.add_argument('--dataset', help="cifar10/?", default='cifar10')
parser.add_argument('--no_padding', action='store_true', help='no padding used')
parser.add_argument('--patches_file', help=".t7 file containing patches", default='')
parser.add_argument('--correct_padding', action='store_true', help='use image instead of 0 padding')
parser.add_argument('--n_channel_convolution', default=256, type=int)
parser.add_argument('--spatialsize_convolution', default=6, type=int)
parser.add_argument('--padding_mode', default='constant', choices=['constant', 'reflect', 'symmetric'], help='type of padding for torch RandomCrop')
parser.add_argument('--learn_patches', action='store_true', help='learn the patches by SGD')
parser.add_argument('--patch_distribution', default='empirical', choices=['empirical', 'random'], help='distribution from which patches are drawn')
parser.add_argument('--channel_pca', type=int, default=0, help='perform PCA along channels')
parser.add_argument('--no_zca', action='store_true', help='no zca whitening performed on the patches')
parser.add_argument('--zca_bias', default=0.001, type=float, help='regularization bias for zca whitening')
parser.add_argument('--whiten_dataset', action='store_true', help='no zca whitening performed on the patches')

# parameters for the extraction
parser.add_argument('--stride_convolution', default=1, type=int)
parser.add_argument('--stride_avg_pooling', default=2, type=int)
parser.add_argument('--spatialsize_avg_pooling', default=5, type=int)
parser.add_argument('--bias', type=float, default=0.1)
parser.add_argument('--shrink', default='softshrink', choices=['softshrink', 'hardshrink', 'heaviside', 'topk', 'topk_heaviside', 'topk_heaviside_and_heaviside'], help='type of shrink function')
parser.add_argument('--topk_fraction', default=0.25, type=float)
parser.add_argument('--positive_shrink', action='store_true', help='use positive and negative part of shrink func')

# parameters of the classifier
parser.add_argument('--batch_norm', action='store_true', help='add batchnorm before classifier')
parser.add_argument('--bottleneck_dim', default=0, type=int, help='bottleneck dimension for the classifier')
parser.add_argument('--batch_norm_bottleneck', action='store_true', help='add batchnorm in bottleneck')
parser.add_argument('--separable_convolution', action='store_true', help='makes the classifier convolution separable in channels and space')
parser.add_argument('--n_bagging_patches', type=int, default=0, help='do model bagging in the patches dimension')
parser.add_argument('--convolutional_classifier', type=int, default=0, help='size of the convolution for convolutional classifier')
parser.add_argument('--convolutional_loss', action='store_true', help='use convolutional loss')
parser.add_argument('--lambda_1', default=0., type=float, help='group lasso penalty on the conv')
parser.add_argument('--multi_classif', action='store_true')

# parameters of the optimizer
parser.add_argument('--batchsize', type=int, default=512)
parser.add_argument('--batchsize_net', type=int, default=0)
parser.add_argument('--lr_schedule', type=str, default='{0:1e-3, 1:1e-4}')
parser.add_argument('--nepochs', type=int, default=90)
parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam')
parser.add_argument('--sgd_momentum', type=float, default=0.)
parser.add_argument('--weight_decay', type=float, default=0.)

# hardware parameters
parser.add_argument('--path_train', help="path to imagenet", default='/d1/dataset/imagenet32/out_data_train')
parser.add_argument('--path_test', help="path to imagenet", default='/d1/dataset/imagenet32/out_data_val')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--multigpu', action='store_true')
parser.add_argument('--no_cudnn', action='store_true', help='disable cuDNN to prevent cuDNN error (slower)')
parser.add_argument('--no_jit', action='store_true', help='disable torch.jit optimization to prevent error (slower)')

# reproducibility parameters
parser.add_argument('--force_recompute', action='store_true')
parser.add_argument('--numpy_seed', type=int, default=0)
parser.add_argument('--torch_seed', type=int, default=0)
parser.add_argument('--save_model', action='store_true', help='saves the model')
parser.add_argument('--save_best_model', action='store_true', help='saves the best model')
parser.add_argument('--resume', default='', help='filepath of checkpoint to load the model')

# logs
parser.add_argument('--no_progress_bar', action='store_true', help="Don't plot progress basr (for FI)")

args = parser.parse_args()

if args.multi_classif:
    assert args.batchsize_net > 0, 'batchsize_net must be positive for multiclass classification'

if args.batchsize_net > 0:
    assert args.n_channel_convolution // args.batchsize_net == args.n_channel_convolution / args.batchsize_net, 'batchsize_net must divide n_channel_convolution'

print('arguments')
print(args)




learning_rates = ast.literal_eval(args.lr_schedule)

# Extract the parameters
n_channel_convolution = args.n_channel_convolution
stride_convolution = args.stride_convolution
spatialsize_convolution = args.spatialsize_convolution
stride_avg_pooling = args.stride_avg_pooling
spatialsize_avg_pooling = args.spatialsize_avg_pooling
if torch.cuda.is_available():
    device = 'cuda'
    n_gpus = torch.cuda.device_count()
else:
    device = 'cpu'
print(f'device: {device}')
torch.manual_seed(args.torch_seed)
np.random.seed(args.numpy_seed)

train_sampler = None

# Define the dataset
if args.dataset == 'cifar10':
    spatial_size = 32
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
    n_classes=10
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)
elif args.dataset in ['imagenet32', 'imagenet64', 'imagenet128']:

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    n_arrays_train = 10
    padding = 4
    spatial_size = 32
    if args.dataset=='imagenet64':
        spatial_size = 64
        padding = 8
    if args.dataset=='imagenet128':
        spatial_size = 128
        padding = 16
        n_arrays_train = 99
    n_classes = 1000

    if args.no_padding:
        padding = 0

    if args.correct_padding:
        transforms_train = [
            transforms.Resize(spatial_size+2*padding),
            transforms.RandomCrop(spatial_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
        transforms_test = [
            transforms.Resize(spatial_size+2*padding),
            transforms.CenterCrop(spatial_size),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # print('pad + random resized crop scale 0.8 - 1 ')
        transforms_train = [
            transforms.RandomCrop(spatial_size, padding=padding, padding_mode=args.padding_mode),
            # transforms.Pad(padding=padding, padding_mode=args.padding_mode),
            # transforms.RandomResizedCrop(spatial_size, scale=(0.8, 1), ratio=(0.8,1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
        transforms_test = [transforms.ToTensor(), normalize]

    trainset = Imagenet32(args.path_train, transform=transforms.Compose(transforms_train), sz=spatial_size, n_arrays=n_arrays_train)
    testset = Imagenet32(args.path_test, transform=transforms.Compose(transforms_test), sz=spatial_size)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batchsize, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    n_classes = 1000


zca_str = 'nozca' if args.no_zca else f'zcabias{args.zca_bias}'
default_patches_file = f'metric_patches/{args.dataset}_seed{args.numpy_seed}_n{args.n_channel_convolution}_size{args.spatialsize_convolution}_{zca_str}_filter.t7'
patches_file = args.patches_file if args.patches_file else default_patches_file

if args.whiten_dataset:
    # Whitening of sets
    train_whiten, test_whiten = preprocess(trainset.data, testset.data)
    trainset.data = train_whiten
    testset.data = test_whiten

if args.patch_distribution == 'empirical':
    if not os.path.exists(patches_file) or args.force_recompute:
            t = None
            if args.dataset in ['cifar10', 'imagenet32', 'imagenet64', 'imagenet128']:
                if hasattr(trainset, 'train_data'):
                    t = trainset.train_data
                elif hasattr(trainset, 'data'):
                    t = trainset.data
                else:
                    raise RuntimeError
                print(f'Trainset : {t.shape}')
                patches, idx = grab_patches(t, seed=args.numpy_seed, patch_size=spatialsize_convolution)
                # patches, patch_classes = grab_patches_2(t, trainset.targets, 500000, patch_size=spatialsize_convolution, seed=args.numpy_seed)
                # imgset = Imagenet32(args.path_train, transform=transforms.Compose([transforms.ToTensor()]), sz=spatial_size)
                # imgloader = torch.utils.data.DataLoader(imgset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                # patches, patch_classes = grab_patches_from_loader(imgloader, n_patches=n_channel_convolution, image_size=spatial_size, patch_size=spatialsize_convolution, n_images=1281159, seed=0)
            elif args.dataset == '':
                imgset = datasets.ImageFolder(
                    args.path_train,
                    transforms.Compose([
                        transforms.Resize(spatial_size),
                        transforms.CenterCrop(spatial_size),
                        transforms.ToTensor(),
                ]))

                imgloader = torch.utils.data.DataLoader(
                    imgset, batch_size=args.batchsize, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, sampler=None)
                patches, patch_classes = grab_patches_from_loader(imgloader, n_patches=n_channel_convolution, image_size=spatial_size, patch_size=spatialsize_convolution, n_images=1281159, seed=0)
            print(f'patches extracted: {patches.shape}')
            # patches = normalize_patches(patches, zca_bias=args.zca_bias)
            patches, patches_2, E, V = normalize_patches_2(patches, zca_bias=args.zca_bias)
            print(f'patches normalized: {patches.shape}')
            idxs = np.random.choice(patches.shape[0], n_channel_convolution, replace=False)
            selected_patches = patches[idxs].astype(np.float32)
            selected_patches_2 = patches_2[idxs].astype(np.float32)
            print(f'patches randomly selected: {selected_patches.shape}')
            kernel_convolution = torch.from_numpy(selected_patches)
            kernel_convolution_2, ev_rs, V = torch.from_numpy(selected_patches_2), torch.from_numpy(1. / E).view(1, -1), torch.from_numpy(V)
            ev_rescale = nn.Parameter(ev_rs.cuda(), requires_grad=True)
            params = [ev_rescale]
            print(f'saving patches in file {patches_file}')
            torch.save(kernel_convolution, patches_file)
    else:
        kernel_convolution = torch.load(patches_file)
elif args.patch_distribution == 'random':
    kernel_convolution = nn.Conv2d(3, n_channel_convolution,
                             spatialsize_convolution, stride=stride_convolution, bias=False).weight.data
    kernel_convolution = kernel_convolution.view(kernel_convolution.size(0), -1)
    kernel_convolution /= kernel_convolution.norm(p=2, dim=1, keepdim=True)
    kernel_convolution = kernel_convolution.view(kernel_convolution.size(0), 3, args.spatialsize_convolution, args.spatialsize_convolution).contiguous()



def compute_classifier_outputs(outputs1, outputs2, targets, args, batch_norm1, batch_norm2, classifier1, classifier2, classifier, dropout_mask, batch_norm_bottleneck, train=True):
    if args.batch_norm:
        outputs1, outputs2 = batch_norm1(outputs1), batch_norm2(outputs2)


    if args.convolutional_classifier == 0:
        outputs1, outputs2 = outputs1.view(outputs1.size(0),-1), outputs2.view(outputs2.size(0),-1)

    outputs1, outputs2 = classifier1(outputs1), classifier2(outputs2)
    outputs = outputs1 + outputs2

    if args.batch_norm_bottleneck:
        outputs = batch_norm_bottleneck(outputs)

    if args.convolutional_classifier > 0:
        if args.convolutional_loss and train:
            b_size, nc1, h, w = outputs.size()
            outputs = outputs.view(b_size, nc1, -1).transpose(1,2).reshape(b_size*h*w, nc1)
            targets = targets.view(b_size, 1).expand(b_size, h*w).reshape(b_size*h*w)
        elif args.separable_convolution or args.bottleneck_dim > 0:
            outputs = classifier(outputs)
            outputs = F.adaptive_avg_pool2d(outputs, 1)
        else:
            outputs = F.adaptive_avg_pool2d(outputs, 1)
    elif args.bottleneck_dim > 0:
        outputs = classifier(outputs)

    outputs = outputs.view(outputs.size(0),-1)

    return outputs, targets

def create_classifier_blocks(out1, out2, args, params, n_classes):
    batch_norm1, batch_norm2, classifier1, classifier2, classifier, dropout_mask, batch_norm_bottleneck =  None, None, None, None, None, None, None

    if args.batch_norm:
        batch_norm1 = nn.BatchNorm2d(out1.size(1)).to(device).float()
        batch_norm2 = nn.BatchNorm2d(out2.size(1)).to(device).float()
        params += list(batch_norm1.parameters()) + list(batch_norm2.parameters())

    if args.batch_norm_bottleneck:
        batch_norm_bottleneck = nn.BatchNorm2d(args.bottleneck_dim).to(device).float()
        params += list(batch_norm_bottleneck.parameters())


    if args.convolutional_classifier > 0:
        if args.separable_convolution:
            # convolution separable in space channels
            if args.bottleneck_dim > 0:
                classifier = nn.Sequential(
                    nn.Conv2d(args.bottleneck_dim, n_classes, 1).to(device).float(),
                    nn.Conv2d(n_classes, n_classes, args.convolutional_classifier, groups=n_classes).to(device).float()
                )
                classifier1 = nn.Conv2d(out1.size(1), args.bottleneck_dim, 1).to(device).float()
                classifier2 = nn.Conv2d(out2.size(1), args.bottleneck_dim, 1).to(device).float()
            else:
                classifier = nn.Conv2d(n_classes, n_classes, args.convolutional_classifier, groups=n_classes).to(device).float()
                classifier1 = nn.Conv2d(out1.size(1), n_classes, 1).to(device).float()
                classifier2 = nn.Conv2d(out2.size(1), n_classes, 1).to(device).float()
            params += list(classifier.parameters())
        else:
            if args.bottleneck_dim > 0:
                # usual setting
                # classifier = nn.Linear(args.bottleneck_dim, n_classes).to(device).float()
                # params += list(classifier.parameters())
                # classifier1 = nn.Conv2d(out1.size(1), args.bottleneck_dim, args.convolutional_classifier).to(device).float()
                # classifier2 = nn.Conv2d(out2.size(1), args.bottleneck_dim, args.convolutional_classifier).to(device).float()
                # less params
                classifier = nn.Conv2d(args.bottleneck_dim, n_classes, args.convolutional_classifier).to(device).float()
                params += list(classifier.parameters())
                # classifier1 = nn.Conv2d(out1.size(1), args.bottleneck_dim, 1).to(device).float()
                # classifier2 = nn.Conv2d(out2.size(1), args.bottleneck_dim, 1).to(device).float()
                print('no bias in classifiers 1 and 2')
                classifier1 = nn.Conv2d(out1.size(1), args.bottleneck_dim, 1, bias=False).to(device).float()
                classifier2 = nn.Conv2d(out2.size(1), args.bottleneck_dim, 1, bias=False).to(device).float()
            else:
                classifier1 = nn.Conv2d(out1.size(1), n_classes, args.convolutional_classifier).to(device).float()
                classifier2 = nn.Conv2d(out2.size(1), n_classes, args.convolutional_classifier).to(device).float()
    else:
        out1, out2 = out1.view(out1.size(0), -1), out2.view(out1.size(0), -1)
        if args.bottleneck_dim > 0:
            classifier = nn.Linear(args.bottleneck_dim, n_classes).to(device).float()
            params += list(classifier.parameters())
            classifier1 = nn.Linear(out1.size(1), args.bottleneck_dim).to(device).float()
            classifier2 = nn.Linear(out2.size(1), args.bottleneck_dim).to(device).float()
        else:
            classifier1 = nn.Linear(out1.size(1), n_classes).to(device).float()
            classifier2 = nn.Linear(out2.size(1), n_classes).to(device).float()

    params += list(classifier1.parameters()) + list(classifier2.parameters())

    return batch_norm1, batch_norm2, classifier1, classifier2, classifier, dropout_mask, batch_norm_bottleneck

def heaviside_half(x, bias):
    return (x > bias).half() - (x < -bias).half()

def heaviside_float(x, bias):
    return (x > bias).float() - (x < -bias).float()

def topk_half(x, k):
    x_abs = torch.abs(x)
    return (x_abs >= x_abs.topk(dim=1, k=k).values.min(dim=1, keepdim=True).values).half() * x.half()

def topk_float(x, k):
    x_abs = torch.abs(x)
    return (x_abs >= x_abs.topk(dim=1, k=k).values.min(dim=1, keepdim=True).values).float() * x.float()

def topk_heaviside_half(x, k):
    x_abs = torch.abs(x)
    return (x_abs >= x_abs.topk(dim=1, k=k).values.min(dim=1, keepdim=True).values).half() * x.sign().half()

def topk_heaviside_float(x, k):
    x_abs = torch.abs(x)
    return (x_abs >= x_abs.topk(dim=1, k=k).values.min(dim=1, keepdim=True).values).float() * x.sign().float()

def topk_heaviside_and_heaviside_half(x, k, bias=args.bias):
    x_abs = torch.abs(x)
    return (x_abs > bias).half() * (x_abs >= x_abs.topk(dim=1, k=k).values.min(dim=1, keepdim=True).values).half() * x.sign().half()


if args.shrink == 'heaviside':
    if args.learn_patches:
        shrink = heaviside_float
    else:
        shrink = heaviside_half
elif args.shrink == 'topk':
    args.bias = int(args.topk_fraction * n_channel_convolution)
    if args.learn_patches:
        shrink = topk_float
    else:
        shrink = topk_half
elif args.shrink == 'topk_heaviside':
    args.bias = int(args.topk_fraction * n_channel_convolution)
    if args.learn_patches:
        shrink = topk_heaviside_float
    else:
        shrink = topk_heaviside_half
elif args.shrink == 'topk_heaviside_and_heaviside':
    args.bias = int(args.topk_fraction * n_channel_convolution)
    shrink = topk_heaviside_and_heaviside_half
elif args.shrink == 'hardshrink':
    shrink = F.hardshrink
elif args.shrink == 'softshrink':
    # shrink = F.softshrink
    shrink = lambda x, lambd: F.relu(x - lambd) - F.relu(-x - lambd)




# Define the model
class Net(nn.Module):
    def __init__(self, spatialsize_avg_pooling, stride_avg_pooling, bias=1.0):
        super(Net, self).__init__()
        self.pool_size = spatialsize_avg_pooling
        self.pool_stride = stride_avg_pooling
        self.bias = bias

    def forward(self, x, conv_weight):
        out = F.conv2d(x, conv_weight)
        out = shrink(out, self.bias)
        out1 = F.avg_pool2d(out, [self.pool_size, self.pool_size], stride=[self.pool_stride, self.pool_stride],
                            ceil_mode=True)
        out = F.relu(out)
        out2 = F.avg_pool2d(out, [self.pool_size, self.pool_size], stride=[self.pool_stride, self.pool_stride],
                            ceil_mode=True)
        return out1, out2

criterion = nn.CrossEntropyLoss()

net = Net(spatialsize_avg_pooling, stride_avg_pooling, bias=args.bias).to(device)

kernel_convolution = kernel_convolution.to(device)
kernel_convolution_2, V = kernel_convolution_2.to(device), V.to(device)


x = torch.rand(1, 3, spatial_size, spatial_size).to(device)

# if torch.cuda.is_available() and not args.learn_patches:
    # kernel_convolution = kernel_convolution.half()
    # x = x.half()


if args.no_cudnn:
    torch.backends.cudnn.enabled = False
else:
    cudnn.benchmark = True


if args.multi_classif:
    kernel_convolution_ = kernel_convolution[:args.batchsize_net]
else:
    kernel_convolution_ = kernel_convolution

out1, out2 = net(x, kernel_convolution_)
out1, out2 = out1.float(), out2.float()
print(f'Net output size: out1 {out1.shape[-3:]} out2 {out2.shape[-3:]}')

if args.multi_classif:
    classifier_blocks = [create_classifier_blocks(out1, out2, args, params, n_classes) for i in range(n_channel_convolution // args.batchsize_net)]
else:
    classifier_blocks = create_classifier_blocks(out1, out2, args, params, n_classes)



print(f'Parameters shape {[param.shape for param in params]}')
print(f'N parameters : {sum([np.prod(list(param.shape)) for param in params])/1e6} millions')


del x, out1, out2


if args.batchsize_net > 0:
    kernel_convolution = [kernel_convolution[i*args.batchsize_net:(i+1)*args.batchsize_net] for i in range(args.n_channel_convolution // args.batchsize_net)]
else:
    kernel_convolution = [kernel_convolution]

if torch.cuda.is_available() and not args.learn_patches and not args.no_jit:
    print('optimizing net execution with torch.jit')
    trial = torch.rand(args.batchsize//n_gpus, 3, spatial_size, spatial_size).to(device)

    inputs = {'forward': (trial, kernel_convolution[0])}
    with torch.jit.optimized_execution(True):
        net = torch.jit.trace_module(net, inputs, check_trace=False, check_tolerance=False)
    del inputs
    del trial


if args.multigpu and n_gpus > 1:
    print(f'{n_gpus} available, using Dataparralel for net')
    net = nn.DataParallel(net)


def train(epoch):
    net.train()
    if not args.multi_classif:
        batch_norm1, batch_norm2, classifier1, classifier2, classifier, dropout_mask, batch_norm_bottleneck = classifier_blocks
        if args.batch_norm:
            batch_norm1.train()
            batch_norm2.train()
        if args.batch_norm_bottleneck:
            batch_norm_bottleneck.train()

    train_loss = 0
    total=0
    correct=0


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # if torch.cuda.is_available():
            # inputs = inputs.half()

        # patches_shape = selected_patches_2.shape
        # operator = (V * (1 / E).reshape((1, -1))).dot(V.T)
        # zca_patches = (selected_patches_2.reshape((patches_shape[0], -1))).dot(operator)
        # zca_patches /= np.linalg.norm(zca_patches, axis=1, keepdims=True) + 1e-8
        # zca_patches = zca_patches.reshape(patches_shape).astype('float32')
        # kernel_conv = torch.from_numpy(zca_patches).half().cuda()


        optimizer.zero_grad()

        patches_shape = kernel_convolution_2.shape
        operator = (V * ev_rescale) @ V.t()
        zca_patches = kernel_convolution_2.view(patches_shape[0], -1) @ operator
        zca_patches_normalized = zca_patches / (zca_patches.norm(dim=1, keepdim=True) + 1e-8)
        kernel_conv = zca_patches_normalized.view(patches_shape)
        outputs1, outputs2 = net(inputs, kernel_conv)

        # outputs1, outputs2 = net(inputs, kernel_convolution[0])


        outputs, targets = compute_classifier_outputs(outputs1, outputs2, targets, args, batch_norm1, batch_norm2, classifier1, classifier2, classifier, dropout_mask, batch_norm_bottleneck, train=True)


        loss = criterion(outputs, targets)
        if args.lambda_1 > 0.:
            group_sparsity_norm = torch.norm(torch.cat([classifier1.weight, classifier2.weight], dim=0), dim=0, p=2).mean()
            loss += args.lambda_1 * group_sparsity_norm
        loss.backward()

        optimizer.step()

        if torch.isnan(loss):
            return False
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # if args.learn_patches:
            # with torch.no_grad():
                # kernel_convolution[0].data /= kernel_convolution[0].data.view(args.n_channel_convolution, -1).norm(p=2, dim=1).view(args.n_channel_convolution, 1, 1, 1)

        progress_bar(batch_idx, len(trainloader), 'Train, epoch: %i; Loss: %.3f | Acc: %.3f%% (%d/%d) ; threshold %.3f' % (
            epoch, train_loss / (batch_idx + 1), 100. * correct / total, correct, total, args.bias), hide=args.no_progress_bar)

    if args.no_progress_bar:
        print('Train, epoch: {}; Loss: {:.2f} | Acc: {:.1f} ; threshold {:.3f}'.format(
            epoch, train_loss / (batch_idx + 1), 100. * correct / total, args.bias))
    return True

def test(epoch, loader=testloader, msg='Test'):
    global best_acc
    net.eval()
    if not args.multi_classif:
        batch_norm1, batch_norm2, classifier1, classifier2, classifier, dropout_mask, batch_norm_bottleneck = classifier_blocks
        if args.batch_norm:
            batch_norm1.eval()
            batch_norm2.eval()
        if args.batch_norm_bottleneck:
            batch_norm_bottleneck.eval()

    test_loss = 0
    correct_top1, correct_top5 = 0, 0
    total = 0
    outputs_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            patches_shape = kernel_convolution_2.shape
            operator = (V * ev_rescale) @ V.t()
            zca_patches = kernel_convolution_2.view(patches_shape[0], -1) @ operator
            zca_patches /= zca_patches.norm(dim=1, keepdim=True) + 1e-8
            zca_patches = zca_patches.view(patches_shape).contiguous()
            kernel_conv = zca_patches

            # outputs1, outputs2 = net(inputs, kernel_convolution[0])
            outputs1, outputs2 = net(inputs, kernel_conv)

            outputs, targets = compute_classifier_outputs(outputs1, outputs2, targets, args, batch_norm1, batch_norm2, classifier1, classifier2, classifier, dropout_mask, batch_norm_bottleneck, train=False)

            loss = criterion(outputs, targets)

            outputs_list.append(outputs)

            test_loss += loss.item()
            cor_top1, cor_top5 = correct_topk(outputs, targets, topk=(1, 5))
            correct_top1 += cor_top1
            correct_top5 += cor_top5
            _, predicted = outputs.max(1)
            total += targets.size(0)
            progress_bar(batch_idx, len(loader),
                         'Test, epoch: %i; Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, test_loss / (batch_idx + 1),
                                                            100. * correct_top1 / total, correct_top1, total), hide=args.no_progress_bar)
        test_loss /= (batch_idx + 1)
        acc1, acc5 = 100. * correct_top1 / total, 100. * correct_top5 / total
        if args.no_progress_bar:
            print(f'{msg}, epoch: {epoch}; Loss: {test_loss:.2f} | Acc: {acc1:.1f} @1 {acc5:.1f} @5 ; threshold {args.bias:.3f}')
        outputs = torch.cat(outputs_list, dim=0).cpu()
        if args.lambda_1 > 0.:
            group_sparsity_norm = torch.norm(torch.cat([classifier1.weight, classifier2.weight], dim=0), dim=0, p=2)
            print(f'non_zero groups {(group_sparsity_norm != 0).int().sum()}')

        return acc1, outputs

hashname = hashlib.md5(str.encode(json.dumps(vars(args), sort_keys=True))).hexdigest()
if args.save_model:
    checkpoint_dir = f'checkpoints/{args.dataset}_{args.n_channel_convolution}patches_{args.spatialsize_convolution}x{args.spatialsize_convolution}/{args.optimizer}_{args.lr_schedule}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_file = os.path.join(checkpoint_dir, f'{hashname}.pth.tar')
    print(f'Model will be saved at file {checkpoint_file}.')

    state = {'args': args}
    if os.path.exists(checkpoint_file):
        state = torch.load(checkpoint_file)

start_epoch = 0
if args.resume:
    state = torch.load(args.resume)
    start_epoch = state['epoch'] + 1
    print(f'Resuming from file {args.resume}, start epoch {start_epoch}...')
    if start_epoch not in learning_rates:
        closest_i = max([i for i in learning_rates if i <= start_epoch])
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=learning_rates[closest_i], weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=learning_rates[closest_i], momentum=args.sgd_momentum, weight_decay=args.weight_decay)
        optimizer.load_state_dict(state['optimizer'])

    classifier1.load_state_dict(state['classifier1'])
    classifier2.load_state_dict(state['classifier2'])
    if args.bottleneck_dim > 0:
        classifier.load_state_dict(state['classifier'])
    if args.batch_norm:
        batch_norm1.load_state_dict(state['batch_norm1'])
        batch_norm2.load_state_dict(state['batch_norm2'])
    acc, outputs = test(-1)

start_time = time.time()
best_test_acc, best_epoch = 0, -1
with np.printoptions(precision=4, suppress=True):
    print(f'ev rescale initial {ev_rescale.data.cpu.numpy()}')

for i in range(start_epoch, args.nepochs):
    if i in learning_rates:
        print('new lr:'+str(learning_rates[i]))
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=learning_rates[i], weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=learning_rates[i], momentum=args.sgd_momentum, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError('optimizer {} not implemented'.format(args.optimizer))
    no_nan_in_train_loss = train(i)
    if not no_nan_in_train_loss:
        print(f'Epoch {i}, nan in loss, stopping training')
        break
    test_acc, outputs = test(i)
    print(f'ev rescale max {ev_rescale.data.max().item():.4f}, min {ev_rescale.data.min().item():.4f}')

    if test_acc > best_test_acc:
        print(f'Best acc ({test_acc}).')
        best_test_acc = test_acc
        best_epoch = i

    if args.save_model or args.save_best_model and best_epoch == i:
        print(f'saving...')
        state.update({
            'optimizer': optimizer.state_dict(),
            'classifier1': classifier1.state_dict(),
            'classifier2': classifier2.state_dict(),
            'epoch': i,
            'acc': test_acc,
            'outputs': outputs,
        })
        if args.bottleneck_dim > 0:
            state.update({
                'classifier': classifier.state_dict(),
            })
        if args.batch_norm:
            state.update({
                f'batch_norm1': batch_norm1.state_dict(),
                f'batch_norm2': batch_norm2.state_dict(),
            })
        torch.save(state, checkpoint_file)

with np.printoptions(precision=4, suppress=True):
    print(f'ev rescale final {ev_rescale.data.cpu.numpy()}')
print(f'Best test acc. {best_test_acc}  at epoch {best_epoch}/{i}')
hours = (time.time() - start_time) / 3600
print(f'Done in {hours:.1f} hours with {n_gpus} GPU')
