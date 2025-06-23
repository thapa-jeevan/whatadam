import sys
import time

import numpy as np
import torch

import utils


def run(experiment, method, nepochs=50, lr=0.05, lamb=0.75, sbatch=128, lr_min=1e-6, lr_factor=2.0,
        lr_patience=5, clipgrad=10.0, smax=1000, seed=1993):
    tstart = time.time()

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        print('[CUDA unavailable]');
        sys.exit()

    # Args -- Experiment
    if experiment == 'mnist2':
        from dataloaders import mnist2 as dataloader
    elif experiment == 'pmnist':
        from dataloaders import pmnist as dataloader
    elif experiment == 'cifar':
        from dataloaders import cifar as dataloader
    elif experiment == 'mixture':
        from dataloaders import mixture as dataloader

    # Args -- Approach
    if method == 'hat':
        from approaches import hat as approach
    elif method == 'hat_adamw':
        from approaches import hat_adamw as approach
    elif method == 'hat_original':
        from approaches import hat_original as approach
    elif method == 'joint':
        from approaches import joint as approach

    # Args -- Network
    if experiment == 'mnist2' or experiment == 'pmnist':
        if 'hat' in method:
            from networks import mlp_hat as network
        else:
            from networks import mlp as network
    else:
        if method == 'lfl':
            from networks import alexnet_lfl as network
        elif 'hat' in method:
            from networks import alexnet_hat as network
        elif method == 'progressive':
            from networks import alexnet_progressive as network
        elif method == 'pathnet':
            from networks import alexnet_pathnet as network
        elif method == 'hat-test':
            from networks import alexnet_hat_test as network
        else:
            from networks import alexnet as network

    ########################################################################################################################

    # Load
    print('Load data...')
    data, taskcla, inputsize = dataloader.get(seed=seed)
    print('Input size =', inputsize, '\nTask info =', taskcla)

    # Inits
    print('Inits...')
    net = network.Net(inputsize, taskcla).cuda()
    utils.print_model_report(net)

    appr = approach.Appr(net, nepochs=nepochs, lr=lr, lamb=lamb, sbatch=sbatch, lr_min=lr_min,
                         lr_factor=lr_factor, lr_patience=lr_patience, clipgrad=clipgrad, smax=smax, args=None)
    print(appr.criterion)
    utils.print_optimizer_config(appr.optimizer)
    print('-' * 100)

    # Loop tasks
    acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    mask_percent_ls = []
    for t, ncla in taskcla:
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']))
        print('*' * 100)

        if method == 'joint':
            # Get data. We do not put it to GPU
            if t == 0:
                xtrain = data[t]['train']['x']
                ytrain = data[t]['train']['y']
                xvalid = data[t]['valid']['x']
                yvalid = data[t]['valid']['y']
                task_t = t * torch.ones(xtrain.size(0)).int()
                task_v = t * torch.ones(xvalid.size(0)).int()
                task = [task_t, task_v]
            else:
                xtrain = torch.cat((xtrain, data[t]['train']['x']))
                ytrain = torch.cat((ytrain, data[t]['train']['y']))
                xvalid = torch.cat((xvalid, data[t]['valid']['x']))
                yvalid = torch.cat((yvalid, data[t]['valid']['y']))
                task_t = torch.cat((task_t, t * torch.ones(data[t]['train']['y'].size(0)).int()))
                task_v = torch.cat((task_v, t * torch.ones(data[t]['valid']['y'].size(0)).int()))
                task = [task_t, task_v]
        else:
            # Get data
            xtrain = data[t]['train']['x'].cuda()
            ytrain = data[t]['train']['y'].cuda()
            xvalid = data[t]['valid']['x'].cuda()
            yvalid = data[t]['valid']['y'].cuda()
            task = t

        # Train
        mask_percent = appr.train(task, xtrain, ytrain, xvalid, yvalid)
        mask_percent_ls.append(mask_percent)
        print('-' * 100)

        # Test
        for u in range(t + 1):
            xtest = data[u]['test']['x'].cuda()
            ytest = data[u]['test']['y'].cuda()
            test_loss, test_acc = appr.eval(u, xtest, ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                          100 * test_acc))
            acc[t, u] = test_acc
            lss[t, u] = test_loss

        # Save

    # Done
    print('*' * 100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end='')
        for j in range(acc.shape[1]):
            print('{:5.1f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*' * 100)
    print("Mask percent ls: ", mask_percent_ls)
    print('Done!')

    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

    return acc
