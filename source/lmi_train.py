from __future__ import print_function

import argparse
import os
import sys
import random
import math
import shutil
import numbers

import torch
torch.set_default_tensor_type(torch.FloatTensor)
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

from source.lmi_model import LMIModel
from source import data_loader
from source import sdf_nn
from source.base import evaluation

debug = False


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='debug',
                        help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.',
                        help='description')
    parser.add_argument('--indir', type=str, default='datasets/ABC_varnoise',
                        help='input folder (meshes)')
    parser.add_argument('--outdir', type=str, default='models',
                        help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainset.txt',
                        help='training set file name')
    parser.add_argument('--testset', type=str, default='testset.txt',
                        help='test set file name')
    parser.add_argument('--save_interval', type=int, default='10',
                        help='save model each n epochs')
    parser.add_argument('--debug_interval', type=int, default='1',
                        help='print logging info each n epochs')
    parser.add_argument('--refine', type=str, default='',
                        help='refine model at this path')
    parser.add_argument('--gpu_idx', type=int, default=0,
                        help='set < 0 to use CPU')
    parser.add_argument('--patch_radius', type=float, default=0.00,
                        help='Neighborhood of points that is queried with the network. '
                             'This enables you to set the trade-off between computation time and tolerance for '
                             'sparsely sampled surfaces. Use r <= 0.0 for k-NN queries.')

    # training parameters
    parser.add_argument('--net_size', type=int, default=1024,
                        help='number of neurons in the largest fully connected layer')
    parser.add_argument('--nepoch', type=int, default=2,
                        help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=2,
                        help='input batch size')
    parser.add_argument('--grid_resolution', type=int, default=256,
                        help='resolution of sampled volume')
    parser.add_argument('--patches_per_shape', type=int, default=1000,
                        help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--sub_sample_size', type=int, default=500,
                        help='number of points of the point cloud that are trained with each patch')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473,
                        help='manual seed')
    parser.add_argument('--training_order', type=str, default='random',
                        help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape \n'
                        'remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False,
                        help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--scheduler_steps', type=int, nargs='+', default=[75],
                        help='the lr will be multiplicated with 0.1 at these epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='gradient descent momentum')
    parser.add_argument('--k', type=int, default=10,
                        help='The number of points used to calculate neighbor features')
    parser.add_argument('--w', type=int, default=4,
                       help='width coefficient for the modified indicator functions.')

    parser.add_argument('--points_per_patch', type=int, default=200,
                        help='max. number of points per patch')
    parser.add_argument('--debug', type=int, default=0,
                        help='set to 1 of you want debug outputs to validate the model')

    return parser.parse_args(args=args)


def do_logging(log_prefix, epoch, opt, loss, batchind, num_batch):

    loss_cpu = [l.detach().cpu().item() for l in loss]
    if batchind % opt.debug_interval == 0:
        state_string = \
            '[{name} {epoch}: {batch}/{n_batches}] {prefix} loss: {dis_loss:+.6f}'.format(
                name=opt.name, epoch=epoch, batch=batchind, n_batches=num_batch - 1,
                prefix=log_prefix, dis_loss=loss_cpu[-1])
        print(state_string)
        


def lmi_train(opt):

    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)
    print('Training on {} GPUs'.format(torch.cuda.device_count()))
    print('Training on ' + ('cpu' if opt.gpu_idx < 0 else torch.cuda.get_device_name(opt.gpu_idx)))

    # colored console output, works e.g. on Ubuntu (WSL)
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % opt.name)
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % opt.name)

    if os.path.exists(log_dirname):
        '''
        if opt.name != 'test':
            response = input('A training run named "{}" already exists, overwrite? (y/n) '.format(opt.name))
            if response == 'y':
                del_log = True
            else:
                return
        else:
            del_log = True
        '''
        del_log = True
        if del_log:
            if os.path.exists(log_dirname):
                try:
                    shutil.rmtree(log_dirname)
                except OSError:
                    print("Can't delete " + log_dirname)

    lmi_model = LMIModel(
        net_size_max=opt.net_size,
        num_points=opt.points_per_patch,
        sub_sample_size=opt.sub_sample_size,
        k=opt.k
    )

    start_epoch = 0
    if opt.refine != '':
        print(f'Refining weights from {opt.refine}')
        lmi_model.cuda(device=device)  # same order as in training
        lmi_model = torch.nn.DataParallel(lmi_model)
        lmi_model.load_state_dict(torch.load(opt.refine))
        try:
            # expecting a file name like 'vanilla_model_50.pth'
            model_file = str(opt.refine)
            last_underscore_pos = model_file.rfind('_')
            last_dot_pos = model_file.rfind('.')
            start_epoch = int(model_file[last_underscore_pos+1:last_dot_pos]) + 1
            print(f'Continuing training from epoch {start_epoch}')
        except:
            print(f'Warning: {opt.refine} has no epoch in the name. The Tensorboard log will continue at '
                  f'epoch 0 and might be messed up!')

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # create train and test dataset loaders
    train_dataset = data_loader.PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.trainset,
        points_per_patch=opt.points_per_patch,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        cache_capacity=opt.cache_capacity,
        sub_sample_size=opt.sub_sample_size,
        recon_flag=False,
        num_workers=int(opt.workers),
        patch_radius=opt.patch_radius,
        epsilon=-1,  # not necessary for training
        k=opt.k
    )
    if opt.training_order == 'random':
        train_datasampler = data_loader.RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        train_datasampler = data_loader.SequentialShapeRandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % opt.training_order)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    test_dataset = data_loader.PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.testset,
        points_per_patch=opt.points_per_patch,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        cache_capacity=opt.cache_capacity,
        sub_sample_size=opt.sub_sample_size,
        recon_flag=False,
        patch_radius=opt.patch_radius,
        num_workers=int(opt.workers),
        epsilon=-1,  # not necessary for training
        k=opt.k
    )
    if opt.training_order == 'random':
        test_datasampler = data_loader.RandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        test_datasampler = data_loader.SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % opt.training_order)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    print('Training set: {} patches (in {} batches) | Test set: {} patches (in {} batches)'.format(
          len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass

    w_x = 1.0 / opt.grid_resolution * opt.w

    train_fraction_done = 0.0

    log_writer = SummaryWriter(log_dirname, comment=opt.name)
    log_writer.add_scalar('LR', opt.lr, 0)

    # milestones in number of optimizer iterations
    optimizer = optim.SGD(lmi_model.parameters(), lr=opt.lr, momentum=opt.momentum)

    # SGD changes lr depending on training progress
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)  # constant lr
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.scheduler_steps, gamma=0.1)

    if opt.refine == '':
        lmi_model.cuda(device=device)
        lmi_model = torch.nn.DataParallel(lmi_model)

    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)

    
    for epoch in range(start_epoch, opt.nepoch, 1):
        train_enum = enumerate(train_dataloader, 0)

        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(test_dataloader, 0)

        for train_batchind, batch_data_train in train_enum:

            # batch data to GPU
            for key in batch_data_train.keys():
                batch_data_train[key] = batch_data_train[key].cuda(non_blocking=True)

            # set to training mode
            lmi_model.train()

            # zero gradients
            optimizer.zero_grad()
            pred_train = lmi_model(batch_data_train)

            loss_train = compute_loss(
                pred=pred_train, batch_data=batch_data_train, w_x=w_x
            )
            ##output_test(batch_data_train, w_x)
            loss_total = sum(loss_train)

            # back-propagate through entire network to compute gradients of loss w.r.t. parameters
            loss_total.backward()

            # parameter optimization step
            optimizer.step()

            train_fraction_done = (train_batchind+1) / train_num_batch

            do_logging(log_prefix=green('train'), epoch=epoch, opt=opt, loss=loss_train, batchind=train_batchind, num_batch=train_num_batch)
            while test_fraction_done <= train_fraction_done and test_batchind + 1 < test_num_batch:

                # set to evaluation mode, no auto-diff
                lmi_model.eval()

                test_batchind, batch_data_test = next(test_enum)

                # batch data to GPU
                for key in batch_data_test.keys():
                    batch_data_test[key] = batch_data_test[key].cuda(non_blocking=True)

                # forward pass
                with torch.no_grad():
                    pred_test = lmi_model(batch_data_test)

                loss_test = compute_loss(
                    pred=pred_test, batch_data=batch_data_test, w_x=w_x
                )


                test_fraction_done = (test_batchind+1) / test_num_batch
                do_logging(log_prefix=blue('test'), epoch=epoch, opt=opt, loss=loss_test, batchind=test_batchind,
                            num_batch=train_num_batch)

        # end of epoch save model, overwriting the old model

        if epoch % opt.save_interval == 0 or epoch == opt.nepoch-1:
            torch.save(lmi_model.state_dict(), os.path.join(opt.outdir, '%s_model_%d.pth' % (opt.name, epoch)))

        # update and log lr
        lr_before_update = scheduler.get_last_lr()
        if isinstance(lr_before_update, list):
            lr_before_update = lr_before_update[0]
        scheduler.step()
        lr_after_update = scheduler.get_last_lr()
        if isinstance(lr_after_update, list):
            lr_after_update = lr_after_update[0]
        if lr_before_update != lr_after_update:
            print('LR changed from {} to {} in epoch {}'.format(lr_before_update, lr_after_update, epoch))
        current_step = (epoch + 1) * train_num_batch * opt.batchSize - 1
        log_writer.add_scalar('LR', lr_after_update, current_step)

        log_writer.flush()

    log_writer.close()
    


def compute_loss(pred, batch_data, w_x):

    loss = []
    o_pred = pred.squeeze()
    o_sdf = batch_data['imp_surf_dist_ms'].squeeze()
    o_indicator = torch.clamp(o_sdf, min=-w_x / 2.0, max=w_x / 2.0) / w_x
    loss.append((sdf_nn.calc_loss_distance(pred=o_pred, target=o_indicator)).float())
    return loss




if __name__ == '__main__':
    train_opt = parse_arguments()
    lmi_train(train_opt)
