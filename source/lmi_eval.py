import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from tqdm import tqdm

from source.lmi_model import LMIModel
#from source.lmi_model_no_sef import LMIModel_NOSEF
from source import data_loader
from source import sdf_nn
from source.base import file_utils


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', type=str, default='datasets/GRSI', help='input folder (meshes)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='output folder (estimated point cloud properties)')
    parser.add_argument('--dataset', nargs='+', type=str, default=['testset.txt'], help='shape set file name')
    parser.add_argument('--query_grid_resolution', type=int, default=None,
                        help='resolution of sampled volume used for reconstruction')
    parser.add_argument('--epsilon', type=int, default=None,
                        help='neighborhood size for reconstruction')
    parser.add_argument('--certainty_threshold', type=float, default=None, help='')
    parser.add_argument('--sigma', type=int, default=None, help='')
    parser.add_argument('--modeldir', type=str, default='models', help='model folder')
    parser.add_argument('--models', type=str, default='lmi',
                        help='names of trained models, can evaluate multiple models')
    parser.add_argument('--modelpostfix', type=str, default='_model_exp.pth', help='model file postfix')
    parser.add_argument('--parampostfix', type=str, default='_params.pth', help='parameter file postfix')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')

    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random '
                        'points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--patches_per_shape', type=int, default=1000,
                        help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')

    parser.add_argument('--sub_sample_size', type=int, default=500,
                        help='number of points of the point cloud that are trained with each patch')
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=0, help='batch size, if 0 the training batch size is used')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--w', type=int, default=4,
                        help='width coefficient for the modified indicator functions.')

    opt = parser.parse_args(args=args)
    if len(opt.dataset) == 1:
        opt.dataset = opt.dataset[0]

    return opt



def make_dataset(train_opt, eval_opt):
    dataset = data_loader.PointcloudPatchDataset(
        root=eval_opt.indir, shape_list_filename=eval_opt.dataset,
        points_per_patch=train_opt.points_per_patch,
        seed=eval_opt.seed,
        cache_capacity=eval_opt.cache_capacity,
        sub_sample_size=train_opt.sub_sample_size,
        query_grid_resolution=eval_opt.query_grid_resolution,
        num_workers=int(eval_opt.workers),
        recon_flag=True,
        patch_radius=train_opt.patch_radius,
        epsilon=eval_opt.epsilon,  # not necessary for training
        k=train_opt.k,   #neighbour_size in the SEF Extractor
    )
    return dataset


def make_datasampler(eval_opt, dataset):
    if eval_opt.sampling == 'full':
        return data_loader.SequentialPointcloudPatchSampler(dataset)
    elif eval_opt.sampling == 'sequential_shapes_random_patches':
        return data_loader.SequentialShapeRandomPointcloudPatchSampler(
            dataset,
            patches_per_shape=eval_opt.patches_per_shape,
            seed=eval_opt.seed,
            sequential_shapes=True,
            identical_epochs=False)
    else:
        raise ValueError('Unknown sampling strategy: %s' % eval_opt.sampling)


def make_dataloader(eval_opt, dataset, datasampler, model_batch_size):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=model_batch_size,
        num_workers=int(eval_opt.workers))
    return dataloader

def make_regressor(train_opt, model_filename, device):
    lmi_model = LMIModel(
        net_size_max=train_opt.net_size if 'net_size' in train_opt else 1024,
        num_points=train_opt.points_per_patch,
        sub_sample_size=train_opt.sub_sample_size,
        k=train_opt.k,
    )

    lmi_model.cuda(device=device)  # same order as in training
    lmi_model = torch.nn.DataParallel(lmi_model)
    lmi_model.load_state_dict(torch.load(model_filename))
    lmi_model.eval()
    return lmi_model


def save_reconstruction_data(imp_surf_dist_ms, dataset, model_out_dir, shape_ind):

    from source import sdf

    shape = dataset.shape_cache.get(shape_ind)

    imp_surf_dist_ms_nan = np.isnan(imp_surf_dist_ms)
    # the predicted distance would be greater than 1 -> not possible with tanh
    imp_surf_dist_ms[imp_surf_dist_ms_nan] = 1.0

    # save query points
    os.makedirs(os.path.join(model_out_dir, 'query_pts_ms'), exist_ok=True)
    np.save(os.path.join(model_out_dir, 'query_pts_ms', dataset.shape_names[shape_ind] + '.xyz.npy'),
            shape.imp_surf_query_point_ms)

    # save query distance in model space
    os.makedirs(os.path.join(model_out_dir, 'dist_ms'), exist_ok=True)
    np.save(os.path.join(model_out_dir, 'dist_ms', dataset.shape_names[shape_ind] + '.xyz.npy'), imp_surf_dist_ms)

    # debug query points with color for distance
    os.makedirs(os.path.join(model_out_dir, 'query_pts_ms_vis'), exist_ok=True)
    sdf.visualize_query_points(
        query_pts_ms=shape.imp_surf_query_point_ms, query_dist_ms=imp_surf_dist_ms,
        file_out_off=os.path.join(model_out_dir, 'query_pts_ms_vis', dataset.shape_names[shape_ind] + '.ply'))


def save_evaluation(dataset, model_out_dir, shape_ind,
                    shape_patch_values, w_x):

    imp_surf_dis_ms = shape_patch_values.cpu().numpy().squeeze()
    imp_surf_dis_ms = imp_surf_dis_ms * w_x
    save_reconstruction_data(imp_surf_dis_ms, dataset, model_out_dir, shape_ind)

def lmi_eval(eval_opt):

    models = eval_opt.models.split()

    if eval_opt.seed < 0:
        eval_opt.seed = random.randint(1, 10000)

    device = torch.device("cpu" if eval_opt.gpu_idx < 0 else "cuda:%d" % eval_opt.gpu_idx)
    w_x = 1.0 / eval_opt.query_grid_resolution * eval_opt.w
    for model_name in models:

        print("Random Seed: %d" % eval_opt.seed)
        random.seed(eval_opt.seed)
        torch.manual_seed(eval_opt.seed)

        model_filename = os.path.join(eval_opt.modeldir, model_name+eval_opt.modelpostfix)
        param_filename = os.path.join(eval_opt.modeldir, model_name+eval_opt.parampostfix)

        # load model and training parameters
        train_opt = torch.load(param_filename)
        
        if eval_opt.batchSize == 0:
            model_batch_size = train_opt.batchSize
        else:
            model_batch_size = eval_opt.batchSize

        
        dataset = make_dataset(train_opt=train_opt, eval_opt=eval_opt)
        datasampler = make_datasampler(eval_opt=eval_opt, dataset=dataset)
        dataloader = make_dataloader(eval_opt=eval_opt, dataset=dataset, datasampler=datasampler,
                                     model_batch_size=model_batch_size)
        lmi_model = make_regressor(train_opt=train_opt, model_filename=model_filename, device=device)

        shape_ind = 0
        shape_patch_offset = 0
        if eval_opt.sampling == 'full':
            shape_patch_count = dataset.shape_patch_count[shape_ind]
        elif eval_opt.sampling == 'sequential_shapes_random_patches':
            shape_patch_count = min(eval_opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
        else:
            raise ValueError('Unknown sampling strategy: %s'.format(eval_opt.sampling))

        shape_patch_values = torch.zeros(shape_patch_count, 1,
                                         dtype=torch.float32, device=device)

        # append model name to output directory and create directory if necessary
        if eval_opt.reconstruction:
            model_out_dir = os.path.join(eval_opt.outdir, 'rec')
        else:
            model_out_dir = os.path.join(eval_opt.outdir, 'eval')
        if not os.path.exists(model_out_dir):
            os.makedirs(model_out_dir)

        print(f'evaluating {len(dataset)} patches')
        for batch_data in tqdm(dataloader):

            # batch data to GPU
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].cuda(non_blocking=True)

            with torch.no_grad():
                batch_pred = lmi_model(batch_data)

            batch_offset = 0
            while batch_offset < batch_pred.size(0):

                shape_patches_remaining = shape_patch_count-shape_patch_offset
                batch_patches_remaining = batch_pred.size(0)-batch_offset
                samples_remaining = min(shape_patches_remaining, batch_patches_remaining)

                # append estimated patch properties batch to properties for the current shape
                patch_properties = batch_pred[batch_offset:batch_offset+samples_remaining]
                shape_patch_values[shape_patch_offset:shape_patch_offset+samples_remaining] = patch_properties
            
                batch_offset = batch_offset + samples_remaining
                shape_patch_offset = shape_patch_offset + samples_remaining

                if shape_patches_remaining <= batch_patches_remaining:
                    save_evaluation(dataset, model_out_dir, shape_ind, shape_patch_values, w_x=w_x)

                    # start new shape
                    if shape_ind + 1 < len(dataset.shape_names):
                        shape_patch_offset = 0
                        shape_ind = shape_ind + 1
                        if eval_opt.sampling == 'full':
                            shape_patch_count = dataset.shape_patch_count[shape_ind]
                        elif eval_opt.sampling == 'sequential_shapes_random_patches':
                            shape_patch_count = len(datasampler.shape_patch_inds[shape_ind])
                        else:
                            raise ValueError('Unknown sampling strategy: %s' % eval_opt.sampling)
                        shape_patch_values = torch.zeros(shape_patch_count, 1,
                                                         dtype=torch.float32, device=device)


if __name__ == '__main__':
    eval_opt = parse_arguments()
    lmi_eval(eval_opt)
