import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from source.base import utils
import numpy as np
input_dims_per_point = 3


class STN(nn.Module):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.net_size_max = net_size_max
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.net_size_max, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(self.net_size_max, int(self.net_size_max / 2))
        self.fc2 = nn.Linear(int(self.net_size_max / 2), int(self.net_size_max / 4))
        self.fc3 = nn.Linear(int(self.net_size_max / 4), self.dim*self.dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.net_size_max)
        self.bn4 = nn.BatchNorm1d(int(self.net_size_max / 2))
        self.bn5 = nn.BatchNorm1d(int(self.net_size_max / 4))

        if self.num_scales > 1:
            self.fc0 = nn.Linear(self.net_size_max * self.num_scales, self.net_size_max)
            self.bn0 = nn.BatchNorm1d(self.net_size_max)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), self.net_size_max * self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*self.net_size_max:(s+1)*self.net_size_max, :] = \
                    self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, self.net_size_max*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x



class LMIModel(nn.Module):  # basing on P2S and PointNet
    def __init__(self, net_size_max=1024, num_points=200,
                 sub_sample_size=1000, k=10):
        super(LMIModel, self).__init__()

        self.net_size_max = net_size_max
        self.num_points = num_points
        self.use_point_stn = False
        self.sub_sample_size = sub_sample_size
        self.k = k
        self.point_stn = STN(net_size_max=net_size_max, num_scales=1,
                        num_points=self.num_points + self.sub_sample_size, dim=3, sym_op='max')
        self.conv0a_patch = torch.nn.Conv1d(3, 64, 1)
        self.bn0a_patch = nn.BatchNorm1d(64)
        self.conv0b_patch = torch.nn.Conv1d(64, 64, 1)
        self.bn0b_patch = nn.BatchNorm1d(64)

        self.conv0c_patch = torch.nn.Conv1d(64, 128, 1)
        self.bn0c_patch = nn.BatchNorm1d(128)
        self.conv0d_patch = torch.nn.Conv1d(128, 256, 1)
        self.bn0d_patch = nn.BatchNorm1d(256)


        self.conv0a_shape = torch.nn.Conv1d(3, 64, 1)
        self.conv0b_shape = torch.nn.Conv1d(64, 64, 1)
        self.bn0a_shape = nn.BatchNorm1d(64)
        self.bn0b_shape = nn.BatchNorm1d(64)


        self.k_neighbour_mp_patch = torch.nn.MaxPool1d(kernel_size=self.k + 1, stride=self.k + 1)
        self.conv1b_patch = torch.nn.Conv1d(256, 128, 1)
        self.bn1b_patch = nn.BatchNorm1d(128)

        self.conv1c_patch = torch.nn.Conv1d(128, 64, 1)
        self.bn1c_patch = nn.BatchNorm1d(64)

        self.conv1_local = torch.nn.Conv1d(128, 128, 1)
        self.conv2_local = torch.nn.Conv1d(128, 128, 1)
        self.conv3_local = torch.nn.Conv1d(128, 1024, 1)
        self.bn1_pn_local = nn.BatchNorm1d(128)
        self.bn2_pn_local = nn.BatchNorm1d(128)
        self.bn3_pn_local = nn.BatchNorm1d(1024)

        self.conv1_global = torch.nn.Conv1d(64, 64, 1)
        self.conv2_global = torch.nn.Conv1d(64, 128, 1)
        self.conv3_global = torch.nn.Conv1d(128, 1024, 1)
        self.bn1_pn_global = nn.BatchNorm1d(64)
        self.bn2_pn_global = nn.BatchNorm1d(128)
        self.bn3_pn_global = nn.BatchNorm1d(1024)

        self.mp1_local = torch.nn.MaxPool1d(self.num_points)
        self.mp1_global = torch.nn.MaxPool1d(self.sub_sample_size)

        self.fc1_local = nn.Linear(2048, 1024)
        self.bn1_local = nn.BatchNorm1d(1024)
        self.fc2_local = nn.Linear(1024, 1024)
        self.bn2_local = nn.BatchNorm1d(1024)
        self.fc1_global = nn.Linear(2048, 1024)
        self.bn1_global = nn.BatchNorm1d(1024)
        self.fc2_global = nn.Linear(1024, 1024)
        self.bn2_global = nn.BatchNorm1d(1024)

        self.conb_local_conv1 = torch.nn.Conv1d(1152, 512, 1)
        self.conb_local_conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conb_local_bn1 = nn.BatchNorm1d(512)
        self.conb_local_bn2 = nn.BatchNorm1d(256)

        self.conb_global_conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conb_global_conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conb_global_bn1 = nn.BatchNorm1d(512)
        self.conb_global_bn2 = nn.BatchNorm1d(256)


        self.fc_last_1 = nn.Linear(512, 256)
        self.bn_last_1 = nn.BatchNorm1d(256)

        self.fc_last_2 = nn.Linear(256, 128)
        self.bn_last_2 = nn.BatchNorm1d(128)

        self.fc_last_3 = nn.Linear(128, 1)

    def forward(self, x):
        patch_points_knn = x['patch_points_knn'].transpose(1, 2) 
        shape_points = x['shape_points'].transpose(1, 2)  
        shape_query_point = x['imp_surf_query_point_ms'].unsqueeze(2)
        
        patch_points_knn -= shape_query_point.expand(patch_points_knn.shape)
        shape_points -= shape_query_point.expand(shape_points.shape)
        
        #point and local feature module
        patch_point_self = patch_points_knn[:, :, 0::(self.k + 1)]
        all_point = torch.cat((patch_point_self, shape_points), axis=2)
        trans_pt = self.point_stn(all_point)
        patch_points_knn = torch.bmm(trans_pt, patch_points_knn)
        shape_points = torch.bmm(trans_pt, shape_points)


        

        patch_points_knn = F.relu(self.bn0a_patch(self.conv0a_patch(patch_points_knn)))
        patch_points_knn = F.relu(self.bn0b_patch(self.conv0b_patch(patch_points_knn)))
        patch_point_feature_self = patch_points_knn[:, :, 0::(self.k + 1)]
        patch_points_knn = F.relu(self.bn0c_patch(self.conv0c_patch(patch_points_knn)))
        patch_points_knn = F.relu(self.bn0d_patch(self.conv0d_patch(patch_points_knn)))
        patch_point_feature = self.k_neighbour_mp_patch(patch_points_knn)
        patch_point_feature = F.relu(self.bn1b_patch(self.conv1b_patch(patch_point_feature)))
        patch_point_feature = F.relu(self.bn1c_patch(self.conv1c_patch(patch_point_feature)))
        
        shape_point_feature = F.relu(self.bn0a_shape(self.conv0a_shape(shape_points)))
        shape_point_feature = F.relu(self.bn0b_shape(self.conv0b_shape(shape_point_feature)))
        patch_point_feature = torch.cat((patch_point_feature, patch_point_feature_self), axis=1)
        patch_local_feature = patch_point_feature
        shape_local_feature = shape_point_feature
        
        
        #global feature module
        patch_point_feature = F.relu(self.bn1_pn_local(self.conv1_local(patch_point_feature)))
        patch_point_feature = F.relu(self.bn2_pn_local(self.conv2_local(patch_point_feature)))
        patch_point_feature = self.bn3_pn_local(self.conv3_local(patch_point_feature))

        shape_point_feature = F.relu(self.bn1_pn_global(self.conv1_global(shape_point_feature)))
        shape_point_feature = F.relu(self.bn2_pn_global(self.conv2_global(shape_point_feature)))
        shape_point_feature = self.bn3_pn_global(self.conv3_global(shape_point_feature))

        
        patch_features = self.mp1_local(patch_point_feature).squeeze()
        shape_features = self.mp1_global(shape_point_feature).squeeze()

        global_feature = torch.cat((patch_features, shape_features), dim=1)  
        patch_features = F.relu(self.bn1_local(self.fc1_local(global_feature))) 
        patch_features = F.relu(self.bn2_local(self.fc2_local(patch_features)))  

        shape_features = F.relu(self.bn1_global(self.fc1_global(global_feature))) 
        shape_features = F.relu(self.bn2_global(self.fc2_global(shape_features)))  

        #indicator prediction module
        patch_features = patch_features.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        patch_features = torch.cat((patch_features, patch_local_feature), axis=1)
        shape_features = shape_features.view(-1, 1024, 1).repeat(1, 1, self.sub_sample_size)  
        shape_features = torch.cat((shape_features, shape_local_feature), axis=1)  
        patch_features = F.relu(self.conb_local_bn1(self.conb_local_conv1(patch_features)))
        patch_features = F.relu(self.conb_local_bn2(self.conb_local_conv2(patch_features)))  
        patch_features = patch_features.sum(axis=2) ##[50, 256]

        shape_features = F.relu(self.conb_global_bn1(self.conb_global_conv1(shape_features)))
        shape_features = F.relu(self.conb_global_bn2(self.conb_global_conv2(shape_features))) ##[50, 256, 1000]
        shape_features = shape_features.sum(axis=2)  ##[50, 256]

        total_feature = torch.cat((patch_features,  shape_features), dim=1).squeeze() ##[50, 512]
        indicator = F.relu(self.bn_last_1(self.fc_last_1(total_feature)))
        indicator = F.relu(self.bn_last_2(self.fc_last_2(indicator)))
        indicator = self.fc_last_3(indicator)

        return indicator
