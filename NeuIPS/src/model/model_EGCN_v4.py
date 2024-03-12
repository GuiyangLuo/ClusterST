import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np
import scipy.sparse as sp
import math
import pandas as pd
import numpy as np
import numpy as np
import cvxpy as cp
import pickle
from cluster.multi_kmeans import MultiKMeans
from  model.VraeModel import VRAE as VRAE_Model
from  model.vis_cluster import Visualization_Cluster as Visualization_Cluster
from cluster.net_hyper_parameter  import Caculate_parameters_conv

class CommonGCN(nn.Module):

    def __init__(self, args, adj_matrix, edge_attribute, sensor_indexes):
        super(CommonGCN, self).__init__()
        self.adj_matrix = adj_matrix
        self.edge_attribute = edge_attribute
        self.sensor_indexes = sensor_indexes
        self.edge_attribute_len = edge_attribute.shape[-1]
        self.edge_att_gcn_module_list = nn.ModuleList()
        self.edge_att_gcn_module_list_activations = nn.ModuleList()
        previous = self.edge_attribute_len
        for index, feature_size in enumerate(args.static_gcn_feature_size):
            block_net = nn.Sequential(nn.Linear(previous, feature_size, bias=True), )
            previous = feature_size
            self.edge_att_gcn_module_list.add_module('edge_att_gcn_{}'.format(index), block_net)
            self.edge_att_gcn_module_list_activations.add_module('edge_att_gcn_activations_{}'.format(index),  nn.ReLU())



    def forward(self,):

        input = self.edge_attribute
        for net,act in zip(self.edge_att_gcn_module_list, self.edge_att_gcn_module_list_activations):
            input = net(input)
            all_nodes_aggregate = torch.spmm(self.adj_matrix, input)
            input = act(input)

        output = input[self.sensor_indexes]

        return output

class Static_features(nn.Module):

    def __init__(self, args, road_sensor_pos, dim_in, dim_out):
        super(Static_features, self).__init__()
        self.road_sensor_pos = road_sensor_pos
        self.static_nets = nn.Sequential(nn.Linear(dim_in, int((dim_in + dim_out)/2), bias=True),
                                         nn.ReLU(),
                                         nn.Linear( int((dim_in + dim_out)/2),dim_out,  bias=True),
                                         nn.ReLU(), )
        # self.static_nets = nn.Linear(dim_in,  dim_out, bias=True)

        self.layernorm = nn.BatchNorm1d(args.number_of_sensors)

    def forward(self, ):
        # print("static netwrok", self.road_sensor_pos.mean())
        # print("************ weights: ", self.static_nets.weight)
        out = self.static_nets(self.road_sensor_pos)
        out = self.layernorm(out)
        return out

class Dynamic_features(nn.Module):

    def __init__(self, args):
        super(Dynamic_features, self).__init__()
        # accepted input should be shape of :  [seq_len, batch, input_size]
        self.gru = nn.GRU(args.in_dims, args.dynamic_gru_feature_size, args.gru_number_of_layers, bidirectional  = args.gru_bidirectional)
        self.gru_bidirectional = 1 if args.gru_bidirectional is False else 2
        # print("  self.gru_bidirectional = ", self.gru_bidirectional, )
        self.h_0 =(torch.randn(args.gru_number_of_layers*self.gru_bidirectional, args.number_of_sensors *  args.batch_size, args.dynamic_gru_feature_size))
        device = torch.device(args.device)
        self.h_0 =  nn.Parameter(torch.Tensor(self.h_0).to(device))

        # self.h_0  = None
        self.layernorm = nn.LayerNorm([args.dynamic_gru_feature_size])

    def init_hidden(self):
        for  param  in self.gru.parameters():
            # print("name = ",  param )
            nn.init.orthogonal(param)


    def forward(self, input):
        # input shape (batch_size, in_dim, #edges, in_seq)
        batch_size, in_dim, num_sensors, in_seq = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
        input = input.permute(3,0,2,1).contiguous()
        # print(" Dynamic_features  input for gru =  ", input.shape)
        input = input.view(in_seq,-1,in_dim)
        # print(" Dynamic_features  input for gru =  ", input.shape)
        output, hidden_0 = self.gru(input, self.h_0)  # (num_layers * num_directions, batch, hidden_size)
        hidden_0 = hidden_0[-self.gru_bidirectional:, :, :]

        hidden_0 = hidden_0.permute(1, 0, 2).contiguous().view(batch_size, num_sensors, -1).contiguous()
        # print(" dynamic  hidden_0 ", hidden_0.shape)
        hidden_0 = self.layernorm(hidden_0)
        return hidden_0

class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, args, input_channel = 1 , output_channel = 256, resnet_layers = [1, 2, 2, 2],  ):
        super(ResNet_FeatureExtractor, self).__init__()

        self.cal_bags = Caculate_parameters_conv(args.gcn_n_class, args.seq_length_x, args.max_allow_spatial_conv, args.max_allow_dilation, total_number = 8)
        kernels_dialations = self.cal_bags.main()
        print(" kernels_dialations ",args.gcn_n_class, args.seq_length_x, kernels_dialations)
        kernels_dialations = np.array(kernels_dialations)
        kernels = kernels_dialations[:,:2]
        dialations = kernels_dialations[:, 2:4]
        self.ConvNet = ResNet( input_channel, output_channel, BasicBlock, resnet_layers,  kernels, dialations)
    def forward(self, input):
        return self.ConvNet(input)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv1x3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = self._conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv1x3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=(1,stride,stride),
                         padding=(0,1,1), bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print('in residual connections ',out.shape, x.shape, residual.shape)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,  input_channel, output_channel, block, layers, kernels, dialations ):
        super(ResNet, self).__init__()

        i = 0
        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv3d(input_channel, int(output_channel / 16),
                                 kernel_size=(1, kernels[i][0], kernels[i][1]), dilation = (1, dialations[i][0], dialations[i][1]),  stride=1, padding=0, bias=False)
        self.bn0_1 = nn.BatchNorm3d(int(output_channel / 16))
        i = i+1
        self.conv0_2 = nn.Conv3d(int(output_channel / 16), self.inplanes,
                                 kernel_size=(1, kernels[i][0], kernels[i][1]),
                                 dilation=(1, dialations[i][0], dialations[i][1]), stride=1, padding=0, bias=False)
        self.bn0_2 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        i = i + 1
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, kernels[i][0], kernels[i][1]),
                                 dilation=(1, dialations[i][0], dialations[i][1]), stride=1,padding=0)

        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        i = i + 1
        self.conv1 = nn.Conv3d(self.output_channel_block[0], self.output_channel_block[
            0],kernel_size=(1, kernels[i][0], kernels[i][1]),
                                 dilation=(1, dialations[i][0], dialations[i][1]), stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.output_channel_block[0])
        i = i + 1
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, kernels[i][0], kernels[i][1]),
                                 dilation=(1, dialations[i][0], dialations[i][1]), stride=1, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        i = i + 1
        self.conv2 = nn.Conv3d(self.output_channel_block[1], self.output_channel_block[
            1], kernel_size=(1, kernels[i][0], kernels[i][1]),
                                 dilation=(1, dialations[i][0], dialations[i][1]), stride=1, bias=False)
        self.bn2 = nn.BatchNorm3d(self.output_channel_block[1])

        i = i + 1
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1, kernels[i][0], kernels[i][1]),
                                 dilation=(1, dialations[i][0], dialations[i][1]), stride=1,)
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        i = i + 1
        self.conv3 = nn.Conv3d(self.output_channel_block[2], self.output_channel_block[
            2], kernel_size=(1, kernels[i][0], kernels[i][1]),
                                 dilation=(1, dialations[i][0], dialations[i][1]), stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.output_channel_block[2])



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("resnet input ", x.shape)
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)
        # print("resnet  00", x.shape)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("maxpool2  before", x.shape)
        x = self.maxpool2(x)
        # print("maxpool2  after", x.shape)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)

        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # print("resnet 33 ", x.shape)
        # x = self.layer4(x)
        # x = self.conv4_1(x)
        # x = self.bn4_1(x)
        # x = self.relu(x)
        # x = self.conv4_2(x)
        # x = self.bn4_2(x)
        # x = self.relu(x)

        return x

class Spatial_temporal(nn.Module):

    def __init__(self, args):
        super(Spatial_temporal, self).__init__()
        # self.spatial_temporal_networks = nn.Sequential()
        channels_conv_start_layer = args.channels_conv_start_layer

        self.start_conv = self.final_convs= nn.Sequential(
            nn.Conv3d(1, channels_conv_start_layer, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                      bias=True),
            nn.ReLU(),
        )
        channels_conv_extractor_out =  args.channels_conv_extractor_out

        self.spatial_temporal_feature_extractor = ResNet_FeatureExtractor(args, input_channel = channels_conv_start_layer , output_channel = channels_conv_extractor_out,  resnet_layers = args.resnet_layers)

        self.final_convs= nn.Sequential(
            nn.Conv3d(channels_conv_extractor_out, int(channels_conv_extractor_out/2), kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                      bias=True),
            nn.ReLU(),
            nn.Conv3d(int(channels_conv_extractor_out/2), 12, kernel_size=(1, 1, 1), dilation=(1, 1, 1), stride=(1, 1, 1),
                      bias=True),
        )

    def forward(self, threeDinput):
        # input shape (batch_size, number_of_sensors, neighbors, 1, in-seq)
        threeDinput = threeDinput.permute(0,3,1,2,4).contiguous()
        output =  self.start_conv(threeDinput)
        output = self.spatial_temporal_feature_extractor(output)
        output = self.final_convs(output)

        output = output.squeeze(-1)

        return output

class EmbedGCN(nn.Module):

    def __init__(self, args,  road_sensor_pos, adj_mx_topk_index,  writer =None):
        super(EmbedGCN, self).__init__()
        self.args = args
        self.road_sensor_pos = road_sensor_pos
        self.adj_mx_topk_index = adj_mx_topk_index

        self.writer = writer
        self.static_feature_module = Static_features(args, road_sensor_pos, road_sensor_pos.shape[-1] ,args.static_feature_size)
        self.dynamic_feature_module = Dynamic_features(args)
        self.spatial_temporal = Spatial_temporal(args)
        self.global_terater = 0

        if self.args.fusion == 'dynamic':
            self.feature_size = args.dynamic_gru_feature_size
        elif self.args.fusion == 'static':
            self.feature_size = args.static_feature_size
        elif self.args.fusion == 'concat':
            self.feature_size = args.static_feature_size + args.dynamic_gru_feature_size
        else:
            assert 1 == 0, f"fusion mechanism {self.args.fusion} is not defined!!"

        if args.centroid_way =="batch":
            self.dynamic_cluster = clustering_dynamic_learning(args, self.feature_size, adj_mx_topk_index)
        elif args.centroid_way =="non_batch_1":
            self.dynamic_cluster = clustering_dynamic_learning_common_center(args, self.feature_size, adj_mx_topk_index)
        elif args.centroid_way == "non_batch_2":
            self.dynamic_cluster = clustering_dynamic_learning_common_center_2(args, self.feature_size, adj_mx_topk_index)
        elif args.centroid_way == "non_batch_3":
            self.dynamic_cluster = clustering_dynamic_learning_common_center_3(args, self.feature_size,
                                                                               adj_mx_topk_index)
        elif args.centroid_way == "non_batch_4":
            self.dynamic_cluster = clustering_dynamic_learning_common_center_4(args, self.feature_size,
                                                                               adj_mx_topk_index)
        elif args.centroid_way == "non_batch_5":
            self.dynamic_cluster = clustering_dynamic_learning_common_center_5(args, self.feature_size,
                                                                               adj_mx_topk_index)
        elif args.centroid_way == "non_batch_6":
            self.dynamic_cluster = clustering_dynamic_learning_common_center_6(args, self.feature_size,
                                                                               adj_mx_topk_index)
        elif  args.centroid_way == "non_batch_7":
            self.dynamic_cluster = clustering_dynamic_learning_common_center_7(args, self.feature_size,
                                                                               adj_mx_topk_index)
        elif args.centroid_way == "non_batch_8":
            self.dynamic_cluster = clustering_dynamic_learning_common_center_8(args, self.feature_size,
                                                                               adj_mx_topk_index)

        self.visualization_cluster = Visualization_Cluster(args, writer, road_sensor_pos)
        self.bn2d = nn.InstanceNorm2d( args.number_of_sensors)


    def cosine_simularity(self, x):
        ## shape of x is (batch_size,  # sensors,fused_dim)
        x = x.permute((1, 2, 0))
        cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
        return cos_sim_pairwise
    def forward(self, input_original):
        self.global_terater =   self.global_terater + 1
        # input shape (batch_size, in_dim, #edges, in_seq)
        # output shape (batch_size, in_seq, #edges, in_dim)

        input_for_dynamic_features = input_original.clone().detach()
        input = input_original
        if self.args.fusion == 'dynamic':
            dynamic_fea = self.dynamic_feature_module(input_for_dynamic_features)
            fushed_features = dynamic_fea
        elif  self.args.fusion == 'static':
            static_fea = self.static_feature_module()
            fushed_features = static_fea
        elif  self.args.fusion == 'concat':
            dynamic_fea = self.dynamic_feature_module(input_for_dynamic_features)
            static_fea = self.static_feature_module()
            fushed_features = torch.cat([static_fea, dynamic_fea], axis=2)  # shape is (batch_size,#sensors,fused_dim)
        else:
            assert 1 == 0,  f"fusion mechanism {self.args.fusion} is not defined!!"

        clustred_output, cluster_difference_loss = self.dynamic_cluster(fushed_features, input)
        # print(" input and clustred_output", input.mean(), clustred_output.mean())
        weightedDinput_topk = clustred_output.unsqueeze(3).contiguous()

        output = self.spatial_temporal(weightedDinput_topk)

        return output, cluster_difference_loss


class clustering_dynamic_learning(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index,margin = 0.5):
        super(clustering_dynamic_learning, self).__init__()
        device = torch.device(args.device)
        self.batch_size = args.batch_size
        # self.centroids = nn.Parameter(torch.randn([ args.number_of_sensors,args.gcn_n_class, fused_dim])).to(device)
        self.centroids = (torch.randn([args.number_of_sensors, args.gcn_n_class, fused_dim])).to(device)
        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors
        self.similarity_net = nn.Sequential(nn.Linear(fused_dim*2, fused_dim),
                                            nn.ReLU(),
                                            nn.Linear(fused_dim, 1),
                                            nn.ReLU(),   )

        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1,-1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = self.adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, args.seq_length_x)
        self.update_rate = torch.tensor(0.01).to(device)
        self.minius_update_rate = torch.tensor(1).to(device) - self.update_rate
        eye_mat = torch.eye(self.gcn_n_class)
        self.target = (torch.ones_like(eye_mat) - eye_mat)*margin
        self.target=  self.target.to(device)

        self.bn_centroid = nn.BatchNorm3d(args.number_of_sensors)
        self.bn_feature = nn.BatchNorm3d(args.number_of_sensors)

    def fast_cdist(self, x1, x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
        # Compute squared distance matrix using quadratic expansion
        # But be clever and do it with a single matmul call
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        # Zero out negative values
        res.clamp_min_(1e-30).sqrt_()
        return res

    def forward(self, fushed_features,  input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """
        input_expand = input_data.expand(-1, self.number_of_sensors, -1, -1)
        input_expand_topk = torch.gather(input_expand, 2, self.adj_mx_topk_index_expanded_seq)
        fushed_features_expand = fushed_features.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        fushed_features_expand_topk = torch.gather(fushed_features_expand, 2, self.adj_mx_topk_index_expanded_seq)

        centroids_expand = self.centroids.unsqueeze(0).unsqueeze(2).expand(self.batch_size, -1, self.neighbors_for_each_nodes, -1, -1)
        centroids_expand = self.bn_centroid(centroids_expand)
        fused_features_expand = fushed_features_expand_topk.unsqueeze(3).expand(-1, -1,  -1, self.gcn_n_class,  -1)
        fused_features_expand = self.bn_feature (fused_features_expand)
        concated_feature = torch.cat([fused_features_expand,centroids_expand ], dim = -1)
        # print("11--concated_feature",self.centroids[0,0,:2,:]," \n ", fused_features_expand_top_k[0,0,3:5,:]," \n ", concated_feature[0,0,3:5,:2,:])
        simi_mat = self.similarity_net(concated_feature).squeeze(-1).softmax(-1)
        # print("33--simi_mat", simi_mat.shape, self.adj_mx_topk_index_expanded.shape)
        updated_input = torch.einsum('mlikj,mlijc->mlikc', simi_mat.unsqueeze(-1), input_expand_topk.unsqueeze(-2)).mean(dim =2)
        # print("55--simi_mat", updated_centroid.shape)
        ## maximize distance
        self.centroids = self.minius_update_rate*self.centroids + self.update_rate * updated_input.mean(0)
        distance_mat = self.fast_cdist(self.centroids,  self.centroids)
        # print( " distance Mat = ",distance_mat[0,0,0:5,0:5])
        cluster_difference_loss = torch.pow(torch.clamp(self.target - distance_mat, min=0.0), 2)
        cluster_difference_loss = cluster_difference_loss.sum(-1).sum(-1).mean()
        # print("55--simi_mat", simi_mat.shape,input_top_k.shape)
        self.centroids = self.centroids.clone().detach()
        return updated_input, cluster_difference_loss
class tensor_deque(nn.Module):
    def __init__(self, args,  max_len = 100):
        super(tensor_deque, self).__init__()
        self.max_len = max_len
        self.tensor_queue = torch.zeros([max_len,args.number_of_sensors, args.neighbors_for_each_nodes, args.gcn_n_class ]).to(torch.device(args.device))
        self.cur_index = 0
    def update_index(self):
        return self.cur_index % self.max_len
    def forward(self, data):
        self.cur_index = self.cur_index + 1
        self.tensor_queue[self.update_index(),...] = data
        if self.cur_index ==1:
            return data
        elif self.cur_index < self.max_len:
            data = self.tensor_queue[:self.update_index(),...]
        else:
            data = self.tensor_queue
        return data.mean(0)



class clustering_dynamic_learning_common_center(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index,margin = 0.5):
        super(clustering_dynamic_learning_common_center, self).__init__()
        device = torch.device(args.device)
        self.batch_size = args.batch_size
        # self.centroids = nn.Parameter(torch.randn([ args.number_of_sensors,args.gcn_n_class, fused_dim])).to(device)
        self.centroids = (torch.randn([args.gcn_n_class, fused_dim])).to(device)
        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors
        self.similarity_net = nn.Sequential(nn.Linear(fused_dim*2, fused_dim),
                                            nn.ReLU(),
                                            nn.Linear(fused_dim, 1),
                                            nn.ReLU(),   )
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)
        self.update_rate = torch.tensor(0.01).to(device)
        self.minius_update_rate = torch.tensor(1).to(device) - self.update_rate
        eye_mat = torch.eye(self.gcn_n_class)
        self.target = (torch.ones_like(eye_mat) - eye_mat)*margin
        self.target=  self.target.to(device)

        self.bn_centroid = nn.BatchNorm2d(args.number_of_sensors)
        self.bn_feature = nn.BatchNorm2d(args.number_of_sensors)

    def fast_cdist(self, x1, x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
        # Compute squared distance matrix using quadratic expansion
        # But be clever and do it with a single matmul call
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        # Zero out negative values
        res.clamp_min_(1e-30).sqrt_()
        return res

    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """
        input_expand = input_data.expand(-1, self.number_of_sensors, -1, -1)
        input_expand_topk = torch.gather(input_expand, 2, self.adj_mx_topk_index_expanded_seq)

        centroids_expand = self.centroids.unsqueeze(0).unsqueeze(1).expand(self.batch_size, self.number_of_sensors, -1, -1)
        # centroids_expand = self.bn_centroid(centroids_expand)
        fused_features_expand = fushed_features.unsqueeze(2).expand(-1, -1,  self.gcn_n_class,-1)
        fused_features_expand = self.bn_feature(fused_features_expand)
        concated_feature = torch.cat([fused_features_expand,centroids_expand ], dim = -1)
        # print("11--concated_feature",self.centroids[0,0,:2,:]," \n ", fused_features_expand_top_k[0,0,3:5,:]," \n ", concated_feature[0,0,3:5,:2,:])
        simi_mat = self.similarity_net(concated_feature).squeeze(-1).softmax(-1)
        # print("33--simi_mat", simi_mat.shape, self.adj_mx_topk_index_expanded.shape)

        simi_mat_expand = simi_mat.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        simi_mat_top_k = torch.gather(simi_mat_expand,2,self.adj_mx_topk_index_expanded_class)

        updated_input = torch.einsum('mlikj,mlijc->mlikc', simi_mat_top_k.unsqueeze(-1), input_expand_topk.unsqueeze(-2)).mean(dim =2)
        # print("55--simi_mat", updated_centroid.shape)
        ## maximize distance

        self.centroids = self.minius_update_rate*self.centroids + self.update_rate * updated_input.mean(0).mean(0)
        distance_mat = self.fast_cdist(self.centroids,  self.centroids)
        # print( " distance Mat = ",distance_mat[0,0,0:5,0:5])
        cluster_difference_loss = torch.pow(torch.clamp(self.target - distance_mat, min=0.0), 2)
        cluster_difference_loss = cluster_difference_loss.sum()
        # print("55--simi_mat", simi_mat.shape,input_top_k.shape)
        self.centroids = self.centroids.clone().detach()
        return updated_input, cluster_difference_loss

class clustering_dynamic_learning_common_center_2(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index,margin = 0.5, dim_out = 6, seq_out = 12):
        super(clustering_dynamic_learning_common_center_2, self).__init__()
        device = torch.device(args.device)
        self.dim_out = dim_out
        self.batch_size = args.batch_size
        # self.centroids = nn.Parameter(torch.randn([ args.number_of_sensors,args.gcn_n_class, fused_dim])).to(device)
        self.centroids = (torch.randn([args.gcn_n_class, fused_dim])).to(device)
        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors
        middle_dim = int(fused_dim/2)
        self.similarity_net_for_centroid_1 = nn.Sequential(nn.Linear(fused_dim, middle_dim),
                                            nn.ReLU(),
                                            nn.Linear(middle_dim, dim_out),
                                            nn.ReLU(),   )
        self.similarity_net_for_centroid_2 = nn.Sequential(nn.Linear(fused_dim, dim_out),
                                                         nn.ReLU() )

        self.similarity_net_for_input_1 = nn.Sequential(nn.Linear(fused_dim, middle_dim),
                                                           nn.ReLU(),
                                                           nn.Linear(middle_dim, dim_out),
                                                           nn.ReLU(), )
        self.similarity_net_for_input_2 = nn.Sequential(nn.Linear(fused_dim, dim_out),
                                                           nn.ReLU())

        self.GCN_cluster = nn.ModuleList([nn.Sequential(nn.Linear(args.seq_length_x, seq_out), nn.ReLU()) for i in range(self.gcn_n_class)])
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)
        self.update_rate = torch.tensor(0.01).to(device)
        self.minius_update_rate = torch.tensor(1).to(device) - self.update_rate
        eye_mat = torch.eye(self.gcn_n_class)
        self.target = (torch.ones_like(eye_mat) - eye_mat)*margin
        self.target=  self.target.to(device)

        self.bn_centroid = nn.BatchNorm2d(args.number_of_sensors)
        self.bn_feature = nn.BatchNorm2d(args.number_of_sensors)

    def fast_cdist(self, x1, x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
        # Compute squared distance matrix using quadratic expansion
        # But be clever and do it with a single matmul call
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        # Zero out negative values
        res.clamp_min_(1e-30).sqrt_()
        return res

    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """
        input_expand = input_data.expand(-1, self.number_of_sensors, -1, -1)
        input_expand_topk = torch.gather(input_expand, 2, self.adj_mx_topk_index_expanded_seq)


        centroids_feature = self.similarity_net_for_centroid_1(self.centroids) + self.similarity_net_for_centroid_2(self.centroids)

        fused_features_expand = self.similarity_net_for_input_1(fushed_features) + self.similarity_net_for_input_2(fushed_features)
        # fused_features_expand = self.bn_feature(fused_features_expand)

        simi_mat = self.fast_cdist(fused_features_expand.view(-1, self.dim_out),centroids_feature)
        simi_mat = simi_mat.view(self.batch_size , self.number_of_sensors, self.gcn_n_class ).softmax(-1)

        # print("33--simi_mat", simi_mat.shape, self.adj_mx_topk_index_expanded.shape)

        simi_mat_expand = simi_mat.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        simi_mat_top_k = torch.gather(simi_mat_expand,2,self.adj_mx_topk_index_expanded_class)
        out_list = []
        for i in range(self.gcn_n_class):
            input_expand_topk_weighted = self.GCN_cluster[i](input_expand_topk)
            simi_mat_top_k_for_i = simi_mat_top_k[..., i].unsqueeze(-1)
            input_expand_topk_weighted = input_expand_topk_weighted*simi_mat_top_k_for_i
            input_expand_topk_weighted = torch.div(input_expand_topk_weighted.sum(-2, keepdim=True), simi_mat_top_k_for_i.sum(-2, keepdim=True))
            out_list.append(input_expand_topk_weighted)

        updated_input = torch.cat(out_list, dim = -2)
        # print("---", updated_input.shape)
        self.centroids = self.minius_update_rate*self.centroids + self.update_rate * updated_input.mean(0).mean(0)
        distance_mat = self.fast_cdist(self.centroids,  self.centroids)
        # print( " distance Mat = ",distance_mat[0,0,0:5,0:5])
        cluster_difference_loss = torch.pow(torch.clamp(self.target - distance_mat, min=0.0), 2)
        cluster_difference_loss = cluster_difference_loss.sum()
        # print("55--simi_mat", simi_mat.shape,input_top_k.shape)
        self.centroids = self.centroids.clone().detach()
        return updated_input, cluster_difference_loss

class clustering_dynamic_learning_common_center_3(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index,margin = 0.5, dim_out = 6, seq_out = 12):
        super(clustering_dynamic_learning_common_center_3, self).__init__()
        device = torch.device(args.device)
        self.dim_out = dim_out
        self.batch_size = args.batch_size
        # self.centroids = nn.Parameter(torch.randn([ args.number_of_sensors,args.gcn_n_class, fused_dim])).to(device)
        self.centroids = (torch.randn([args.gcn_n_class, fused_dim])).to(device)
        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors
        middle_dim = int(fused_dim/2)
        self.similarity_net_for_centroid_1 = nn.Sequential(nn.Linear(fused_dim, middle_dim),
                                            nn.ReLU(),
                                            nn.Linear(middle_dim, dim_out),
                                            nn.ReLU(),   )
        self.similarity_net_for_centroid_2 = nn.Sequential(nn.Linear(fused_dim, dim_out),
                                                         nn.ReLU() )

        self.similarity_net_for_input_1 = nn.Sequential(nn.Linear(fused_dim, middle_dim),
                                                           nn.ReLU(),
                                                           nn.Linear(middle_dim, dim_out),
                                                           nn.ReLU(), )
        self.similarity_net_for_input_2 = nn.Sequential(nn.Linear(fused_dim, dim_out),
                                                           nn.ReLU())

        self.GCN_cluster = nn.ModuleList([nn.Sequential(nn.Linear(args.seq_length_x, seq_out), nn.ReLU()) for i in range(self.gcn_n_class)])
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)
        self.update_rate = torch.tensor(0.01).to(device)
        self.minius_update_rate = torch.tensor(1).to(device) - self.update_rate
        eye_mat = torch.eye(self.gcn_n_class)
        self.target = (torch.ones_like(eye_mat) - eye_mat)*margin
        self.target=  self.target.to(device)
        self.constant_zeros = torch.tensor(0).to(device)

        self.bn_centroid = nn.BatchNorm2d(args.number_of_sensors)
        self.bn_feature = nn.BatchNorm2d(args.number_of_sensors)

    def fast_cdist(self, x1, x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
        # Compute squared distance matrix using quadratic expansion
        # But be clever and do it with a single matmul call
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        # Zero out negative values
        res.clamp_min_(1e-30).sqrt_()
        return res

    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """
        input_expand = input_data.expand(-1, self.number_of_sensors, -1, -1)
        input_expand_topk = torch.gather(input_expand, 2, self.adj_mx_topk_index_expanded_seq)

        centroids_feature = self.similarity_net_for_centroid_1(self.centroids) + self.similarity_net_for_centroid_2(self.centroids)

        fused_features_expand = self.similarity_net_for_input_1(fushed_features) + self.similarity_net_for_input_2(fushed_features)
        # fused_features_expand = self.bn_feature(fused_features_expand)

        simi_mat = self.fast_cdist(fused_features_expand.view(-1, self.dim_out),centroids_feature)
        simi_mat = simi_mat.view(self.batch_size , self.number_of_sensors, self.gcn_n_class ).softmax(-1)

        # print("33--simi_mat", simi_mat.shape, self.adj_mx_topk_index_expanded.shape)
        # print(" simimat = ",  simi_mat[0, :10, :])

        simi_mat = simi_mat.masked_fill(simi_mat < 1/self.gcn_n_class, 0)
        # print(" simimat = ", simi_mat[0, :10, :])
        simi_mat_expand = simi_mat.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        simi_mat_top_k = torch.gather(simi_mat_expand,2,self.adj_mx_topk_index_expanded_class).to(torch.float)
        out_list = []
        for i in range(self.gcn_n_class):
            input_expand_topk_weighted = self.GCN_cluster[i](input_expand_topk)
            simi_mat_top_k_for_i = simi_mat_top_k[..., i].unsqueeze(-1)
            input_expand_topk_weighted = input_expand_topk_weighted*simi_mat_top_k_for_i
            count = simi_mat_top_k_for_i.sum(-2, keepdim=True)
            count = count.masked_fill(count == 0, 1)
            input_expand_topk_weighted = torch.div(input_expand_topk_weighted.sum(-2, keepdim=True), count)
            out_list.append(input_expand_topk_weighted)

        updated_input = torch.cat(out_list, dim = -2)
        # print("---", updated_input.shape)
        self.centroids = self.minius_update_rate*self.centroids + self.update_rate * updated_input.mean(0).mean(0)
        distance_mat = self.fast_cdist(self.centroids,  self.centroids)
        # print( " distance Mat = ",distance_mat[0,0,0:5,0:5])
        cluster_difference_loss = torch.pow(torch.clamp(self.target - distance_mat, min=0.0), 2)
        cluster_difference_loss = cluster_difference_loss.sum()
        # print("55--simi_mat", simi_mat.shape,input_top_k.shape)
        self.centroids = self.centroids.clone().detach()
        return updated_input, cluster_difference_loss

class clustering_dynamic_learning_common_center_4(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index,margin = 0.2, dim_out = 6, seq_out = 12):
        super(clustering_dynamic_learning_common_center_4, self).__init__()
        device = torch.device(args.device)
        self.dim_out = dim_out
        self.batch_size = args.batch_size

        # self.centroids = nn.Parameter(torch.randn([ args.number_of_sensors,args.gcn_n_class, fused_dim])).to(device)
        self.centroids = nn.Parameter( (torch.randn([args.batch_size, args.gcn_n_class, fused_dim],dtype=torch.float))).to(device)
        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors
        middle_dim = int(fused_dim/2)
        self.similarity_net_for_centroid_1 = nn.Sequential(nn.Linear(fused_dim, middle_dim),
                                            nn.ReLU(),
                                            nn.Linear(middle_dim, dim_out),
                                            nn.ReLU(),   )
        self.similarity_net_for_centroid_2 = nn.Sequential(nn.Linear(fused_dim, dim_out),
                                                         nn.ReLU() )

        self.similarity_net_for_input_1 = nn.Sequential(nn.Linear(fused_dim, middle_dim),
                                                           nn.ReLU(),
                                                           nn.Linear(middle_dim, dim_out),
                                                           nn.ReLU(), )
        self.similarity_net_for_input_2 = nn.Sequential(nn.Linear(fused_dim, dim_out),
                                                           nn.ReLU())

        self.GCN_cluster = nn.ModuleList([nn.Sequential(nn.Linear(args.seq_length_x, seq_out), nn.ReLU()) for i in range(self.gcn_n_class)])
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)
        self.update_rate = torch.tensor(0.01).to(device)
        self.minius_update_rate = torch.tensor(1).to(device) - self.update_rate
        eye_mat = torch.eye(self.gcn_n_class)

        self.target_ones = (torch.ones_like(eye_mat) - eye_mat).to(device)
        self.target=  self.target_ones.to(device)*margin
        self.constant_zeros = torch.tensor(0).to(device)

        self.bn_centroid = nn.BatchNorm2d(args.number_of_sensors)
        self.bn_feature = nn.BatchNorm1d(args.number_of_sensors)

    def fast_cdist(self, x1, x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        res.clamp_min_(1e-30).sqrt_()
        return res

    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """
        input_expand = input_data.expand(-1, self.number_of_sensors, -1, -1)
        input_expand_topk = torch.gather(input_expand, 2, self.adj_mx_topk_index_expanded_seq)

        # centroids_feature = self.similarity_net_for_centroid_1(self.centroids) + self.similarity_net_for_centroid_2(self.centroids)
        #
        # fused_features_expand = self.similarity_net_for_input_1(fushed_features) + self.similarity_net_for_input_2(fushed_features)
        # fushed_features = self.bn_feature(fushed_features)

        simi_mat = self.fast_cdist(fushed_features, self.centroids)
        simi_mat = simi_mat.view(self.batch_size , self.number_of_sensors, self.gcn_n_class ).softmax(-1).softmax(-2)

        simi_mat = simi_mat.masked_fill(simi_mat < 1/self.number_of_sensors, 0)

        simi_mat_expand = simi_mat.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        simi_mat_top_k = torch.gather(simi_mat_expand,2,self.adj_mx_topk_index_expanded_class).to(torch.float)
        # print(" simi_mat_top_k ", simi_mat_top_k.shape, simi_mat_top_k[0,:2,:,:])
        out_list = []
        for i in range(self.gcn_n_class):
            input_expand_topk_weighted = self.GCN_cluster[i](input_expand_topk)
            simi_mat_top_k_for_i = simi_mat_top_k[..., i].unsqueeze(-1)
            input_expand_topk_weighted = input_expand_topk_weighted*simi_mat_top_k_for_i
            # count = simi_mat_top_k_for_i.sum(-2, keepdim=True)
            # count = count.masked_fill(count == 0, 1)
            # input_expand_topk_weighted = torch.div(input_expand_topk_weighted.sum(-2, keepdim=True), count)
            out_list.append(input_expand_topk_weighted.sum(-2, keepdim=True))
        updated_input = torch.cat(out_list, dim = -2)

        # # print("---", updated_input.shape)
        # # self.centroids = self.minius_update_rate*self.centroids + self.update_rate * updated_input.mean(0).mean(0)
        # self.centroids_softmax = self.centroids.softmax(-1)
        # self.centroids_y = self.centroids_softmax.unsqueeze(0).expand(self.gcn_n_class, -1,  -1)
        # self.centroids_x = self.centroids_softmax.unsqueeze(1).expand(-1, self.gcn_n_class, -1,)
        # # print(" self.centroids_x  and self.centroids_y", self.centroids_x[0,:,:], self.centroids_y[0,:,:])
        # distance_mat_kl = torch.nn.functional.kl_div(self.centroids_x.log(),  self.centroids_y , reduction='none').sum(-1)
        # # print(" distance_mat  for centroid",distance_mat_kl.shape,distance_mat_kl[0,:])
        # cluster_kl = self.target_ones*distance_mat_kl
        # cluster_kl =  cluster_kl.mean()
        #
        # # print( " distance Mat = ",distance_mat[0,0,0:5,0:5])
        # distance_mat = self.fast_cdist(self.centroids, self.centroids)
        # # print(" distance_mat------------- \n",self.centroids, '\n', distance_mat)
        # cluster_difference_loss = torch.pow(torch.clamp(self.target - distance_mat, min=0.0), 2)
        # cluster_difference_loss = cluster_difference_loss.sum()
        # # print("55--simi_mat", simi_mat.shape,input_top_k.shape)
        # # self.centroids = self.centroids.clone().detach()

        return updated_input, updated_input.mean()

class clustering_dynamic_learning_common_center_5(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index,margin = 1, dim_out = 6, seq_out = 12):
        super(clustering_dynamic_learning_common_center_5, self).__init__()
        device = torch.device(args.device)
        self.dim_out = dim_out
        self.batch_size = args.batch_size

        # self.centroids = nn.Parameter(torch.randn([ args.number_of_sensors,args.gcn_n_class, fused_dim])).to(device)
        self.centroids = nn.Parameter( (torch.randn([args.batch_size, args.gcn_n_class, fused_dim],dtype=torch.float))).to(device)
        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors
        middle_dim = int(fused_dim/2)
        self.similarity_net_for_centroid_1 = nn.Sequential(nn.Linear(fused_dim, fused_dim),
                                            nn.ReLU(),
                                            nn.Linear(fused_dim, fused_dim),
                                            nn.ReLU(),   )
        self.similarity_net_for_centroid_2 = nn.Sequential(nn.Linear(fused_dim, fused_dim),
                                                         nn.ReLU() )

        self.similarity_net_for_input_1 = nn.Sequential(nn.Linear(fused_dim, middle_dim),
                                                           nn.ReLU(),
                                                           nn.Linear(middle_dim, dim_out),
                                                           nn.ReLU(), )
        self.similarity_net_for_input_2 = nn.Sequential(nn.Linear(fused_dim, dim_out),
                                                           nn.ReLU())


        self.GCN_cluster = nn.ModuleList([nn.Sequential(nn.Linear(args.seq_length_x, seq_out), nn.ReLU()) for i in range(self.gcn_n_class)])
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)
        self.update_rate = torch.tensor(0.01).to(device)
        self.minius_update_rate = torch.tensor(1).to(device) - self.update_rate
        eye_mat = torch.eye(self.gcn_n_class)

        self.cluster_results = torch.randn([args.number_of_sensors, args.neighbors_for_each_nodes, args.gcn_n_class], dtype=torch.float).to(device)

        self.target_ones = (torch.ones_like(eye_mat) - eye_mat).to(device)
        self.target=  self.target_ones.to(device)*margin
        self.constant_zeros = torch.tensor(0).to(device)

        self.bn_centroid = nn.BatchNorm2d(args.number_of_sensors)
        self.bn_feature = nn.BatchNorm1d(args.number_of_sensors)

    def fast_cdist(self, x1, x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        res.clamp_min_(1e-30).sqrt_()
        return res

    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """
        input_expand = input_data.expand(-1, self.number_of_sensors, -1, -1)
        input_expand_topk = torch.gather(input_expand, 2, self.adj_mx_topk_index_expanded_seq)

        centroids_feature = self.similarity_net_for_centroid_1(self.centroids) + self.similarity_net_for_centroid_2(self.centroids)
        #
        # fused_features_expand = self.similarity_net_for_input_1(fushed_features) + self.similarity_net_for_input_2(fushed_features)
        fushed_features = self.bn_feature(fushed_features)

        simi_mat = self.fast_cdist(fushed_features, self.centroids)
        simi_mat_averaged = simi_mat.view(self.batch_size , self.number_of_sensors, self.gcn_n_class ).softmax(-1)

        simi_mat_averaged = simi_mat_averaged.masked_fill(simi_mat_averaged < 1/self.gcn_n_class, 0)

        simi_mat_expand = simi_mat_averaged.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        simi_mat_top_k = torch.gather(simi_mat_expand,2,self.adj_mx_topk_index_expanded_class).to(torch.float)
        #
        print(" self.centroids ", self.centroids[0,:,:])
        print(" simi_mat_top_k ", simi_mat_top_k[0, 0, :, :])
        out_list = []
        for i in range(self.gcn_n_class):
            input_expand_topk_weighted = self.GCN_cluster[i](input_expand_topk)
            simi_mat_top_k_for_i = simi_mat_top_k[..., i].unsqueeze(-1)
            input_expand_topk_weighted = input_expand_topk_weighted*simi_mat_top_k_for_i
            count = simi_mat_top_k_for_i.sum(-2, keepdim=True)
            count = count.masked_fill(count == 0, 1)
            input_expand_topk_weighted = torch.div(input_expand_topk_weighted.sum(-2, keepdim=True), count)
            out_list.append(input_expand_topk_weighted.sum(-2, keepdim=True))
        updated_input = torch.cat(out_list, dim = -2)


        cluster_results_expand = self.cluster_results.softmax(-1).unsqueeze(0).expand( self.batch_size, -1, -1, -1)
        distance_mat_kl = torch.nn.functional.kl_div(simi_mat_top_k.softmax(-1).log(),cluster_results_expand, reduction='none').sum(-1).sum(-1)
        cluster_last_step_loss= distance_mat_kl.mean()

        self.cluster_results = self.minius_update_rate*  self.cluster_results + self.update_rate * simi_mat_top_k.mean(0).clone().detach()


        self.centroids_softmax = centroids_feature.softmax(-1)
        self.centroids_y = self.centroids_softmax.unsqueeze(1).expand(-1,self.gcn_n_class, -1,  -1)
        self.centroids_x = self.centroids_softmax.unsqueeze(2).expand(-1, -1, self.gcn_n_class, -1,)

        distance_mat_kl = torch.nn.functional.kl_div(self.centroids_x.log(),  self.centroids_y , reduction='none').sum(-1)
        # print("----shape --",distance_mat_kl.shape, self.target_ones.shape)
        cluster_kl = self.target_ones*distance_mat_kl
        cluster_kl_loss =  cluster_kl.sum(-1).sum(-1).mean()

        distance_mat = self.fast_cdist(self.centroids, self.centroids)
        cluster_difference_loss = torch.pow(torch.clamp(self.target - distance_mat, min=0.0), 2)
        cluster_centroid_loss = cluster_difference_loss.sum(-1).sum(-1).mean()


        return updated_input, [cluster_last_step_loss, cluster_centroid_loss, -1*cluster_kl_loss ]

class clustering_dynamic_learning_common_center_6(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index,margin = 1, dim_out = 6, seq_out = 12):
        super(clustering_dynamic_learning_common_center_6, self).__init__()
        device = torch.device(args.device)
        self.dim_out = dim_out
        self.batch_size = args.batch_size

        # self.centroids = nn.Parameter(torch.randn([ args.number_of_sensors,args.gcn_n_class, fused_dim])).to(device)
        self.centroids = nn.Parameter( (torch.randn([ args.gcn_n_class, fused_dim],dtype=torch.float))).to(device)
        self.cluster_results = torch.randn([args.number_of_sensors, args.neighbors_for_each_nodes, args.gcn_n_class],
                                           dtype=torch.float).to(device)
        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors
        middle_dim = int(fused_dim/2)
        self.similarity_net_for_centroid_1 = nn.Sequential(nn.Linear(fused_dim, fused_dim),
                                            nn.ReLU(),
                                            nn.Linear(fused_dim, fused_dim),
                                            nn.ReLU(),   )
        self.similarity_net_for_centroid_2 = nn.Sequential(nn.Linear(fused_dim, fused_dim),
                                                         nn.ReLU() )

        self.similarity_net_for_input_1 = nn.Sequential(nn.Linear(fused_dim, middle_dim),
                                                           nn.ReLU(),
                                                           nn.Linear(middle_dim, dim_out),
                                                           nn.ReLU(), )
        self.similarity_net_for_input_2 = nn.Sequential(nn.Linear(fused_dim, dim_out),
                                                           nn.ReLU())


        self.GCN_cluster = nn.ModuleList([nn.Sequential(nn.Linear(args.seq_length_x, seq_out, bias = False), nn.ReLU()) for i in range(self.gcn_n_class)])
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)
        self.update_rate = torch.tensor(0.01).to(device)
        self.minius_update_rate = torch.tensor(1).to(device) - self.update_rate
        eye_mat = torch.eye(self.gcn_n_class)
        self.target_ones = (torch.ones_like(eye_mat) - eye_mat).to(device)
        self.target=  self.target_ones.to(device)*margin
        self.constant_zeros = torch.tensor(0).to(device)

        self.bn_centroid = nn.BatchNorm2d(args.number_of_sensors)
        self.bn_feature = nn.BatchNorm1d(args.number_of_sensors)

        self.testing = nn.Sequential(nn.Linear(12, 12),
                                                           nn.ReLU(),
                                                           nn.Linear(12, 12),
                                                           nn.ReLU(), )

    def fast_cdist(self, x1, x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        res.clamp_min_(1e-30).sqrt_()
        return res

    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """
        input_expand = input_data.expand(-1, self.number_of_sensors, -1, -1)
        input_expand_topk = torch.gather(input_expand, 2, self.adj_mx_topk_index_expanded_seq)

        centroids_feature = self.similarity_net_for_centroid_1(self.centroids) + self.similarity_net_for_centroid_2(self.centroids)
        #
        # fused_features_expand = self.similarity_net_for_input_1(fushed_features) + self.similarity_net_for_input_2(fushed_features)
        fushed_features = self.bn_feature(fushed_features)

        simi_mat = self.fast_cdist(fushed_features, self.centroids)
        simi_mat_averaged = simi_mat.view(self.batch_size , self.number_of_sensors, self.gcn_n_class ).softmax(-1)


        simi_mat_averaged_filtered = simi_mat_averaged.masked_fill(simi_mat_averaged < 1/self.gcn_n_class, 0)

        simi_mat_expand = simi_mat_averaged_filtered.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        simi_mat_top_k = torch.gather(simi_mat_expand,2,self.adj_mx_topk_index_expanded_class).to(torch.float)
        simi_mat_top_k_for_loss = simi_mat_top_k.clone()
        #
        # print(" self.centroids ", self.centroids[0,:,:])
        # print(" simi_mat_top_k ", simi_mat_top_k[0, :2, :, :])
        simi_mat_top_k[simi_mat_top_k>0] = 1
        print(" simi_mat_top_k ", simi_mat_top_k[0, 0, :, :])

        out_list = []
        for i in range(self.gcn_n_class):
            input_expand_topk_weighted = self.GCN_cluster[i](input_expand_topk)
            # input_expand_topk_weighted = input_expand_topk
            simi_mat_top_k_for_i = simi_mat_top_k[..., i].unsqueeze(-1)
            # print(" data shape is ", input_expand_topk_weighted.shape, simi_mat_top_k_for_i.shape  )
            # print(" 11 --", i, input_expand_topk_weighted[0,0,:2,:])
            input_expand_topk_weighted = input_expand_topk_weighted*simi_mat_top_k_for_i
            # print(" 22 --", i, input_expand_topk_weighted[0, 0, :2, :])
            count = simi_mat_top_k_for_i.sum(-2, keepdim=True)
            count = count.masked_fill(count == 0, 1)
            input_expand_topk_weighted = torch.div(input_expand_topk_weighted.sum(-2, keepdim=True), count)
            out_list.append(input_expand_topk_weighted)
        updated_input = torch.cat(out_list, dim = -2)
        # print("  concated values = ", updated_input[0,0,:,:6])
        updated_input = updated_input.clone().detach()

        cluster_results_expand = self.cluster_results.softmax(-1).unsqueeze(0).expand( self.batch_size, -1, -1, -1)
        distance_mat_kl = torch.nn.functional.kl_div(simi_mat_top_k_for_loss.softmax(-1).log(),cluster_results_expand.softmax(dim = -1), reduction='none').sum(-1).sum(-1)
        cluster_last_step_loss= distance_mat_kl.mean()

        self.cluster_results = self.minius_update_rate*self.cluster_results + self.update_rate * cluster_results_expand.mean(0).clone().detach()

        self.centroids_softmax = centroids_feature.softmax(-1)
        # print("-- self.centroids_softmax---", self.centroids_softmax.shape)
        self.centroids_y = self.centroids_softmax.unsqueeze(0).expand(self.gcn_n_class, -1,  -1)
        self.centroids_x = self.centroids_softmax.unsqueeze(1).expand( -1, self.gcn_n_class, -1,)

        distance_mat_kl = torch.nn.functional.kl_div(self.centroids_x.log(),  self.centroids_y , reduction='none').sum(-1)
        # print("----shape --",distance_mat_kl.shape, self.target_ones.shape)
        cluster_kl = self.target_ones*distance_mat_kl
        cluster_kl_loss =  cluster_kl.sum(-1).sum(-1).mean()

        distance_mat = self.fast_cdist(self.centroids, self.centroids)
        cluster_difference_loss = torch.pow(torch.clamp(self.target - distance_mat, min=0.0), 2)
        cluster_centroid_loss = cluster_difference_loss.sum(-1).sum(-1).mean()

        # updated_input_1 = self.testing(input_expand_topk).mean(-2,keepdim=True)
        # updated_input_2 = torch.zeros_like(updated_input_1)

        # updated_input = torch.cat([updated_input_1,updated_input_1,updated_input_1], dim = -2)

        return updated_input, [cluster_last_step_loss, cluster_centroid_loss, -1*cluster_kl_loss ]

class clustering_dynamic_learning_common_center_7(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index, margin = 1, dim_out = 6, seq_out = 12):
        super(clustering_dynamic_learning_common_center_7, self).__init__()
        device = torch.device(args.device)
        self.dim_out = dim_out
        self.batch_size = args.batch_size

        # self.centroids = nn.Parameter(torch.randn([ args.number_of_sensors,args.gcn_n_class, fused_dim])).to(device)
        self.centroids = nn.Parameter( (torch.randn([ args.gcn_n_class, fused_dim],dtype=torch.float))).to(device)
        self.cluster_results = tensor_deque(args )
        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors
        middle_dim = int(fused_dim/2)
        self.similarity_net_for_centroid_1 = nn.Sequential(nn.Linear(fused_dim, fused_dim),
                                            nn.ReLU(),
                                            nn.Linear(fused_dim, fused_dim),
                                            nn.ReLU(),   )
        self.similarity_net_for_centroid_2 = nn.Sequential(nn.Linear(fused_dim, fused_dim),
                                                         nn.ReLU() )

        self.similarity_net_for_input_1 = nn.Sequential(nn.Linear(fused_dim, middle_dim),
                                                           nn.ReLU(),
                                                           nn.Linear(middle_dim, dim_out),
                                                           nn.ReLU(), )
        self.similarity_net_for_input_2 = nn.Sequential(nn.Linear(fused_dim, dim_out),
                                                           nn.ReLU())

        self.GCN_cluster = nn.ModuleList([nn.Sequential(nn.Linear(args.seq_length_x, seq_out, bias = False), nn.ReLU()) for i in range(self.gcn_n_class)])
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)
        self.update_rate = torch.tensor(0.01).to(device)
        self.minius_update_rate = torch.tensor(1).to(device) - self.update_rate

        eye_mat = torch.eye(self.gcn_n_class)
        self.target_ones = (torch.ones_like(eye_mat) - eye_mat).to(device)
        self.target_gcn_class=  self.target_ones.to(device)*margin

        eye_mat = torch.eye(self.neighbors_for_each_nodes)
        self.target_ones = (torch.ones_like(eye_mat) - eye_mat).to(device)
        self.target_sensors = self.target_ones.to(device) * margin

        self.constant_zeros = torch.tensor(0).to(device)

        self.bn_centroid = nn.BatchNorm2d(args.number_of_sensors)
        self.bn_feature = nn.BatchNorm1d(args.number_of_sensors)

        self.testing = nn.Sequential(nn.Linear(12, 12),
                                                           nn.ReLU(),
                                                           nn.Linear(12, 12),
                                                           nn.ReLU(), )

    def fast_cdist(self, x1, x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        res.clamp_min_(1e-30).sqrt_()
        return res

    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """
        input_expand = input_data.expand(-1, self.number_of_sensors, -1, -1)
        input_expand_topk = torch.gather(input_expand, 2, self.adj_mx_topk_index_expanded_seq)

        centroids_feature = self.similarity_net_for_centroid_1(self.centroids) + self.similarity_net_for_centroid_2(self.centroids)

        # fused_features_expand = self.similarity_net_for_input_1(fushed_features) + self.similarity_net_for_input_2(fushed_features)
        fushed_features = self.bn_feature(fushed_features)

        simi_mat = self.fast_cdist(fushed_features, self.centroids)
        simi_mat_averaged = simi_mat.view(self.batch_size , self.number_of_sensors, self.gcn_n_class).softmax(-1)

        simi_mat_expand = simi_mat_averaged.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        simi_mat_expand_original = torch.gather(simi_mat_expand, 2, self.adj_mx_topk_index_expanded_class).to(torch.float)
        # print("  simi_mat_expand_original values = ", simi_mat_expand_original[0,0,:,:])
        out_list = []
        for i in range(self.gcn_n_class):
            input_expand_topk_weighted = self.GCN_cluster[i](input_expand_topk)
            # input_expand_topk_weighted = input_expand_topk
            simi_mat_top_k_for_i = simi_mat_expand_original[..., i].unsqueeze(-1)
            # print(" data shape is ", input_expand_topk_weighted.shape, simi_mat_top_k_for_i.shape  )
            # print(" 11 --", i, input_expand_topk_weighted[0,0,:2,:])
            input_expand_topk_weighted = input_expand_topk_weighted*simi_mat_top_k_for_i
            # print(" 22 --", i, input_expand_topk_weighted[0, 0, :2, :])
            count = simi_mat_top_k_for_i.sum(-2, keepdim=True)
            count = count.masked_fill(count == 0, 1)
            input_expand_topk_weighted = torch.div(input_expand_topk_weighted.sum(-2, keepdim=True), count)
            out_list.append(input_expand_topk_weighted)
        updated_input = torch.cat(out_list, dim = -2)
        # print("  concated values = ", updated_input[0,0,:,:6])
        a = simi_mat_expand_original.argmax(-1)
        mask_values = torch.nn.functional.one_hot(a,  num_classes= self.gcn_n_class).float()
        # print("  simi_mat_expand_original values = ", mask_values[0,0,:,:])
        cur_cluster_result = mask_values.mean(0)
        # print("----shape---", cur_cluster_result.shape)
        cluster_result_gt = self.cluster_results(cur_cluster_result.clone().detach())
        cluster_result_gt_expand = cluster_result_gt.softmax(-1).unsqueeze(0).expand(self.batch_size, -1, -1, -1)
        distance_mat_kl = torch.nn.functional.kl_div(simi_mat_expand_original.softmax(-1).log(),cluster_result_gt_expand, reduction='none').sum(-1).sum(-1)
        cluster_last_step_loss= distance_mat_kl.mean()


        distance_mat = self.fast_cdist(self.centroids,self.centroids)
        cluster_difference_loss = torch.pow(torch.clamp(self.target_gcn_class - distance_mat, min=0.0), 2)
        cluster_centroid_loss = cluster_difference_loss.sum(-1).sum(-1).mean()

        distance_mat = self.fast_cdist(simi_mat_expand_original, simi_mat_expand_original)
        cluster_neighbor_loss = torch.pow(torch.clamp(self.target_sensors - distance_mat, min=0.0), 2)
        cluster_neighbor_loss = cluster_neighbor_loss.sum(-1).sum(-1).mean()

        return updated_input, [cluster_last_step_loss, cluster_centroid_loss, cluster_neighbor_loss]

class clustering_dynamic_learning_common_center_8(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index, margin = 1, dim_out = 6, seq_out = 12):
        super(clustering_dynamic_learning_common_center_8, self).__init__()
        device = torch.device(args.device)
        self.dim_out = dim_out
        self.batch_size = args.batch_size

        # self.centroids = nn.Parameter(torch.randn([ args.number_of_sensors,args.gcn_n_class, fused_dim])).to(device)
        self.centroids = nn.Parameter( (torch.randn([ args.gcn_n_class, fused_dim],dtype=torch.float))).to(device)
        self.cluster_results = tensor_deque(args )
        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors
        middle_dim = int(fused_dim/2)
        self.similarity_net_for_centroid_1 = nn.Sequential(nn.Linear(fused_dim, fused_dim),
                                            nn.ReLU(),
                                            nn.Linear(fused_dim, fused_dim),
                                            nn.ReLU(),   )
        self.similarity_net_for_centroid_2 = nn.Sequential(nn.Linear(fused_dim, fused_dim),
                                                         nn.ReLU() )

        self.similarity_net_for_input_1 = nn.Sequential(nn.Linear(fused_dim, middle_dim),
                                                           nn.ReLU(),
                                                           nn.Linear(middle_dim, dim_out),
                                                           nn.ReLU(), )
        self.similarity_net_for_input_2 = nn.Sequential(nn.Linear(fused_dim, dim_out),
                                                           nn.ReLU())

        self.GCN_cluster = nn.ModuleList([nn.Sequential(nn.Linear(args.seq_length_x, seq_out, bias = False), nn.ReLU()) for i in range(self.gcn_n_class)])
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)
        self.update_rate = torch.tensor(0.01).to(device)
        self.minius_update_rate = torch.tensor(1).to(device) - self.update_rate

        eye_mat = torch.eye(self.gcn_n_class)
        self.target_ones = (torch.ones_like(eye_mat) - eye_mat).to(device)
        self.target_gcn_class=  self.target_ones.to(device)*margin

        eye_mat = torch.eye(self.neighbors_for_each_nodes)
        self.target_ones = (torch.ones_like(eye_mat) - eye_mat).to(device)
        self.target_sensors = self.target_ones.to(device) * margin

        self.constant_zeros = torch.tensor(0).to(device)

        self.bn_centroid = nn.BatchNorm2d(args.number_of_sensors)
        self.bn_feature = nn.BatchNorm1d(args.number_of_sensors)

        self.testing = nn.Sequential(nn.Linear(12, 12),    nn.ReLU(),
                                                           nn.Linear(12, 12),
                                                           nn.ReLU(), )

    def fast_cdist(self, x1, x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        res.clamp_min_(1e-30).sqrt_()
        return res

    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """
        input_expand = input_data.expand(-1, self.number_of_sensors, -1, -1)
        input_expand_topk = torch.gather(input_expand, 2, self.adj_mx_topk_index_expanded_seq)
        out_list = []
        for i in range(self.gcn_n_class):
            begin = int(self.neighbors_for_each_nodes / self.gcn_n_class)
            end = np.minimum((i+1)*begin, self.neighbors_for_each_nodes)
            cur_data = input_expand_topk[:,:, i*begin:end,:].mean(-2, keepdim=True)
            # print("111---", cur_data.shape)
            out_list.append(cur_data)
        updated_input = torch.cat(out_list, dim=-2)
        # print("222---", updated_input.shape)

        return updated_input, [updated_input.mean(), updated_input.mean(), updated_input.mean()]


