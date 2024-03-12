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
from  model.VraeModel import VRAE as VRAE_Model


class Caculate_parameters_conv():

    def __init__(self, number_neighbors, seq_input_x, max_allow_spatial_conv = 3, max_allow_dilation = 2, weight = 'std'):
        super(Caculate_parameters_conv, self).__init__()
        self.number_neighbors = number_neighbors
        self.seq_input_x = seq_input_x
        self.max_allow_spatial_conv = max_allow_spatial_conv
        self.max_allow_dilation = max_allow_dilation
        self.weight =  weight

    def main(self):
        w = []
        b = []
        v = []
        bags = self.constructed_bags()
        for bag in bags:
            w.append(bag[4])
            b.append(bag[5])
            if self.weight == 'std':
                v.append(- np.array([bag[4],bag[5]]).std())
            elif  self.weight == 'mean':
                v.append(- np.array([bag[4], bag[5]]).mean())

        x_list = self.interger_programming_conv(w,b,v,self.number_neighbors - 1, self.seq_input_x - 1  )
        final_convs = []
        for index,x in enumerate(x_list):
            x = int(x)
            [final_convs.append(bags[index]) for i in range(x)]
        np.random.shuffle(final_convs)
        return final_convs



    def constructed_bags(self):
        bags = []
        for ker1 in range(1,self.max_allow_spatial_conv+1):
            for ker2 in range(1,self.max_allow_spatial_conv+1):
                for dila1 in range(1, self.max_allow_dilation + 1):
                    for dila2 in range(1, self.max_allow_dilation + 1):
                        if ker1 < dila1 or ker2 < dila2:
                            continue
                        weig1 = dila1 * (ker1 - 1)
                        weig2 = dila2 * (ker2 - 1)
                        bag = (ker1, ker2, dila1, dila2, weig1, weig2)
                        bags.append(bag)

        def compare_func(x):
            return  max(x[0],x[1]) * 10 +  min(x[0],x[1])
        bags = sorted(bags, key = compare_func, reverse=True)
        return bags

    def interger_programming_conv(self,w,b, v, neighbors,seq_in):
        n = len(w)

        c = np.array(v)

        a = np.array([w , b]).reshape(2,-1)

        # 输入b值（3×1）
        b = np.array([neighbors,seq_in])

        # 创建x，个数是3
        x = cp.Variable(n, integer=True)

        # 明确目标函数（此时c是3×1，x是3×1,但python里面可以相乘）
        objective = cp.Maximize(cp.sum(c * x))

        # 明确约束条件，其中a是3×3，x是3×1,a*x=b(b为3×1的矩阵)

        constriants = [0 <= x, a * x == b]
        # 求解问题
        prob = cp.Problem(objective, constriants)

        resluts = prob.solve(solver=cp.CPLEX)

        return x.value

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

    def forward(self, ):
        return self.static_nets(self.road_sensor_pos)


class Dynamic_features(nn.Module):

    def __init__(self, args):
        super(Dynamic_features, self).__init__()
        # accepted input should be shape of :  [seq_len, batch, input_size]
        self.gru = nn.GRU(args.in_dims, args.dynamic_gru_feature_size, args.gru_number_of_layers, bidirectional  = args.gru_bidirectional)
        self.gru_bidirectional = 1 if args.gru_bidirectional is False else 2
        # print("  self.gru_bidirectional = ", self.gru_bidirectional, )
        self.h_0 =(torch.randn(args.gru_number_of_layers*self.gru_bidirectional, args.number_of_sensors *  args.batch_size, args.dynamic_gru_feature_size))
        device = torch.device(args.device)
        self.h_0 = torch.Tensor(self.h_0).to(device)


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
        output, hidden_0 = self.gru(input,self.h_0)  # (num_layers * num_directions, batch, hidden_size)

        hidden_0 = hidden_0[-self.gru_bidirectional:,:,:]
        # print(" Dynamic_features =  ", hidden_0.shape)
        hidden_0 = hidden_0.permute(1,0,2).contiguous().view(batch_size,num_sensors,-1).contiguous()
        # print(" Dynamic_features =  ", hidden_0.shape )
        return hidden_0


class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel = 1 , output_channel = 256, resnet_layers = [1, 2, 2, 2], neighbors = 12 ):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, resnet_layers, neighbors)

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

    def __init__(self, input_channel, output_channel, block, layers, neighbors):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv3d(input_channel, int(output_channel / 16),
                                 kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)
        self.bn0_1 = nn.BatchNorm3d(int(output_channel / 16))
        self.conv0_2 = nn.Conv3d(int(output_channel / 16), self.inplanes,
                                 kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)
        self.bn0_2 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv3d(self.output_channel_block[0], self.output_channel_block[
                               0], kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)
        self.bn1 = nn.BatchNorm3d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv3d(self.output_channel_block[1], self.output_channel_block[
                               1], kernel_size=(1,2,2), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(self.output_channel_block[1])

        if neighbors in [8,10]:
            self.maxpool3 = nn.MaxPool3d(kernel_size=(1,1,2), stride=(1,1, 2), padding=(0,0, 1))
            self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
            self.conv3 = nn.Conv3d(self.output_channel_block[2], self.output_channel_block[
                2], kernel_size=(1, 1, 2), stride=1, padding=(0, 0, 0), bias=False)
            self.bn3 = nn.BatchNorm3d(self.output_channel_block[2])
        elif neighbors in [12, 14, 16]:
            self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
            self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
            self.conv3 = nn.Conv3d(self.output_channel_block[2], self.output_channel_block[
                2], kernel_size=(1, 2, 2), stride=1, padding=(0, 0, 0), bias=False)
            self.bn3 = nn.BatchNorm3d(self.output_channel_block[2])
        # elif neighbors in [18]:
        #     self.maxpool3 = nn.Sequential(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1)),
        #                                   self._make_layer(block, self.output_channel_block[2], 1,
        #                                                    stride=1),
        #                                   )
        #     self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        #     self.conv3 = nn.Conv3d(self.output_channel_block[2], self.output_channel_block[
        #         2], kernel_size=(1, 2, 2), stride=1, padding=(0, 0, 0), bias=False)
        #     self.bn3 = nn.BatchNorm3d(self.output_channel_block[2])


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

        self.spatial_temporal_feature_extractor = ResNet_FeatureExtractor(input_channel = channels_conv_start_layer , output_channel = channels_conv_extractor_out,  resnet_layers = args.resnet_layers, neighbors = args.gcn_n_class)

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
        # print("shape of input ",threeDinput.shape)

        output =  self.start_conv(threeDinput)
        # print("shape of start_conv", output.shape  )
        output = self.spatial_temporal_feature_extractor(output)
        # print("shape of spatial_temporal_feature_extractor", output.shape)

        output = self.final_convs(output)

        output = output.squeeze(-1)
        # output shape (batch_size, in_seq, #edges, in_dim)
        return output
# Custom Contrastive Loss

class EmbedGCN(nn.Module):

    def __init__(self, args,  road_sensor_pos, adj_mx_topk_index,  writer =None):
        super(EmbedGCN, self).__init__()
        self.args = args
        self.road_sensor_pos = road_sensor_pos
        self.adj_mx_topk_index = adj_mx_topk_index
        self.top_k_values = args.top_k_neighbors
        self.writer = writer
        self.static_feature_module = Static_features(args, road_sensor_pos, road_sensor_pos.shape[-1] ,args.static_feature_size)
        self.dynamic_feature_module = Dynamic_features(args)
        self.spatial_temporal = Spatial_temporal(args)
        self.global_terater = 0
        self.contrastiveLoss = ContrastiveLoss(args.neighbors_for_each_nodes, adj_mx_topk_index , margin=0.1)

        if self.args.fusion == 'dynamic':
            self.feature_size = args.dynamic_gru_feature_size
        elif self.args.fusion == 'static':
            self.feature_size = args.static_feature_size
        elif self.args.fusion == 'concat':
            self.feature_size = args.static_feature_size + args.dynamic_gru_feature_size
        else:
            assert 1 == 0, f"fusion mechanism {self.args.fusion} is not defined!!"
        self.feature_norm1d = nn.BatchNorm1d(args.number_of_sensors)
        self.nodes_clustering_learn = Nodes_Clustering_Learn(self.feature_size, args.gcn_n_class, args.number_of_sensors)
        self.nodes_clustering_euclidean = Nodes_Clustering_Euclidean(args.gcn_n_class)

        self.cluster_GCN = Ordered_GCN( args,  args.seq_length_x, args.seq_length_x)

        ### for VREA
        self.vrae_net = VRAE_Model(args.seq_length_y, args.in_dims, args.vrae_hidden_size, args.vrae_hidden_layer_depth, args.vrae_latent_length, args.batch_size*args.number_of_sensors, block = args.vrae_rnn_model, dropout_rate = args.vrae_dropout_rate, device = torch.device(args.device) )

    def cosine_simularity(self, x):
        ## shape of x is (batch_size,  # sensors,fused_dim)
        x = x.permute((1, 2, 0))
        cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
        return cos_sim_pairwise
    def forward(self, input):
        self.global_terater =   self.global_terater + 1
        # input shape (batch_size, in_dim, #edges, in_seq)
        # output shape (batch_size, in_seq, #edges, in_dim)
        # print("input  = ", input.shape,input[2,:, 122:125,:5])

        input_for_vrae = input.clone().detach()
        input_for_vrae = input_for_vrae.permute(3, 0, 2, 1).view(self.args.seq_length_x, -1, self.args.in_dims)
        total_loss, latent_mean = self.vrae_net(input_for_vrae)
        # print("total_loss, latent_mean  ",total_loss.shape, latent_mean.shape)
        fushed_features = latent_mean.view(self.args.batch_size,self.args.number_of_sensors,-1)

        adjacency_mat = self.cosine_simularity(fushed_features)  # shape is (batch_size, #sensors, #sensors)

        if self.args.dist_metric == 'cosine':
            nodes_pairs_classes_details = self.nodes_clustering_learn(fushed_features)  ###shape: (batch,sensors, sensors, gcn_n_class)
            _, clustered_index = nodes_pairs_classes_details.max(dim=-1)
        else:
            adjacency_mat = (adjacency_mat + 1)/2
            clustered_index = self.nodes_clustering_euclidean(adjacency_mat)
            clustered_index = clustered_index.to(torch.long)   ###shape: (batch,sensors, sensors)

        # print(" adjacency_mat",adjacency_mat[0,0,:])
        input = input.expand(-1,  self.args.number_of_sensors, -1,-1)  # input shape (batch_size, #edges, #edges,in_seq)

        # print("input and index  = ", input.shape, adjacency_mat_topk_indexes.shape,adjacency_mat_topk_indexes[:,:1,:])
        adjacency_mat_values = adjacency_mat.unsqueeze(-1).expand(-1, -1, -1, self.args.seq_length_x)
        weightedDinput = torch.mul(input, adjacency_mat_values)
        clustered_index_topk = torch.gather(clustered_index, 2, self.adj_mx_topk_index)
        if True:
            ### considering the fixed number neighbors for each nodes!!!
            adj_mx_topk_index_expanded = self.adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.args.seq_length_x)  #
            weightedDinput_topk = torch.gather(weightedDinput, 2, adj_mx_topk_index_expanded)
            # print("--clustered_index_topk",clustered_index_topk[0,0,:])
            threeDinput = self.cluster_GCN(clustered_index_topk, weightedDinput_topk)
        else:
            threeDinput = self.cluster_GCN(clustered_index, weightedDinput)

        threeDinput = threeDinput.unsqueeze(3).contiguous() # shape of threeDinput is (batch_size, number_sensors, neighbors, 1, in-seq)

        output = self.spatial_temporal(threeDinput)
        # output = output.view(batch_size, number_sensors, -1, seq_in)
        # output = output.permute(0, 3, 1, 2)
        # contrastiveLoss_out = self.contrastiveLoss(clustered_index_topk, adjacency_mat)
        return output, total_loss

class EmbedGCN1(nn.Module):

    def __init__(self, args,  road_sensor_pos, adj_mx_topk_index,  writer =None):
        super(EmbedGCN1, self).__init__()
        self.args = args
        self.road_sensor_pos = road_sensor_pos
        self.adj_mx_topk_index = adj_mx_topk_index
        self.top_k_values = args.top_k_neighbors
        self.writer = writer
        self.static_feature_module = Static_features(args, road_sensor_pos, road_sensor_pos.shape[-1] ,args.static_feature_size)
        self.dynamic_feature_module = Dynamic_features(args)
        self.spatial_temporal = Spatial_temporal(args)
        self.global_terater = 0
        self.contrastiveLoss = ContrastiveLoss(args.neighbors_for_each_nodes, adj_mx_topk_index , margin=0.1)

        if self.args.fusion == 'dynamic':
            self.feature_size = args.dynamic_gru_feature_size
        elif self.args.fusion == 'static':
            self.feature_size = args.static_feature_size
        elif self.args.fusion == 'concat':
            self.feature_size = args.static_feature_size + args.dynamic_gru_feature_size
        else:
            assert 1 == 0, f"fusion mechanism {self.args.fusion} is not defined!!"
        self.feature_norm1d = nn.BatchNorm1d(args.number_of_sensors)
        self.nodes_clustering_learn = Nodes_Clustering_Learn(self.feature_size, args.gcn_n_class, args.number_of_sensors)
        self.nodes_clustering_euclidean = Nodes_Clustering_Euclidean(args.gcn_n_class)

        self.cluster_GCN = Ordered_GCN( args,  args.seq_length_x, args.seq_length_x)

        ### for VREA
        self.vrae_net = VRAE_Model(args.seq_length_y, args.in_dims, args.vrae_hidden_size, args.vrae_hidden_layer_depth, args.vrae_latent_length, args.batch_size*args.number_of_sensors, block = args.vrae_rnn_model, dropout_rate = args.vrae_dropout_rate, device = torch.device(args.device) )

    def cosine_simularity(self, x):
        ## shape of x is (batch_size,  # sensors,fused_dim)
        x = x.permute((1, 2, 0))
        cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
        return cos_sim_pairwise
    def forward(self, input):
        self.global_terater =   self.global_terater + 1
        # input shape (batch_size, in_dim, #edges, in_seq)
        # output shape (batch_size, in_seq, #edges, in_dim)
        # print("input  = ", input.shape,input[2,:, 122:125,:5])
        input_for_dynamic_features = input.clone().detach()
        dynamic_fea = self.dynamic_feature_module(input_for_dynamic_features)
        static_fea =  self.static_feature_module()

        if self.args.fusion == 'dynamic':
            fushed_features = dynamic_fea
        elif  self.args.fusion == 'static':
            fushed_features = static_fea
        elif  self.args.fusion == 'concat':
            fushed_features = torch.cat([static_fea, dynamic_fea], axis=2)  # shape is (batch_size,#sensors,fused_dim)
        else:
            assert 1 == 0,  f"fusion mechanism {self.args.fusion} is not defined!!"
        fushed_features = self.feature_norm1d(fushed_features)
        ####------- fushed_features for clustering!!!-----
        adjacency_mat = self.cosine_simularity(fushed_features)  # shape is (batch_size, #sensors, #sensors)

        if self.args.dist_metric == 'cosine':
            nodes_pairs_classes_details = self.nodes_clustering_learn(fushed_features)  ###shape: (batch,sensors, sensors, gcn_n_class)
            _, clustered_index = nodes_pairs_classes_details.max(dim=-1)
        else:
            adjacency_mat = torch.abs(adjacency_mat)
            clustered_index = self.nodes_clustering_euclidean(adjacency_mat)
            clustered_index = clustered_index.to(torch.long)   ###shape: (batch,sensors, sensors)

        # print(" adjacency_mat",adjacency_mat[0,0,:])
        input = input.expand(-1,  self.args.number_of_sensors, -1,-1)  # input shape (batch_size, #edges, #edges,in_seq)

        # print("input and index  = ", input.shape, adjacency_mat_topk_indexes.shape,adjacency_mat_topk_indexes[:,:1,:])
        adjacency_mat_values = adjacency_mat.unsqueeze(-1).expand(-1, -1, -1, self.args.seq_length_x)
        weightedDinput = torch.mul(input, adjacency_mat_values)
        clustered_index_topk = torch.gather(clustered_index, 2, self.adj_mx_topk_index)
        if True:
            ### considering the fixed number neighbors for each nodes!!!

            adj_mx_topk_index_expanded = self.adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.args.seq_length_x)  #
            weightedDinput_topk = torch.gather(weightedDinput, 2, adj_mx_topk_index_expanded)
            threeDinput = self.cluster_GCN(clustered_index_topk, weightedDinput_topk)
        else:
            threeDinput = self.cluster_GCN(clustered_index, weightedDinput)

        threeDinput = threeDinput.unsqueeze(3).contiguous() # shape of threeDinput is (batch_size, number_sensors, neighbors, 1, in-seq)

        output = self.spatial_temporal(threeDinput)
        # output = output.view(batch_size, number_sensors, -1, seq_in)
        # output = output.permute(0, 3, 1, 2)
        contrastiveLoss_out = self.contrastiveLoss(clustered_index_topk, adjacency_mat)
        return output, contrastiveLoss_out

class Nodes_Clustering_Learn(nn.Module):

    def __init__(self, dim_feature,  number_class, number_of_sensors,):
        super(Nodes_Clustering_Learn, self).__init__()

        self.number_of_sensors = number_of_sensors
        self.linearKey = nn.Linear(dim_feature, dim_feature, bias=True)
        self.linearQuery = nn.Linear(dim_feature, dim_feature, bias=True)
        self.MapingToClass = nn.Linear(dim_feature, number_class, bias=True)
        self.softmax = nn.Softmax(dim = 3)


    def forward(self, features):
        # features input shape (number_of_sensors, dim_feature)
        key = self.linearKey(features)
        query = self.linearQuery(features)
        # print(" key ", key[0,1,:])
        # print(" query ", query[0, 2, :])
        key_new = torch.unsqueeze(key, 2)
        key_expand = key_new.expand(-1,-1,self.number_of_sensors,-1)
        # print(" 11 shape of key ", key_new.shape, key_expand.shape , key[0,1,:])
        query_new = torch.unsqueeze(query, 1)
        query_expand = query_new.expand(-1, self.number_of_sensors,-1,  -1)
        # print(" 22 shape of query ", query.shape, query_expand.shape , query[0,2,:])
        # print("  key_expand + query_expand ",  (key_expand + query_expand)[0, 1,2,  :])

        key_query = torch.tanh(key_expand + query_expand)
        # print(" 33 shape of key_expand + query_expand ", key_query[0,1, 2, :])
        # print("  key_query ", key_query[0, 1, 2, :])
        key_query_classes = self.MapingToClass(key_query)
        # print("  key_query_classes ", key_query_classes[0, 1, 2, :])
        key_query_classes = self.softmax(key_query_classes)
        # print(" shape of query key_query_classes ", key_query_classes.shape, key_query_classes[0,0,:10,:])
        return key_query_classes

class Nodes_Clustering_Euclidean(nn.Module):

    def __init__(self, gcn_n_class,):
        super(Nodes_Clustering_Euclidean, self).__init__()
        self.gcn_n_class = gcn_n_class
    def forward(self, adjacency_mat):
        adjacency_mat = adjacency_mat*self.gcn_n_class - 0.00001
        return torch.floor(adjacency_mat)

class Ordered_GCN(nn.Module):

    def __init__(self, args,   seq_in_len, seq_out_len):
        super(Ordered_GCN, self).__init__()
        self.number_class = args.gcn_n_class
        self.number_of_sensors = args.number_of_sensors
        self.linears = nn.ModuleList([nn.Linear(seq_in_len, seq_out_len, bias=False) for _ in range(self.number_class)])
        self.non_linear_act = nn.ModuleList([nn.Tanh() for _ in range(self.number_class)])
        self.activation_func = nn.ReLU()
        self.seq_out_len = seq_out_len
        self.seq_in_len = seq_in_len
        self.args = args
        device = torch.device(args.device)
        self.constant_zeros = torch.zeros([1]).to(device)
        self.constant_ones = torch.ones([1]).to(device)

    def forward(self, clustered_index_topk,  weightedDinput_topk):
        # print("----------",class_for_sensor_index.shape, input.shape)
        key_query_class_index = clustered_index_topk.to(torch.float32)
        # print("-1111111--------key_query_class_index",key_query_class_index[0,0,:5])
        class_for_sensor_index = torch.unsqueeze(key_query_class_index, -1)
        key_query_class_index_expand = class_for_sensor_index.expand(-1, -1, -1, self.seq_in_len)
        # print("-22222--------key_query_class_index_expand", key_query_class_index_expand[0, 0, :5,:])

        out_list = []

        for i, net_block in enumerate(self.linears):

            threeDinput = torch.where(key_query_class_index_expand == i, weightedDinput_topk,  self.constant_zeros)

            class_count = torch.eq(key_query_class_index, i).to(torch.float32).sum(dim=-1, keepdim=True)
            class_count = torch.where(class_count == self.constant_zeros, self.constant_ones, class_count)

            threeDinput = torch.div(threeDinput.sum(dim=2), class_count)
            # print(i,"input ---33  ", threeDinput.mean())
            threeDinput = self.non_linear_act[i](net_block(threeDinput))
            # print(i,"input ---44  ", threeDinput.mean())
            out_list.append(threeDinput)

        out_values = torch.stack(out_list,dim = 2)
        # print("  out_values ", out_values.shape)
        return out_values

class ContrastiveLoss(nn.Module):

    def __init__(self, neighbors_for_each_nodes, adj_mx_topk_index , margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.neighbors_for_each_nodes = neighbors_for_each_nodes
        self.margin = margin
        self.adj_mx_topk_index = adj_mx_topk_index  # shape: (batch, sensors, neighbors_for_each_nodes)

    def forward(self, clustered_ids_topk, adjacency_mat,):
        """
        :param self:
        :param fused_features: shape: (batch, sensors,  dim_feature)
        :param clustered_index_topk: shape: (batch, sensors,  neighbors_for_each_nodes)
        :param adjacency_mat: shape: (batch, sensors, sensors)
        :return:
        """
        # print(" clustered Ids ", clustered_ids[0,0,:] )
        clustered_ids_filtered = clustered_ids_topk
        adjacency_dist_filtered = 1 - torch.abs(torch.gather(adjacency_mat, 2, self.adj_mx_topk_index))
        # print("111----adjacency_dist_filtered = ", adjacency_dist_filtered.shape, adjacency_dist_filtered[0,0,:])
        adjacency_dist_filtered = adjacency_dist_filtered.unsqueeze(2).expand(-1,-1,self.neighbors_for_each_nodes, -1)
        # print("222----adjacency_dist_filtered = ", adjacency_dist_filtered.shape, adjacency_dist_filtered[0, 0, :2,:])
        clustered_ids_filtered_1 = clustered_ids_filtered.unsqueeze(2).expand(-1, -1, self.neighbors_for_each_nodes, -1)
        clustered_ids_filtered_2 = clustered_ids_filtered.unsqueeze(-1).expand(-1, -1, -1, self.neighbors_for_each_nodes)
        # print("333----clustered_ids_filtered_1 = ", clustered_ids_filtered_1.shape, clustered_ids_filtered_1[0, 0, :1, :])
        # print("444----clustered_ids_filtered_2 = ", clustered_ids_filtered_2.shape, clustered_ids_filtered_2[0, 0, :2, :])

        label = torch.eq(clustered_ids_filtered_1,clustered_ids_filtered_2).to(torch.float32)
        loss_contrastive = torch.mean((1 - label) * adjacency_dist_filtered +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - adjacency_dist_filtered, min=0.0), 2))
        return loss_contrastive
