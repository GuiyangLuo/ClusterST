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
import seaborn as sns
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

        self.spatial_temporal = Spatial_temporal(args)
        self.global_terater = 0

        if args.centroid_way =="attention":
            self.dynamic_cluster = clustering_attention(args, args.seq_length_x, adj_mx_topk_index)
        if args.centroid_way == "attention_multihead":
            self.dynamic_cluster = clustering_attention_multi_head(args, args.seq_length_x, adj_mx_topk_index)
        if args.centroid_way == "cluster_0":
            self.dynamic_cluster = clustering_attention_dynamic_learning0(args, args.seq_length_x, adj_mx_topk_index, writer = writer)
        if args.centroid_way == "cluster_1":
            self.dynamic_cluster = clustering_attention_dynamic_learning1(args,  args.seq_length_x, adj_mx_topk_index, writer = writer)
        if args.centroid_way == "cluster_2":
            self.dynamic_cluster = clustering_attention_dynamic_learning2(args, args.seq_length_x, adj_mx_topk_index,
                                                                          writer=writer)
        if args.centroid_way == "cluster_3":
            self.dynamic_cluster = clustering_attention_dynamic_learning3(args, args.seq_length_x, adj_mx_topk_index,
                                                                          writer=writer)
        # if args.centroid_way == "cluster_3":
        #     self.dynamic_cluster = clustering_attention_dynamic_learning3(args, args.seq_length_x, adj_mx_topk_index,
        #                                                                   writer=writer)
        # if args.centroid_way == "cluster_4":
        #     self.dynamic_cluster = clustering_attention_dynamic_learning4(args, args.seq_length_x, adj_mx_topk_index,
        #                                                                   writer=writer)
        # if args.centroid_way == "cluster_5":
        #     self.dynamic_cluster = clustering_attention_dynamic_learning5(args, args.seq_length_x, adj_mx_topk_index,
        #                                                                   writer=writer)
        # if args.centroid_way == "cluster_6":
        #     self.dynamic_cluster = clustering_attention_dynamic_learning6(args, args.seq_length_x, adj_mx_topk_index,
        #                                                                   writer=writer)
        # if args.centroid_way == "cluster_7":
        #     self.dynamic_cluster = clustering_attention_dynamic_learning7(args, args.seq_length_x, adj_mx_topk_index,
        #                                                                   writer=writer)
        # if args.centroid_way == "cluster_8":
        #     self.dynamic_cluster = clustering_attention_dynamic_learning8(args, args.seq_length_x, adj_mx_topk_index,
        #                                                                   writer=writer)
        # self.visualization_cluster = Visualization_Cluster(args, writer, road_sensor_pos)

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

        clustred_output, cluster_difference_loss = self.dynamic_cluster(input_original, input_original)

        weightedDinput_topk = clustred_output.unsqueeze(3).contiguous()

        output = self.spatial_temporal(weightedDinput_topk)

        return output, cluster_difference_loss

class clustering_attention(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index, margin = 1, alpha = 0.5,  dim_out = 6, seq_out = 12, writer = None ):
        super(clustering_attention, self).__init__()
        device = torch.device(args.device)
        self.writer = writer
        self.global_terater = 0
        self.dim_out = dim_out
        self.batch_size = args.batch_size

        self.centroids = nn.Parameter( (torch.randn([ args.gcn_n_class, fused_dim],dtype=torch.float))).to(device)

        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors

        self.GCN_cluster = nn.ModuleList([nn.Sequential(nn.Linear(args.seq_length_x, seq_out, bias = False), nn.ReLU()) for i in range(self.gcn_n_class)])
        self.adj_mx_topk_index = adj_mx_topk_index
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)

        self.W = nn.Parameter(torch.empty(size=(args.seq_length_x, seq_out)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * seq_out, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

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
    def vis_cluster_result(self, attention_cluster, print_every=100, ):
        if self.writer and self.global_terater % print_every == 0:
            timeseries = attention_cluster.detach().cpu().numpy()
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            fig = plt.figure(figsize=(10,5))
            data = timeseries[0,0,:,:]
            sns.heatmap(data.transpose(), linewidths=0.05, annot=True, cmap="RdBu_r")
            self.writer.add_figure('time_series_data',
                                   fig,
                                   global_step=self.global_terater,
                                   )
    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """
        input_data = input_data.squeeze()
        wh = torch.matmul(input_data, self.W)
        wh_x = wh.unsqueeze(2).expand(-1, -1, self.number_of_sensors, -1)
        wh_y = wh.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        wh_concat = torch.cat([wh_x, wh_y], dim=-1)
        attention = self.leakyrelu(torch.matmul(wh_concat, self.a))
        # print(" attention", attention.shape, self.adj_mx_topk_index.shape)
        attention_mask = torch.gather(attention,2, self.adj_mx_topk_index)
        attention_mask = attention_mask.softmax(dim = -1)

        input_expand = input_data.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        input_expand_topk = torch.gather(input_expand, 2, self.adj_mx_topk_index_expanded_seq)
        output_data = attention_mask.unsqueeze(-1)*input_expand_topk

        return output_data, [output_data.mean(), output_data.mean(), output_data.mean()]

class clustering_attention_multi_head(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index, margin=1, alpha=0.5, dim_out=6, seq_out=12):
        super(clustering_attention_multi_head, self).__init__()
        self.gcn_n_class = args.gcn_n_class
        self.clustering_attention = nn.ModuleList([clustering_attention(args, fused_dim, adj_mx_topk_index) for i in range(args.gcn_n_class)])

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
        out_list = []
        for i in range(self.gcn_n_class):
            output, _ = self.clustering_attention[i](fushed_features, input_data)
            out_list.append(output.mean(dim = -2, keepdim= True))
        output_data = torch.cat(out_list, dim = -2)
        # print("-output_data-", output_data.shape)
        return output_data, [output_data.mean(), output_data.mean(), output_data.mean()]

class clustering_attention_dynamic_learning0(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index, margin = 1, alpha = 0.5,  dim_out = 6, seq_out = 12, writer = None ):
        super(clustering_attention_dynamic_learning0, self).__init__()
        device = torch.device(args.device)
        self.writer = writer
        self.global_terater = 0
        self.dim_out = dim_out
        self.batch_size = args.batch_size

        self.centroids = nn.Parameter( (torch.randn([ args.gcn_n_class, fused_dim],dtype=torch.float))).to(device)

        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors

        self.GCN_cluster = nn.ModuleList([nn.Sequential(nn.Linear(args.seq_length_x, seq_out, bias = False), nn.ReLU()) for i in range(self.gcn_n_class)])
        self.adj_mx_topk_index = adj_mx_topk_index
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)

        self.W = nn.Parameter(torch.empty(size=(args.seq_length_x, seq_out)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * seq_out, self.gcn_n_class)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

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

    def vis_cluster_result(self, attention_cluster, print_every=300, ):
        self.global_terater = self.global_terater + 1
        if self.writer and self.global_terater % print_every == 0:
            timeseries = attention_cluster.detach().cpu().numpy()
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            fig = plt.figure(figsize=(10, 5))
            data = timeseries[0, 0, :, :]
            sns.heatmap(data.transpose(), linewidths=0.05,fmt ='.2f', annot=True, cmap="RdBu_r")
            self.writer.add_figure('time_series_data',
                                   fig,
                                   global_step=self.global_terater,
                                   )
    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """
        input_data = input_data.squeeze()
        wh = torch.matmul(input_data, self.W)
        wh_x = wh.unsqueeze(2).expand(-1,-1,self.number_of_sensors,-1)
        wh_y = wh.unsqueeze(1).expand(-1,  self.number_of_sensors,-1, -1)
        wh_concat = torch.cat([wh_x,wh_y],dim = -1)
        attention = self.leakyrelu(torch.matmul(wh_concat, self.a))
        # print(" attention", attention.shape, self.adj_mx_topk_index.shape)
        attention_mask = torch.gather(attention, 2, self.adj_mx_topk_index_expanded_class)
        attention_mask = attention_mask.softmax(dim = -1)
        self.vis_cluster_result(attention_mask)
        attention_mask_transpose = attention_mask.transpose(-1,-2)

        input_expand = input_data.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        input_expand_topk = torch.gather(input_expand, 2, self.adj_mx_topk_index_expanded_seq)

        output_data = torch.matmul(attention_mask_transpose, input_expand_topk)
        # print("shape---", attention_mask_transpose.shape, input_expand_topk.shape)
        return output_data, [output_data.mean(), output_data.mean(), output_data.mean()]

class clustering_attention_dynamic_learning1(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index, margin = 1, alpha = 0.5,  dim_out = 6, seq_out = 12, writer = None ):
        super(clustering_attention_dynamic_learning1, self).__init__()
        device = torch.device(args.device)
        self.sim_metric = args.simi_metric
        self.writer = writer
        self.global_terater = 0
        self.dim_out = dim_out
        self.batch_size = args.batch_size

        self.centroids = nn.Parameter( (torch.randn([ args.gcn_n_class, fused_dim],dtype=torch.float))).to(device)

        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors

        self.GCN_cluster = nn.ModuleList([nn.Sequential(nn.Linear(args.seq_length_x, seq_out, bias = False), nn.ReLU()) for i in range(self.gcn_n_class)])
        self.adj_mx_topk_index = adj_mx_topk_index
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)



        self.w_net = nn.Sequential(nn.Linear(args.seq_length_x, seq_out),
                                   nn.LeakyReLU(alpha)
        )
        self.a_net = nn.Sequential(nn.Linear(2 * seq_out, 4 * seq_out),
                                   nn.LeakyReLU(alpha),
                                   nn.Linear(4 * seq_out, self.gcn_n_class),
                                   nn.LeakyReLU(alpha)
                                   )

        self.bth = nn.BatchNorm2d(1)

    @staticmethod
    def cos_sim(a, b):
        """
          Compute cosine similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """
          Compute euclidean similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        return -2 * a @ b.transpose(-2, -1) + (a ** 2).sum(dim=-1)[..., :, None] + (b ** 2).sum(dim=-1)[..., None, :]

    def vis_cluster_result(self, prob, distance, print_every=300, ):
        self.global_terater = self.global_terater + 1
        if self.writer and self.global_terater % print_every == 0:
            prob_data = prob.detach().cpu().numpy()
            distance_data = distance.detach().cpu().numpy()
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            fig,  (ax1,ax2)= plt.subplots(figsize=(10,10),nrows=2)
            prob_data = prob_data[0,0,:,:]
            distance_data = distance_data[0,0,:,:]
            sns.heatmap(prob_data, linewidths=0.05, ax = ax1, fmt ='.2f', annot=True, cmap="RdBu_r")
            sns.heatmap(distance_data.transpose(), linewidths=0.05,  ax = ax2,fmt='.2f', annot=True, cmap="RdBu_r")
            self.writer.add_figure('time_series_data',
                                   fig,
                                   global_step=self.global_terater,
                                   )

    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """

        # input_data_expand = input_data.squeeze().unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        # input_data_expand_expand_top_k = torch.gather(input_data_expand, 2, self.adj_mx_topk_index_expanded_seq)
        # print(" --input --", input_data_expand_expand_top_k[0,0,:,:])
        input_data = input_data.squeeze()
        # print(100*'*')
        # print(" input_data ---- = ", input_data.max(), input_data.min())
        wh = self.w_net(input_data)

        wh_x = wh.unsqueeze(2).expand(-1,-1,self.number_of_sensors,-1)
        wh_y = wh.unsqueeze(1).expand(-1,  self.number_of_sensors,-1, -1)
        wh_concat = torch.cat([wh_x,wh_y],dim = -1)

        attention = self.a_net(wh_concat)
        attention_mask = torch.gather(attention, 2, self.adj_mx_topk_index_expanded_class)
        attention_mask = attention_mask.softmax(dim = -1)
        attention_mask_transpose = attention_mask.transpose(-1,-2)
        wh_expand = wh.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        wh_expand_top_k = torch.gather(wh_expand, 2, self.adj_mx_topk_index_expanded_seq)
        output_data = torch.matmul(attention_mask_transpose, wh_expand_top_k)
        # print("shape---", attention_mask_transpose.shape, input_expand_topk.shape)
        ############------------------Loss for clustering---------------################
        # based on attention_mask, whose shape is (batch, sensors, neighbors, class )
        # print(" 0000---- = ", wh_expand_top_k.max(), wh_expand_top_k.min())
        wh_expand_top_k = wh_expand_top_k.detach()

        if self.sim_metric == 'euc':
            dist_mat = self.euc_sim(wh_expand_top_k, wh_expand_top_k)
            dist_mat_same = dist_mat.le(0.2).float()
        elif self.sim_metric == 'cos':
            dist_mat = self.cos_sim(wh_expand_top_k, wh_expand_top_k)
            dist_mat_same = dist_mat.ge(0.5).float()

        dist_mat_diff = 1 - dist_mat_same
        # print(" 111---- = ", dist_mat_same.max(), dist_mat_same.min(), dist_mat_diff.max(), dist_mat_diff.min())
        probability = torch.matmul(attention_mask, attention_mask_transpose)
        # print(" --attention_mask-- ",  probability[0,0,:,:])
        probability = probability.clamp(min = 1e-4, max = 1  -1e-4)
        self.vis_cluster_result( attention_mask, dist_mat)
        # print(" 222---- = ", probability.max(), probability.min(), probability.mean())
        cluster_loss = torch.mul(dist_mat_same, probability.log()) - torch.mul(dist_mat_diff,probability.log())
        # print(" 445454---- = ",  probability[0, 0, :, :],  probability.log()[0, 0, :, :],  cluster_loss[0, 0, :, :])
        cluster_loss = cluster_loss*-1
        # cluster_loss = nn.functional.binary_cross_entropy(probability, dist_mat.detach(),reduction="none")

        cluster_loss = (cluster_loss.sum(-1).sum(-1) -  torch.einsum('...ii',cluster_loss)).mean()
        # print(" cluster_loss = ", cluster_loss )
        # print(" attention_mask = ", attention_mask[0,0,:,:])
        # dist_mat = (dist_mat+1).softmax(-1).clamp(min=1e-4, max=1 - 1e-4)
        # entropy_loss = torch.mul(dist_mat,  dist_mat.log()).sum(-1).mean()

        return output_data, [cluster_loss, dist_mat.mean(), wh.mean()]

class clustering_attention_dynamic_learning2(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index, margin = 1, alpha = 0.5,  dim_out = 6, seq_out = 12, writer = None ):
        super(clustering_attention_dynamic_learning2, self).__init__()
        device = torch.device(args.device)
        self.sim_metric = args.simi_metric
        self.writer = writer
        self.global_terater = 0
        self.dim_out = dim_out
        self.batch_size = args.batch_size

        self.centroids = nn.Parameter( (torch.randn([ args.gcn_n_class, fused_dim],dtype=torch.float))).to(device)

        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors

        self.adj_mx_topk_index = adj_mx_topk_index
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)
        self.adj_mx_topk_index_expanded_doubleseq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                     args.seq_length_x*2)
        self.w_net = nn.Sequential(nn.Linear(args.seq_length_x, seq_out, bias= False), )
        self.a_net = nn.Sequential(nn.Linear(2 * seq_out, self.gcn_n_class,  bias= False),
                                   )
        self.distance_net  =   nn.Sequential(
                    nn.Linear(2 * args.seq_length_x,   12, bias=True),
                    nn.LeakyReLU(),
                    nn.Linear(12, 6, bias=True),
                    nn.LeakyReLU(), )

        self.bth_input = nn.BatchNorm2d(1)
        self.bth = nn.BatchNorm2d(1)
        self.bth_attention = nn.BatchNorm2d(207)
        self.loss_cluster_func =  nn.KLDivLoss(reduce=False)

    @staticmethod
    def cos_sim(a, b):
        """
          Compute cosine similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """
          Compute euclidean similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        return -2 * a @ b.transpose(-2, -1) + (a ** 2).sum(dim=-1)[..., :, None] + (b ** 2).sum(dim=-1)[..., None, :]

    def vis_cluster_result(self,attention,  prob, distance, wconcat,weight, print_every=50, ):

        self.global_terater = self.global_terater + 1
        if self.writer and self.global_terater % print_every == 0:
            attention = attention.detach().cpu().numpy()
            prob_data = prob.detach().cpu().numpy()
            distance_data = distance.detach().cpu().numpy()
            wconcat = wconcat.detach().cpu().numpy()
            weight = weight.detach().cpu().numpy()
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            fig,  (ax0, ax1,ax2, ax3, ax4)= plt.subplots(figsize=(10,10),nrows=5)
            attention_data = attention[0,0,:,:]
            prob_data = prob_data[0,0,:,:]
            distance_data = distance_data[0,0,:,:]
            wconcat = wconcat[0,0,:,:]
            weight =  weight[0,0,:,:]
            sns.heatmap(attention_data, linewidths=0.05, ax=ax0, fmt='.2f', annot=True, cmap="RdBu_r")
            sns.heatmap(prob_data.transpose(-1,-2), linewidths=0.05, ax = ax1, fmt ='.2f', annot=True, cmap="RdBu_r")
            sns.heatmap(distance_data.transpose(), linewidths=0.05,  ax = ax2,fmt='.2f', annot=True, cmap="RdBu_r")
            sns.heatmap(wconcat, linewidths=0.05, ax=ax3, fmt='.2f', annot=True, cmap="RdBu_r")
            sns.heatmap(weight, linewidths=0.05, ax=ax4, fmt='.2f', annot=True, cmap="RdBu_r")
            self.writer.add_figure('time_series_data',
                                   fig,
                                   global_step=self.global_terater,
                                   )

    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """

        wh = self.w_net(input_data.squeeze())
        wh_x = wh.unsqueeze(2).expand(-1,-1,self.number_of_sensors,-1)
        wh_y = wh.unsqueeze(1).expand(-1,  self.number_of_sensors,-1, -1)
        wh_concat = torch.cat([wh_x,wh_y],dim = -1)

        attention = self.a_net(wh_concat)
        attention = torch.gather(attention, 2, self.adj_mx_topk_index_expanded_class)
        attention_mask = attention.softmax(dim = -2)
        attention_mask_transpose = attention_mask.transpose(-1,-2)

        # input_data_expand = input_data.expand(-1, self.number_of_sensors, -1, -1)
        # input_data_expand_top_k = torch.gather(input_data_expand, 2, self.adj_mx_topk_index_expanded_seq)
        # output_data = torch.matmul(attention_mask_transpose, input_data_expand_top_k)

        wh_expand = wh.unsqueeze(1).expand(-1, self.number_of_sensors, -1, -1)
        wh_expand_top_k = torch.gather(wh_expand, 2, self.adj_mx_topk_index_expanded_seq)
        output_data = torch.matmul(attention_mask_transpose, wh_expand_top_k)

        ############------------------Loss for clustering---------------################
        # based on attention_mask, whose shape is (batch, sensors, neighbors, class )
        ###  First loss
        wh_concat_detach = wh_concat.detach() # wh_expand_top_k.detach()  or  input_data_expand_top_k
        wh_concat_topk =  torch.gather(wh_concat_detach, 2, self.adj_mx_topk_index_expanded_doubleseq)
        wh_concat_topk = self.distance_net(wh_concat_topk)
        if self.sim_metric == 'euc':
            dist_mat = self.euc_sim(wh_concat_topk, wh_concat_topk)
        elif self.sim_metric == 'cos':
            dist_mat = self.cos_sim(wh_expand_top_k, wh_expand_top_k)
            dist_mat = 1 - dist_mat

        # print("000000--shape--", attention_mask.min(), attention_mask.max(),attention_mask.mean())

        attention_mask_i = attention.unsqueeze(-2).expand(-1,-1,-1,self.neighbors_for_each_nodes,-1)
        attention_mask_j = attention.unsqueeze(-3).expand(-1, -1, self.neighbors_for_each_nodes, -1, -1)
        prob_i_j = torch.mul(attention_mask_i, attention_mask_j)
        prob_i_j = prob_i_j.softmax(-1)
        # prob_i_j = torch.where(prob_i_j<0.09, torch.zeros_like(prob_i_j), prob_i_j)
        # print("111--shape--", attention_mask[0,0,:,:], prob_i_j[0,0,:2,:2,:])
        prob_dist = torch.mul(dist_mat.unsqueeze(-1), prob_i_j)
        prob_dist_sum = prob_dist.permute(0,1,4,2,3).triu(diagonal=1)
        prob_i_j_sum = prob_i_j.permute(0,1,4,2,3).triu(diagonal=1)
        prob_i_j_sum = prob_i_j_sum.clamp(min = 1e-5)
        prob_i_j = prob_dist_sum.sum(-1).sum(-1) / prob_i_j_sum.sum(-1).sum(-1)
        prob_i_j = prob_i_j.sum(-1)
        cluster_loss = prob_i_j.mean()


        ##
        # attention_trans_sft = attention.transpose(-1, -2).softmax(-1)
        # cluster_features = torch.matmul(attention_trans_sft, wh_concat_topk)
        dist_mat_cluster = self.euc_sim(output_data, output_data)
        cluster_center_loss = torch.pow(torch.clamp(5 - dist_mat_cluster, min = 0),2).triu_(diagonal=1)
        cluster_center_loss = cluster_center_loss.sum(-1).sum(-1).mean()

        self.vis_cluster_result(attention_mask_transpose, output_data.transpose(-1, -2), dist_mat, dist_mat_cluster,wh_concat_topk )


        return output_data, [ cluster_loss, cluster_center_loss, wh.mean(), attention.mean(), dist_mat.mean()]

class clustering_attention_dynamic_learning3(nn.Module):
    def __init__(self, args, fused_dim, adj_mx_topk_index, margin = 1, alpha = 0.5,  dim_out = 6, seq_out = 12, writer = None ):
        super(clustering_attention_dynamic_learning3, self).__init__()
        device = torch.device(args.device)
        self.sim_metric = args.simi_metric
        self.writer = writer
        self.global_terater = 0
        self.dim_out = dim_out
        self.batch_size = args.batch_size

        self.gcn_n_class = args.gcn_n_class
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors

        self.adj_mx_topk_index = adj_mx_topk_index
        self.adj_mx_topk_index_expanded_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.adj_mx_topk_index_expanded_seq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                        args.seq_length_x)
        self.adj_mx_topk_index_expanded_doubleseq = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                     args.seq_length_x*2)
        self.transform_net = nn.Sequential(
                                    nn.Linear(args.seq_length_x*self.neighbors_for_each_nodes, 64, bias= False),
                                    nn.LeakyReLU(),
                                    nn.Linear( 64,  self.neighbors_for_each_nodes*self.gcn_n_class, bias=False),
                                   )

        self.distance_net  =   nn.Sequential(
                    nn.Linear( args.seq_length_x,   8, bias=True),
                    nn.LeakyReLU(),
                    nn.Linear(8, 6, bias=True),
                    nn.LeakyReLU(), )

        self.bth_input = nn.BatchNorm2d(1)
        self.bth = nn.BatchNorm2d(1)
        self.bth_attention = nn.BatchNorm2d(207)
        self.loss_cluster_func =  nn.KLDivLoss(reduce=False)

    @staticmethod
    def cos_sim(a, b):
        """
          Compute cosine similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """
          Compute euclidean similarity of 2 sets of vectors
          Parameters:
          a: torch.Tensor, shape: [m, n_features]
          b: torch.Tensor, shape: [n, n_features]
        """
        return -2 * a @ b.transpose(-2, -1) + (a ** 2).sum(dim=-1)[..., :, None] + (b ** 2).sum(dim=-1)[..., None, :]

    def vis_cluster_result(self,attention,  prob, distance, wconcat,weight, print_every=50, ):

        self.global_terater = self.global_terater + 1
        if self.writer and self.global_terater % print_every == 0:
            attention = attention.detach().cpu().numpy()
            prob_data = prob.detach().cpu().numpy()
            distance_data = distance.detach().cpu().numpy()
            wconcat = wconcat.detach().cpu().numpy()
            weight = weight.detach().cpu().numpy()
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            fig,  (ax0, ax1,ax2, ax3, ax4)= plt.subplots(figsize=(10,10),nrows=5)
            attention_data = attention[0,0,:,:]
            prob_data = prob_data[0,0,:,:]
            distance_data = distance_data[0,0,:,:]
            wconcat = wconcat[0,0,:,:]
            weight =  weight[0,0,:,:]
            sns.heatmap(attention_data, linewidths=0.05, ax=ax0, fmt='.2f', annot=True, cmap="RdBu_r")
            sns.heatmap(prob_data.transpose(-1,-2), linewidths=0.05, ax = ax1, fmt ='.2f', annot=True, cmap="RdBu_r")
            sns.heatmap(distance_data.transpose(), linewidths=0.05,  ax = ax2,fmt='.2f', annot=True, cmap="RdBu_r")
            sns.heatmap(wconcat, linewidths=0.05, ax=ax3, fmt='.2f', annot=True, cmap="RdBu_r")
            sns.heatmap(weight, linewidths=0.05, ax=ax4, fmt='.2f', annot=True, cmap="RdBu_r")
            self.writer.add_figure('time_series_data',
                                   fig,
                                   global_step=self.global_terater,
                                   )

    def forward(self, fushed_features, input_data):
        """
        :param fused_features: (batch, sensor_node,   dim_feature)
        :param input_top_k: (batch, sensor_node,  in_sequence)
        :return: (batch, sensor_node, c, in_sequence)
        """

        input_data_expand = input_data.expand(-1, self.number_of_sensors, -1, -1)
        input_data_expand_top_k = torch.gather(input_data_expand, 2, self.adj_mx_topk_index_expanded_seq)

        input_data_expand_top_k_viewed = input_data_expand_top_k.view(self.batch_size, self.number_of_sensors, -1)

        input_data_expand_top_k_out = self.transform_net(input_data_expand_top_k_viewed)

        attention = input_data_expand_top_k_out.view(self.batch_size, self.number_of_sensors, self.neighbors_for_each_nodes, self.gcn_n_class)
        attention_mask = attention.softmax(dim = -2)
        attention_mask_transpose = attention_mask.transpose(-1,-2)


        # output_data = torch.matmul(attention_mask_transpose, input_data_expand_top_k)


        output_data = torch.matmul(attention_mask_transpose, input_data_expand_top_k)

        ############------------------Loss for clustering---------------################
        # based on attention_mask, whose shape is (batch, sensors, neighbors, class )
        ###  First loss
        wh_concat_detach = input_data_expand_top_k.detach() # wh_expand_top_k.detach()  or  input_data_expand_top_k
        wh_concat_topk = self.distance_net(wh_concat_detach)
        if self.sim_metric == 'euc':
            dist_mat = self.euc_sim(wh_concat_topk, wh_concat_topk)
        elif self.sim_metric == 'cos':
            dist_mat = self.cos_sim(wh_concat_topk, wh_concat_topk)
            dist_mat = 1 - dist_mat

        # print("000000--shape--", attention_mask.min(), attention_mask.max(),attention_mask.mean())

        attention_mask_i = attention.unsqueeze(-2).expand(-1,-1,-1,self.neighbors_for_each_nodes,-1)
        attention_mask_j = attention.unsqueeze(-3).expand(-1, -1, self.neighbors_for_each_nodes, -1, -1)
        prob_i_j = torch.mul(attention_mask_i, attention_mask_j)
        prob_i_j = prob_i_j.softmax(-1)
        # prob_i_j = torch.where(prob_i_j<0.09, torch.zeros_like(prob_i_j), prob_i_j)
        # print("111--shape--", attention_mask[0,0,:,:], prob_i_j[0,0,:2,:2,:])
        prob_dist = torch.mul(dist_mat.unsqueeze(-1), prob_i_j)
        prob_dist_sum = prob_dist.permute(0,1,4,2,3).triu(diagonal=1)
        prob_i_j_sum = prob_i_j.permute(0,1,4,2,3).triu(diagonal=1)
        prob_i_j_sum = prob_i_j_sum.clamp(min = 1e-5)
        prob_i_j = prob_dist_sum.sum(-1).sum(-1) / prob_i_j_sum.sum(-1).sum(-1)
        prob_i_j = prob_i_j.sum(-1)
        cluster_loss = prob_i_j.mean()


        ##
        # attention_trans_sft = attention.transpose(-1, -2).softmax(-1)
        # cluster_features = torch.matmul(attention_trans_sft, wh_concat_topk)

        cluster_features = torch.matmul(attention_mask_transpose, wh_concat_topk)
        dist_mat_cluster = self.euc_sim(cluster_features, cluster_features)
        cluster_center_loss = torch.pow(torch.clamp(1 - dist_mat_cluster, min = 0),2).triu_(diagonal=1)
        cluster_center_loss = cluster_center_loss.sum(-1).sum(-1).mean()

        self.vis_cluster_result(attention_mask_transpose, output_data.transpose(-1, -2), dist_mat, dist_mat_cluster,wh_concat_topk )


        return output_data, [ cluster_loss, cluster_center_loss, attention.mean(), attention.mean(), dist_mat.mean()]












