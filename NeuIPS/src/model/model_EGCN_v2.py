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
        self.adj_mx_topk_index_expand_featuresize = self.adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                                self.feature_size)
        # self.feature_norm1d = nn.BatchNorm1d(args.number_of_sensors)
        # self.nodes_clustering_learn = Nodes_Clustering_Learn(self.feature_size, args.gcn_n_class, args.number_of_sensors)

        self.cluster_GCN = Ordered_GCN( args,  args.seq_length_x, args.seq_length_x)
        # self.contrastiveLoss = ContrastiveLoss(args, args.neighbors_for_each_nodes, adj_mx_topk_index, self.feature_size, margin=0.1)
        self.clustering = K_means_clustering(args)
        self.visualization_cluster = Visualization_Cluster(args, writer, road_sensor_pos)
        self.bn2d = nn.InstanceNorm2d( args.number_of_sensors)


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

        input_for_dynamic_features = input.clone().detach()

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


        fushed_features_expand = fushed_features.unsqueeze(1).expand(-1,self.args.number_of_sensors,-1,-1)
        fushed_features_topk = torch.gather(fushed_features_expand, 2, self.adj_mx_topk_index_expand_featuresize)
        fushed_features_topk= self.bn2d(fushed_features_topk)
        clustered_index_topk, centers, loss_k_means, uniform_loss= self.clustering(fushed_features_topk)

        if  self.args.neighbors_for_each_nodes >1:
            self.visualization_cluster.vis_figure(fushed_features_topk[0,0,:,:], clustered_index_topk[0,0,:], self.adj_mx_topk_index[0,0,:])

        if True:
            input = input[:, :1, :, :]
            weightedDinput = input.expand(-1, self.args.number_of_sensors, -1,   -1)

            adj_mx_topk_index_expanded = self.adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,
                                                                                     self.args.seq_length_x)
            weightedDinput_topk = torch.gather(weightedDinput, 2, adj_mx_topk_index_expanded)
            weightedDinput_topk = self.cluster_GCN(clustered_index_topk, weightedDinput_topk)
            weightedDinput_topk = weightedDinput_topk.unsqueeze(3).contiguous()

            output = self.spatial_temporal(weightedDinput_topk)


        return output , loss_k_means



class ContrastiveLoss(nn.Module):
    ## Best Margin is 3
    def __init__(self, args, neighbors_for_each_nodes, adj_mx_topk_index , feature_size, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.neighbors_for_each_nodes = neighbors_for_each_nodes
        self.number_of_sensors = args.number_of_sensors
        self.gcn_n_class = args.gcn_n_class
        self.margin = margin
        self.adj_mx_topk_index = adj_mx_topk_index  # shape: (batch, sensors, neighbors_for_each_nodes)
        self.adj_mx_topk_index_feature = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1,feature_size )
        self.batch_size = args.batch_size
        self.adj_mx_topk_index_class = adj_mx_topk_index.unsqueeze(-1).expand(-1, -1, -1, self.gcn_n_class)
        self.target = torch.ones([args.batch_size,args.number_of_sensors,args.gcn_n_class]).to(torch.device(args.device) )/args.gcn_n_class
        self.klloss = nn.KLDivLoss(size_average=False, reduce=False)

        device = torch.device(args.device)
        self.constant_zeros = torch.zeros([1]).to(device)
        self.constant_ones = torch.ones([1]).to(device)

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
    def forward(self, fusedFeatures, nodes_pairs_classes_details):
        """
        :param adjacency_mat: (batch, sensor, sensor)
        :param nodes_pairs_classes_details:  (batch, sensor, sensor, class)
        :return:
        """
        fusedFeatures_expanded =  fusedFeatures.unsqueeze(1).expand(-1,self.number_of_sensors,  -1, -1)
        fusedFeatures_expanded = torch.gather(fusedFeatures_expanded, 2, self.adj_mx_topk_index_feature)
        fusedFeatures_expanded = fusedFeatures_expanded.view(self.batch_size*self.number_of_sensors , self.neighbors_for_each_nodes, -1)
        # print("fusedFeatures_expanded  ",fusedFeatures_expanded.shape)
        distance_mat = self.fast_cdist(fusedFeatures_expanded,fusedFeatures_expanded)
        distance_mat = distance_mat.view(self.batch_size,self.number_of_sensors, self.neighbors_for_each_nodes, self.neighbors_for_each_nodes)
        ########################
        # print(" nodes_pairs_classes_details ",nodes_pairs_classes_details[0,0,0,:])
        nodes_pairs_classes_details_topk = torch.gather(nodes_pairs_classes_details, 2, self.adj_mx_topk_index_class)
        nodes_pairs_classes_details_topk_latter = nodes_pairs_classes_details_topk.transpose(2,3)
        labels = torch.einsum('lmik,lmkj->lmij', [nodes_pairs_classes_details_topk,nodes_pairs_classes_details_topk_latter])
        positive = torch.mul(labels, torch.pow(distance_mat,2))
        negtive = (1 - labels) * torch.pow(
            torch.clamp(self.margin - distance_mat, min=0.0), 2)
        loss_contrastive = torch.sum( (positive + negtive)/2, dim = -1)
        loss_contrastive = torch.sum(loss_contrastive, dim=-1)
        loss_contrastive = loss_contrastive.mean()

        ############### Method 000


        ## The First!!!
        labels = torch.einsum('lmik,lmkj->lmijk',
                              [nodes_pairs_classes_details_topk, nodes_pairs_classes_details_topk_latter])
        # labels_eachclass = labels.sum(dim = 2).sum(dim = 2)
        # # print("----", labels_eachclass.shape, labels_eachclass[0,:2,:])
        # expected_number = self.neighbors_for_each_nodes/self.gcn_n_class*  self.neighbors_for_each_nodes/self.gcn_n_class
        # uniform_loss = torch.pow(labels_eachclass-expected_number ,2).sum(dim = -1).mean()
        # maxed_values = labels_eachclass.softmax(dim=-1)
        # maxed_values_entropy = torch.mul(maxed_values,torch.log(maxed_values))
        # uniform_loss = maxed_values_entropy.sum(dim = -1)
        # uniform_loss = uniform_loss.mean()

        ###The Second

        nodes_pairs_classes,_ = torch.max(labels, -1, keepdim=True)
        # print(" nodes_pairs_classes ===== ", labels[0, 0, 0, :5, :])
        labels = torch.where( labels==nodes_pairs_classes, labels , self.constant_zeros )
        # print(" labels ===== ", labels[0,0,0,:5,:] )
        labels_eachclass = labels.sum(dim=2).sum(dim=2)
        expected_number = self.neighbors_for_each_nodes / self.gcn_n_class * self.neighbors_for_each_nodes / self.gcn_n_class
        uniform_loss = torch.pow(labels_eachclass - expected_number, 2).sum(dim=-1).mean()



        ############### Method 111
        # nodes_pairs_classes_details_topk = nodes_pairs_classes_details_topk.sum(dim=2).softmax(dim = -1)
        # maxed_values_entropy = torch.mul(nodes_pairs_classes_details_topk,torch.log(nodes_pairs_classes_details_topk))
        # uniform_loss = maxed_values_entropy.sum(dim = -1)
        # uniform_loss = uniform_loss.mean()

        ############### Method 222
        # nodes_pairs_classes,_ = torch.max(nodes_pairs_classes_details_topk, -1, keepdim=True)
        # maxed_values = torch.where( nodes_pairs_classes_details_topk==nodes_pairs_classes,self.constant_ones , self.constant_zeros  )
        # # print("maxed_values ",maxed_values.shape, maxed_values[0,0,0,:])
        # maxed_values = maxed_values.sum(dim=2).softmax(-1)
        # # print("maxed_values ",maxed_values.shape, maxed_values[0,0,:])
        # maxed_values_entropy = torch.mul(maxed_values,torch.log(maxed_values))
        # uniform_loss = maxed_values_entropy.sum(dim = -1)
        # uniform_loss = uniform_loss.mean()

        ############### Method 333
        # for i in range(self.gcn_n_class):
        #     count = torch.eq(nodes_pairs_classes_index,i).to(torch.float32).sum(-1)
        #     print(" count ",i, count[0,0])
        #     count = torch.pow(count - self.neighbors_for_each_nodes/self.gcn_n_class,2)
        #     uniform_loss.append(count)
        # uniform_loss = torch.cat(uniform_loss).mean()

        # uniform_loss = labels.sum(dim = -1)
        # uniform_loss = uniform_loss.std(dim = -1)
        #
        # uniform_loss = uniform_loss.mean()
        print("loss---", loss_contrastive, uniform_loss)


        return [loss_contrastive, uniform_loss]

class Nodes_Clustering_Learn(nn.Module):

    def __init__(self, dim_feature,  number_class, number_of_sensors,):
        super(Nodes_Clustering_Learn, self).__init__()

        self.number_of_sensors = number_of_sensors
        self.linearKey = nn.Sequential(nn.Linear(dim_feature, dim_feature, bias=True),nn.ReLU())
        self.linearQuery = nn.Sequential(nn.Linear(dim_feature, dim_feature, bias=True),nn.ReLU())
        self.MapingToClass = nn.Sequential(nn.Linear(dim_feature, number_class, bias=True))
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

        key_query = torch.tanh(self.linearKey(key_expand) + self.linearQuery (query_expand))
        # print(" 33 shape of key_expand + query_expand ", key_query[0,1, 2, :])
        # print("  key_query ", key_query[0, 1, 2, :])
        key_query_classes = self.MapingToClass(key_query)
        # print("  key_query_classes ", key_query_classes[0, 1, 2, :])
        key_query_classes = self.softmax(key_query_classes)
        # print(" shape of query key_query_classes ", key_query_classes.shape, key_query_classes[0,0,:10,:])
        return key_query_classes

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
        key_query_class_index = clustered_index_topk.to(torch.float32)
        class_for_sensor_index = torch.unsqueeze(key_query_class_index, -1)
        key_query_class_index_expand = class_for_sensor_index.expand(-1, -1, -1, self.seq_in_len)
        out_list = []
        for i, net_block in enumerate(self.linears):
            threeDinput = torch.where(key_query_class_index_expand == i, weightedDinput_topk,  self.constant_zeros)
            class_count = torch.eq(key_query_class_index, i).to(torch.float32).sum(dim=-1, keepdim=True)
            class_count = torch.clamp(class_count, 1)
            threeDinput = torch.div(threeDinput.sum(dim=2), class_count)
            threeDinput = self.non_linear_act[i](net_block(threeDinput))
            out_list.append(threeDinput)

        out_values = torch.stack(out_list,dim = 2)
        return out_values

class K_means_clustering(nn.Module):
    def __init__(self, args):
        super(K_means_clustering, self).__init__()
        ## dist can be euclidean  or cosine
        self.kmeans = MultiKMeans(n_clusters=args.gcn_n_class, n_kmeans=args.batch_size * args.number_of_sensors, device =torch.device(args.device) , mode='euclidean', verbose=0)
        self.batch_size = args.batch_size
        self.number_of_sensors = args.number_of_sensors
        self.neighbors_for_each_nodes = args.neighbors_for_each_nodes
        self.gcn_n_class = args.gcn_n_class

    def forward(self, feature):
        """
        :param feature: (batch, sensors, neighbors, dim)
        :return: closest (batch, sensors, )
        :return: centers (batch, sensors, self.gcn_n_class, dim)
        :return: loss
        """
        feature = feature.view(self.batch_size*self.number_of_sensors, self.neighbors_for_each_nodes, -1)
        cluster_id, centers, kmeans_loss, uniform_loss = self.kmeans.fit_predict(feature)
        centers = centers.view(self.batch_size,self.number_of_sensors,  self.gcn_n_class , -1)
        cluster_id = cluster_id.view(self.batch_size,self.number_of_sensors,self.neighbors_for_each_nodes)

        return cluster_id, centers, kmeans_loss, uniform_loss
