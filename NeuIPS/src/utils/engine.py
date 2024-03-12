import torch.optim as optim
import torch
import os
import torch.nn as nn

import numpy as np
np.set_printoptions(threshold=1e2)
import torch.nn.functional as F
import scipy.sparse as sp
import  utils.util as util

import model.model_EGCN as model_EGCN
import utils.util as util
import pandas as pd
from scipy.spatial import distance
from scipy.special import softmax
def plot_compared_results_bar(pos, topk_values, topk_indexes, index):
    import matplotlib.pyplot as plt

    topk_values= topk_values[index,:]
    topk_indexes = topk_indexes[index, :]
    plt.title("Distribution of sensors")
    # plt.xlim(xmax=7, xmin=0)
    # plt.ylim(ymax=7, ymin=0)
    plt.xlabel("x Locations")
    plt.ylabel("y Locations")
    plt.scatter(pos[index, 0], pos[index, 1], marker='o', c='red')
    plt.scatter(pos[topk_indexes,0], pos[topk_indexes,1],marker='v', c = 'black',s = topk_values*10 )
    plt.show()

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


class trainer():
    def __init__(self, scaler,args , writer = None):
        device = torch.device(args.device)
        self.writer = writer
        self.global_iterators = 0
        if True:

            road_connections = pd.read_csv(os.path.join(args.data_origin_path, args.connections_road_network),
                                               index_col=0).values  # shape (# raods, attributes_1)

            road_connections = road_connections + np.identity(road_connections.shape[-1])
            # print("   self.road_connections = ", self.road_connections.shape)

            road_attributes = pd.read_csv(os.path.join(args.data_origin_path, args.edge_att_road_network),  index_col=0).fillna(value=0).values # shape (# raods, attributes_1)
            # print("   self.road_attributes = ",  road_attributes.shape)

            sensor_in_traffic_network = pd.read_csv(os.path.join(args.data_origin_path,args.sensor_in_traffic_network),  index_col=0).fillna(value=0).values # shape (# sensor, attributes_2)
            # print("   self.sensor_in_traffic_network = ", sensor_in_traffic_network.shape)


            sensors_index = sensor_in_traffic_network[:,4]

            sensor_pos = road_attributes[sensors_index.astype(np.int),:2]
            sensor_pos = util.standardization(sensor_pos)
            self.road_sensor_pos = torch.tensor(sensor_pos).to(torch.float).to(device)
            self.road_sensor_pos = torch.unsqueeze(self.road_sensor_pos, 0)
            self.road_sensor_pos = self.road_sensor_pos.expand(args.batch_size, -1, -1)


            distance_cor_mat = distance.cdist(sensor_pos, sensor_pos, 'euclidean')
            distance_std = distance_cor_mat.std()
            adj_mx = np.exp(-np.square(distance_cor_mat / distance_std))
            adj_mx_topk_values, adj_mx_topk_index = torch.tensor(adj_mx).topk(args.neighbors_for_each_nodes, dim=1,
                                                                                   largest=True, sorted=True)

            self.adj_mx_topk_index = torch.unsqueeze(adj_mx_topk_index, 0)
            self.adj_mx_topk_index = self.adj_mx_topk_index.expand(args.batch_size, -1, -1).to(device)

            # print(" adj_mx_topk_index ", adj_mx_topk_index[:10,:])

            # for i in range(10):
            #     plot_compared_results_bar(sensor_pos, adj_mx_topk_values, adj_mx_topk_index, index = i)

            # print(" shape----",adj_mx_topk_index.shape,adj_mx_topk_values.shape)


        if args.cur_model == 'model_EGCN':
            # self.model = model_EGCN.EmbedGCN_VRAE(args, self.road_sensor_pos, self.adj_mx_topk_index, writer=writer)
            self.model = model_EGCN.EmbedGCN(args, self.road_sensor_pos, self.adj_mx_topk_index, writer=writer)

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps = 1.0e-8, amsgrad = False)
        self.args = args
        # learning rate decay
        self.lr_scheduler = None
        if args.lr_decay:
            print('Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer ,
                                                                milestones=lr_decay_steps,
                                                                gamma=args.lr_decay_rate)

        self.loss =  util.masked_mae ##  self.marginal_loss

        self.scaler = scaler
        self.clip = 5

    def StepLR(self):
        if self.args.lr_decay:
            self.lr_scheduler.step()



    def train(self, input, real_val, weights = 1):
        """
        :param input:  shape (batch_size, in_dim, #edges, seq_len)
        :param real_val: shape (batch_size, #edges, seq_len)
        :return:
        """
        self.model.train()
        self.optimizer.zero_grad()
        output, contrastiveLoss = self.model(input)

        output = output.transpose(1,3)

        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)

        ( loss + contrastiveLoss[0] +  contrastiveLoss[1] ).backward()
        # ( contrastiveLoss[0] +  contrastiveLoss[1]).backward()
        # ( loss + contrastiveLoss[0] ).backward(retain_graph=True)
        # (loss ).backward(retain_graph=True)
        self.writer.add_scalar('Loss/loss_mae', loss,  self.global_iterators)

        self.writer.add_scalar('Loss/loss_0', contrastiveLoss[0], self.global_iterators)
        self.writer.add_scalar('Loss/loss_1', contrastiveLoss[1], self.global_iterators)
        self.writer.add_scalar('Loss/loss_2', contrastiveLoss[2], self.global_iterators)
        self.writer.add_scalar('Loss/loss_3', contrastiveLoss[3], self.global_iterators)
        self.writer.add_scalar('Loss/loss_4', contrastiveLoss[4], self.global_iterators)

        self.global_iterators = self.global_iterators + 1

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        return loss.item(), mape, rmse

    def eval(self, input, real_val):

        self.model.eval()
        output,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()

        return loss.item(),mape,rmse


