from multi_kmeans import MultiKMeans
import torch
import numpy
from matplotlib import pyplot

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
numpy.random.seed(1)

# 定义一个构建神经网络的类
class Net(torch.nn.Module):  # 继承torch.nn.Module类
    def __init__(self, n_feature,  n_output,n_point):
        super(Net, self).__init__()  # 获得Net类的超类（父类）的构造方法
        # 定义神经网络的每层结构形式
        # 各个层的信息都是Net类对象的属性
        self.hidden = torch.nn.Linear(n_feature, n_output)  # 隐藏层线性输出
        self.batchnm1d = torch.nn.BatchNorm1d(n_point)

    # 将各层的神经元搭建成完整的神经网络的前向通路
    def forward(self, x):
        x =self.hidden(x)  # 对隐藏层的输出进行relu激活

        return self.batchnm1d(x)

    # 定义神经网络


n_clusters = 4
n_point = 8
feature_dim = 2
arr = numpy.empty((n_point, feature_dim), dtype=numpy.float32)
dist = int(n_point/n_clusters)
dist_cluster = 1
# arr = numpy.random.rand(n_point, feature_dim)
arr = numpy.random.randint(0,2,[n_point, feature_dim])
# arr[dist*2:dist*3] = numpy.random.rand(dist, feature_dim)
# arr[dist*3:] = numpy.random.rand(dist, feature_dim)
x = torch.from_numpy(arr).to(torch.device('cuda:0'))
true_batch = 1
x = x.unsqueeze(0).expand(true_batch, -1, -1)

net = Net(feature_dim, feature_dim,n_point)
net.cuda()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入网络参数和学习率
loss_function = torch.nn.MSELoss()  # 最小均方误差

for t in range(1):
    # prediction = net(x)  # 把数据x喂给net，输出预测值

    kmeans = MultiKMeans(n_clusters=n_clusters, n_kmeans=true_batch, device = torch.device('cuda'),  mode='euclidean', verbose=0) #euclidean cosine
    labels, center, kmeans_loss, uniform_loss = kmeans.fit_predict(x.float())

    print("-----",labels)
    #
    # optimizer.zero_grad()  # 清空上一步的更新参数值
    # (kmeans_loss+ uniform_loss  ).backward()  # 误差反相传播，计算新的更新参数值
    # # for name, parms in net.named_parameters():
    # #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    # #           ' -->grad_value:', parms.grad)
    # optimizer.step()

