from multi_kmeans import MultiKMeans
import torch
import numpy
from matplotlib import pyplot
n_clusters = 4
n_point = 8
feature_dim = 2
arr = numpy.empty((n_point, feature_dim), dtype=numpy.float32)
dist = int(n_point/n_clusters)
dist_cluster = 0.01
arr = numpy.random.rand(n_point, feature_dim)
# arr[dist:dist*2] = numpy.random.rand(dist, feature_dim)
# arr[dist*2:dist*3] = numpy.random.rand(dist, feature_dim)
# arr[dist*3:] = numpy.random.rand(dist, feature_dim)
x = torch.from_numpy(arr).to(torch.device('cuda:0')).float()

# kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=1)
# labels = kmeans.fit_predict(x)
# # print(labels)
# labels = labels.cpu().numpy()
# pyplot.scatter(arr[:, 0], arr[:, 1], c=labels)
# pyplot.show()
true_batch = 1
kmeans = MultiKMeans(n_clusters = n_clusters, n_kmeans = true_batch, device = torch.device('cuda'),  mode='euclidean', verbose=1)
# x = torch.randn(207,50*64, 10, device='cuda')
x = x.unsqueeze(0).expand(true_batch,-1,-1)
labels, center, loss,_ = kmeans.fit_predict(x)
print(" -shape ", labels.shape, loss.shape, center.shape)
print(" -shape  values",  loss, )
labels = labels.cpu().numpy()
pyplot.scatter(arr[:, 0], arr[:, 1], c=labels)
pyplot.show()
# kmeans = MultiKMeans(n_clusters = n_clusters, n_kmeans = true_batch,  mode='euclidean', verbose=1)
# for i in range(500):
#     labels = kmeans.fit_predict_faster(x)
# labels = labels.cpu().numpy()
# pyplot.scatter(arr[:, 0], arr[:, 1], c=labels)
# pyplot.show()

# print(labels)
# labels = labels.cpu().numpy()
# pyplot.scatter(arr[:, 0], arr[:, 1], c=labels)
# pyplot.show()
