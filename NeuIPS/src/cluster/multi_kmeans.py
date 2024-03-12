import math
import torch
from time import time
import numpy as np
from line_profiler import LineProfiler
class MultiKMeans:
  '''
  Kmeans clustering algorithm implemented with PyTorch
  Parameters:
    n_kmeans: int,
      Number of concurrent KMeans algorithms
    n_clusters: int, 
      Number of clusters
    max_iter: int, default: 100
      Maximum number of iterations
    tol: float, default: 0.0001
      Tolerance
    
    verbose: int, default: 0
      Verbosity
    mode: {'euclidean', 'cosine'}, default: 'euclidean'
      Type of distance measure
    minibatch: {None, int}, default: None
      Batch size of MinibatchKmeans algorithm
      if None perform full KMeans algorithm
      
  Attributes:
    centroids: torch.Tensor, shape: [n_clusters, n_features]
      cluster centroids
  '''
  def __init__(self, n_clusters, n_kmeans, device , max_iter=100, tol=0.0001, verbose=0,  cluster_min_dist = 0.1, mode="euclidean", minibatch=None):
    self.n_clusters = n_clusters
    self.n_kmeans = n_kmeans
    self.max_iter = max_iter
    self.tol = tol*n_kmeans
    self.verbose = verbose
    self.mode = mode
    self.cluster_min_dist = cluster_min_dist
    self.minibatch = minibatch
    self._loop = False
    self._show = False
    self.device = device
    eye_mat = torch.eye(self.n_clusters).unsqueeze(0).expand(self.n_kmeans, -1, -1).to(self.device)
    self.target = torch.ones_like(eye_mat) - eye_mat
    self.target = cluster_min_dist*self.target
    try:
      import PYNVML
      self._pynvml_exist = True
    except ModuleNotFoundError:
      self._pynvml_exist = False
    
    self.centroids = None

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
    return  -2 * a @ b.transpose(-2, -1) +(a**2).sum(dim=-1)[..., :, None] +(b**2).sum(dim=-1)[..., None, :]

  def remaining_memory(self):
    """
      Get remaining memory in gpu
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if self._pynvml_exist:
      pynvml.nvmlInit()
      gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
      remaining = info.free
    else:
      remaining = torch.cuda.memory_allocated()
    return remaining

  def c_distance(self, a, b):
    if self.mode == 'cosine':
      sim_func = self.cos_sim
    elif self.mode == 'euclidean':
      sim_func = self.euc_sim

    sim = sim_func(a, b)
    return sim

  def max_sim(self, a, b):
    """
      Compute maximum similarity (or minimum distance) of each vector
      in a with all of the vectors in b
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    if self.mode == 'cosine':
      sim_func = self.cos_sim
    elif self.mode == 'euclidean':
      sim_func = self.euc_sim

    sim = sim_func(a, b)
    return sim


  # @profile
  def fit_predict(self, X, centroids=None):
    n_stream, n_points, emb_dim = X.shape
    start_time = time()
    if self.centroids is None:
      self.centroids = X[:, np.random.choice(n_points, size=[self.n_clusters], replace=False)]
    closest = None
    for i in range(self.max_iter):
      iter_time = time()
      x = X.float()
      sim = self.max_sim(a=x, b=self.centroids)
      closest_origin = sim.min(dim=-1)[1]
      expanded_closest = closest_origin[:, None].expand(-1, self.n_clusters, -1)
      mask = (expanded_closest==torch.arange(self.n_clusters, device= self.device)[None, :, None]).float()
      c_grad = mask @ x / mask.sum(-1, keepdim=True)
      error = (c_grad - self.centroids).pow(2).sum(dim=-1).sum(dim=-1).mean()
      if error <= self.tol:
        c_grad [c_grad!=c_grad] = 0
        self.centroids = c_grad
        break
      else:
      ##### for balanced clustering!!
        closest = closest_origin +1
        values, _ = torch.mode(closest, 1)
        mask_2 = torch.eq(closest, values[...,0]).float()
        top_k_value = int( n_points / self.n_clusters)
        top_k_index = mask_2.topk( top_k_value,dim = -1, sorted = False)[1]
        random_index = torch.randint(0, top_k_value, [self.n_kmeans,self.n_clusters]).to(self.device)
        random_index = torch.gather(top_k_index,-1,random_index)
        random_index = random_index.unsqueeze(-1).expand(-1,-1, emb_dim)
        selected_centers = torch.gather(x,1, random_index)
        c_grad = torch.where(c_grad == c_grad, c_grad, selected_centers)
        self.centroids = c_grad

    if self.verbose >= 1:
      print(
        f'used {i + 1} iterations ({round(time() - start_time, 4)}s)  into {self.n_clusters} clusters')

    ## Clustering Loss
    mask = mask.clone().detach()
    self.centroids_detached = self.centroids.clone().detach()
    distance = self.max_sim(a=self.centroids_detached, b=x)
    masked_distance = torch.mul(distance, mask)
    diff = masked_distance.sum(dim = -1)
    cluster_mask = mask.sum(dim = -1)
    cluster_mask = torch.clamp(cluster_mask,1)
    diff = torch.div(diff, cluster_mask)
    kmeans_loss = diff.sum(dim = -1).mean()

    ### another loss for maximize distance between  clusters

    cluster_distance = self.max_sim(a=self.centroids, b=self.centroids)
    uniform_loss = torch.pow(self.target-cluster_distance,2).mean()

    return closest_origin,  self.centroids,  kmeans_loss, uniform_loss #  self.centroids.mean()



  def fit_predict_faster(self, X, centroids=None):
    """
      Combination of fit() and predict() methods.
      This is faster than calling fit() and predict() seperately.
      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]
      centroids: {torch.Tensor, None}, default: None
        if given, centroids will be initialized with given tensor
        if None, centroids will be randomly chosen from X
      Return:
      labels: torch.Tensor, shape: [n_samples]
    """
    n_stream, n_points, emb_dim = X.shape
    print(X.shape)
    device = X.device.type
    start_time = time()
    if self.centroids is None:
      self.centroids = X[:, np.random.choice(n_points, size=[self.n_clusters], replace=False)]

    if centroids is not None:
      self.centroids = centroids

    closest = None
    for i in range(self.max_iter):
      iter_time = time()
      if self.minibatch is not None:
        x = X[:, np.random.choice(n_points, size=[self.minibatch], replace=False)]
      else:
        x = X

      # A faster way to calculate mask!!!!
      x_expand = x.unsqueeze(1).expand(-1,self.n_clusters,-1,-1)
      centroids_expand = self.centroids.unsqueeze(2).expand(-1,-1,n_points, -1)
      distance = (x_expand - centroids_expand).pow(2).sum(dim = -1)
      closest = distance.min(dim = 1)[1]
      mask_index = closest.unsqueeze(-1)
      final_mask = torch.zeros(self.n_kmeans,  n_points, self.n_clusters,).cuda().scatter_(2, mask_index, 1)
      mask = final_mask.permute(0,2,1)
      c_grad = mask @ x / mask.sum(-1, keepdim=True)
      c_grad[c_grad!=c_grad] = 0 # remove NaNs

      error = (c_grad - self.centroids).pow(2).sum()

      self.centroids = c_grad
      if error <= self.tol * self.n_kmeans:
        break

    print(
      f'used {i + 1} iterations ({round(time() - start_time, 4)}s) into {self.n_clusters} clusters')

    return closest

