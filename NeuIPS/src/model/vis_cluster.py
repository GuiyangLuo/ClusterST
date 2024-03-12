from sklearn.manifold import TSNE
# from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
# config InlineBackend.figure_format = "svg"
from sklearn.cluster import KMeans
# digits = load_digits()
# X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)
# X_pca = PCA(n_components=2).fit_transform(digits.data)

class Visualization_Cluster():
    def __init__(self, args,  writer, pos, print_every = 100):
        super(Visualization_Cluster, self).__init__()
        self.writer = writer
        self.n_cluster = args.gcn_n_class
        self.pos = pos[0,:,:].detach().cpu().numpy()
        self.global_iterators = 0
        self.print_every = print_every

    def  prepare_figures(self,features, cluster_class,  cur_pos_index):
        """
        :param features: (m,n)
        :param cluster_class: (m,1)
        :param fig:
        :param plt:
        :return:
        """

        X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
        X_pca = PCA(n_components=2).fit_transform(features)

        font = {"color": "darkred",
                "size": 13,
                "family": "serif"}
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_class, alpha=0.6,
                    cmap=plt.cm.get_cmap('rainbow',  self.n_cluster))
        plt.title("t-SNE", fontdict=font)
        cbar = plt.colorbar(ticks=range( self.n_cluster))
        cbar.set_label(label='digit value', fontdict=font)
        plt.clim(-0.5, 9.5)

        plt.subplot(1, 3, 2)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_class, alpha=0.6,
                    cmap=plt.cm.get_cmap('rainbow',  self.n_cluster))
        plt.title("PCA", fontdict=font)
        cbar = plt.colorbar(ticks=range( self.n_cluster))
        cbar.set_label(label='digit value', fontdict=font)
        plt.clim(-0.5, 9.5)


        plt.subplot(1, 3, 3)
        plt.scatter(self.pos[cur_pos_index][:, 0], self.pos[cur_pos_index][:, 1], c=cluster_class, alpha=0.6,
                    cmap=plt.cm.get_cmap('rainbow',  self.n_cluster))
        plt.title("sensor Distribution", fontdict=font)
        cbar = plt.colorbar(ticks=range( self.n_cluster))
        cbar.set_label(label='digit value', fontdict=font)
        plt.clim(-0.5, 9.5)

        plt.tight_layout()

        return fig


    def vis_figure(self, features, cluster_class, cur_pos_index):
        self.global_iterators = self.global_iterators + 1
        if self.writer and self.global_iterators % self.print_every == 0:

            features = features.detach().cpu().numpy()
            cluster_class = cluster_class.detach().cpu().numpy()
            cur_pos_index = cur_pos_index.detach().cpu().numpy()
            fig = self.prepare_figures(features, cluster_class, cur_pos_index)
            self.writer.add_figure('time_series_data',
                                   fig,
                                   global_step=self.global_iterators,
                                   )


