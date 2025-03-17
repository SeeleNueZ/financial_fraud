import torch
from sklearn.cluster import KMeans

from deploy.local_client import local_client


class kmeans_local(local_client):
    def __init__(self, c_id, train_data,  test_data,  n_cluster):
        super().__init__(c_id=c_id, train_data=train_data, test_data=test_data
                         ,  model=KMeans())
        self.cluster_center = None
        self.n_cluster = n_cluster
        self.dict = None

    def kmeans_task1(self, client):
        self.model = KMeans(n_clusters=self.n_cluster, tol=1e-3)
        label = self.model.fit_predict(self.train_data)
        cluster_center = self.model.cluster_centers_
        # print("label----")
        # print(label)
        # print("cluster----")
        # print(cluster_center)
        dist = self.model.transform(self.train_data)
        # print("======\n", dist)
        temp = {self.c_id: [label, cluster_center]}
        # print(temp[0][1])
        self.send(client, temp)

    def kmeans_task2(self, client):
        self.cluster_center = self.msg[-1]
        # print("client,center:",self.cluster_center)
        distances = torch.cdist(self.test_data, torch.tensor(self.cluster_center, dtype=torch.float32))
        self.send(client, {self.c_id: distances})
        # print(out)

    def get_center(self):
        return self.dict, self.cluster_center
