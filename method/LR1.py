import torch

from deploy.LR1.LR1_global import LR1_global
from deploy.LR1.LR1_local import LR1_local
from model.LR1.LR1_global import LR1_g


class LR1:
    def __init__(self, train_data, train_data_val, test_data, test_data_val, data_agg, data_agg_val, config, num_dim,
                 num_data, num_client):
        self.device = 'cuda'
        self.train_data = train_data
        self.train_data_val = train_data_val
        self.test_data = test_data
        self.test_data_val = test_data_val
        self.data_agg = data_agg
        self.data_agg_val = data_agg_val
        self.config = config
        self.num_data = num_data
        self.num_dim = num_dim
        self.num_client = num_client
        self.local_client_list = []
        self.global_client = None
        self.emb_config = [8]
        self.emb_out = 8
        self.classifier_config = [64, 32, 4]
        self.lr = 1e-3

    def init_client(self):
        model_list = []
        for i in range(4):
            model = LR1_g(self.num_dim, self.emb_config, self.emb_out)
            model_list.append(model)
        classifier_model = LR1_g(self.num_dim * self.num_client, self.classifier_config, 1,)
        self.global_client = LR1_global(data=self.data_agg, data_val=self.data_agg_val, model=model_list, lr=self.lr,
                                        classifier=classifier_model)
        for i in range(0, len(self.config)):
            A = LR1_local(c_id=i, train_data=self.train_data[i], test_data=self.test_data[i], lr=self.lr)
            self.local_client_list.append(A)
            self.global_client.connect(i, A)
        self.global_client.init()

    def run(self):
        for i in range(0, len(self.config)):
            self.local_client_list[i].kmeans_task1(self.global_client)
        self.global_client.kmeans_task1(num_client=self.num_client, num_dim=self.num_dim, num_data=self.num_data,
                                        config=self.config)
        print(self.global_client.cluster_center)
        for i in range(0, len(self.config)):
            self.local_client_list[i].kmeans_task2(self.global_client)
        self.dis = self.global_client.kmeans_task2(num_client=self.num_client, num_data=self.num_data)
