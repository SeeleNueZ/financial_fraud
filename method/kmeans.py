import torch

from deploy.kmeans.kmeans_global import kmeans_global
from deploy.kmeans.kmeans_local import kmeans_local


class kmeans:
    def __init__(self, train_data, train_data_val, test_data, test_data_val, data_agg, data_agg_val, config, num_dim,
                 num_data, num_client, n_cluster):
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
        self.n_cluster = n_cluster
        self.local_client_list = []
        self.global_client = None
        self.center = None
        self.n_cluster = 2
        self.dis = None

    def init_client(self):
        self.global_client = kmeans_global(data=self.data_agg, data_val=self.data_agg_val, n_cluster=self.n_cluster)
        for i in range(0, len(self.config)):
            A = kmeans_local(c_id=i, train_data=self.train_data[i], test_data=self.test_data[i],
                             n_cluster=self.n_cluster)
            self.local_client_list.append(A)
            self.global_client.connect(i, A)

    def run(self):
        for i in range(0, len(self.config)):
            self.local_client_list[i].kmeans_task1(self.global_client)
        self.global_client.kmeans_task1(num_client=self.num_client, num_dim=self.num_dim, num_data=self.num_data,
                                        config=self.config)
        print(self.global_client.cluster_center)
        for i in range(0, len(self.config)):
            self.local_client_list[i].kmeans_task2(self.global_client)
        self.dis = self.global_client.kmeans_task2(num_client=self.num_client, num_data=self.num_data)
        ### test ###
        # subarrays = {}
        # for key, indices in self.config.items():
        #     subarrays[key] = self.global_client.cluster_center[:, indices]
        #     # print("subarray:",subarrays[key])
        # temp = torch.zeros(self.test_data[0].shape[0], 2)
        # for i in range(len(self.config)):
        #     # print(subarrays[i].shape)
        #     distances = torch.cdist(self.test_data[i], torch.tensor(subarrays[i], dtype=torch.float32))
        #     # print("dist:", distances.shape)
        #     temp = temp + distances ** 2
        #     # print("temp", temp)
        # temp = torch.sqrt(temp)
        # # print("result:", temp)
        # temp = 1 / (torch.exp(temp))
        # temp = temp / temp.sum(axis=1)[:, None]
        # print("temp:")
        # print(temp)

    def cal(self):
        self.dis = 1 / (torch.exp(self.dis))
        self.dis = self.dis / self.dis.sum(axis=1)[:, None] z
        _, max_indices = torch.max(self.dis, dim=1)
        # 逐元素比较两个 Tensor
        equal_elements = max_indices == self.test_data_val
        # print(max_indices.shape)
        # print(self.test_data_val.shape)

        # 计算相等元素的数量
        num_equal = equal_elements.sum().item()
        # 计算准确率
        accuracy = num_equal / self.num_data
        if accuracy < 0.5:
            accuracy = 1 - accuracy

        print(f"准确率: {accuracy:.4f}")
