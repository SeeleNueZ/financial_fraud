import torch

from deploy.LR1.LR1_global import LR1_global
from deploy.LR1.LR1_local import LR1_local
from model.LR1.LR1_global import LR1_g
from tools import DataLoader


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
        self.batch_size_pre = 32
        self.epoch_pre = 80
        self.batch_size = 32
        self.epoch = 100
        self.lr = 1e-3

    def init_client(self):
        model_list = []
        for i in range(4):
            model = LR1_g(len(self.config[i]), self.emb_config, self.emb_out)
            model_list.append(model)
        classifier_model = LR1_g(self.emb_out * self.num_client, self.classifier_config, 1, )
        self.global_client = LR1_global(data=self.data_agg, data_val=self.data_agg_val, model=model_list, lr=self.lr,
                                        classifier=classifier_model)
        for i in range(0, len(self.config)):
            A = LR1_local(c_id=i, train_data=self.train_data[i], test_data=self.test_data[i], lr=self.lr)
            self.local_client_list.append(A)
            self.global_client.connect(i, A)
        self.global_client.init()

    def global_pre_train(self):
        DL = DataLoader.DataLoader(self.data_agg.shape[0], self.batch_size, shuffle=True)
        for ep in range(0, self.epoch_pre):
            loss = 0
            n = 0
            for i, batch in enumerate(DL):
                loss = loss + self.global_client.pre_train(batch=batch, config=self.config)
                n = n + 1
            print("epoch:{},loss:{:.6f}".format(ep, (loss / n).item()))
        print("acc:{:.5f}".format(self.global_client.global_acc_cal()))
