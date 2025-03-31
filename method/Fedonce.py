import torch
from torch import nn

from deploy.Fedonce.Fedonce_global import Fedonce_global
from deploy.Fedonce.Fedonce_local import Fedonce_local
from model.LR1.LR1_model_global import LR1_g
from tools import DataLoader


class Fedonce:
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
        self.nat_config = [8, 32, 16]
        self.nat_out = 8
        self.classifier_config = [64, 32, 4]
        self.batch_size = 32
        self.epoch = 50
        self.lr = 1e-4
        self.epoch_global = 500
        self.batch_size_global = 32

    def init_client(self):
        for i in self.config:
            model = LR1_g(len(self.config[i]), self.nat_config, self.nat_out)
            a = Fedonce_local(c_id=i, train_data=self.train_data[i], test_data=self.test_data[i], model=model,
                              lr=self.lr)
            self.local_client_list.append(a)
            a.init(nat_out=self.nat_out)
        model = LR1_g(self.num_client * self.nat_out, self.classifier_config, 1)
        self.global_client = Fedonce_global(data=self.data_agg, data_val=self.data_agg_val, classifier=model,
                                            lr=self.lr, model=None)

    def local_train(self):
        for ep in range(self.epoch):
            for i in self.config:
                x = self.local_client_list[i].local_update_extraction()
                # print("ep:{},c_id:{},loss{:.6f}".format(ep, i, x))
            if ep + 1 % 10 == 0:
                for i in self.config:
                    self.local_client_list[i].local_update_p()

    def model_trans(self):
        model_list = []
        for i in self.config:
            model = LR1_g(len(self.config[i]), self.nat_config, self.nat_out)
            model_list.append(model)
            self.local_client_list[i].model_trans(client=self.global_client)
        self.global_client.init(config=self.config, model_list=model_list)

    def global_train(self):
        for ep in range(self.epoch_global):
            DL = DataLoader.DataLoader(len=self.data_agg.shape[0], batch_size=self.batch_size_global, shuffle=True)
            loss = 0
            n = 0
            # update local
            for i, batch in enumerate(DL):
                loss = loss + self.global_client.global_update(config=self.config, batch=batch)
                n = n + 1
            # print("epoch:{},local,loss:{:.6f}".format(ep, (loss / n)))

    def test_val(self):
        y = []
        for i in self.config:
            y1 = self.global_client.model[i](self.test_data[i])
            y.append(y1)
        y = torch.cat(y,dim=1)
        y = self.global_client.classifier(y)
        y = y.squeeze()
        m = nn.Sigmoid()
        y = m(y)
        y = y.clone().detach()
        y = (y >= 0.5).float()
        correct = torch.eq(y, self.test_data_val)
        correct_count = correct.long().sum()
        accuracy = correct_count.item() / y.shape[0]
        print("accuracy:{:.4f}".format(accuracy))

