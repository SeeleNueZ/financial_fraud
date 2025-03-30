import torch
from torch import optim

from deploy.local_client import local_client
from model.LR1 import LR1_global


class LR1_local(local_client):
    def __init__(self, c_id, train_data, test_data, lr):
        super().__init__(c_id=c_id, train_data=train_data, test_data=test_data
                         , model=None)
        self.cluster_center = None
        self.model_opti = None
        self.lr = lr

    def init(self, model):
        self.model = model
        self.model_opti = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=self.lr, eps=1e-08,
                                           weight_decay=0)
        self.model.load_state_dict(self.msg[-1][0])
        self.model_opti.load_state_dict(self.msg[-1][1])

    def train_update_local_1(self, batch, client, noise):
        self.model.train()
        train_data = self.train_data[batch, :]
        train_data_noise = self.train_data[batch, :]
        train_data_noise = train_data_noise + noise
        # data_noisy = torch.clip(train_data_noise, 0, 1)  # 假设数据范围为[0,1]
        self.model_opti.zero_grad()
        y = self.model(train_data)
        y_noise = self.model(train_data_noise)
        self.send(client, {self.c_id: [y, y_noise]})

    def train_update_local_2(self, batch, client):
        a = [p.grad.clone().detach() for p in self.model.parameters()]
        b = self.model_opti.state_dict()
        self.send(client, {self.c_id: [a, b]})
        self.model_opti.step()
