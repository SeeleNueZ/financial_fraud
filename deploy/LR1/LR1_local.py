import torch
from torch import optim

from deploy.local_client import local_client
from model.LR1 import LR1_model_global


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
        self.model_opti.zero_grad()
        train_data = self.train_data[batch, :]
        train_data_noise = self.train_data[batch, :]
        train_data_noise = train_data_noise + noise
        # data_noisy = torch.clip(train_data_noise, 0, 1)  # 假设数据范围为[0,1]
        y = self.model(train_data)
        y_noise = self.model(train_data_noise)
        self.send(client, {self.c_id: [y, y_noise]})

    def train_update_local_2(self, client):
        a = [p.grad.clone().detach() for p in self.model.parameters()]
        b = self.model_opti.state_dict()
        self.send(client, {self.c_id: [a, b]})
        self.model_opti.step()

    def train_update_global_1(self, batch, client, noise):
        with torch.no_grad():
            self.model.train()
            self.model_opti.zero_grad()
            train_data = self.train_data[batch, :]
            train_data_noise = self.train_data[batch, :]
            train_data_noise = train_data_noise + noise
            # data_noisy = torch.clip(train_data_noise, 0, 1)  # 假设数据范围为[0,1]
            y = self.model(train_data)
            y_noise = self.model(train_data_noise)
            self.send(client, {self.c_id: [y, y_noise]})

    def train_update_global_2(self):
        for param, grad in zip(self.model.parameters(), self.msg[-1][0]):
            param.grad = grad.clone() if grad is not None else None
        self.model_opti.load_state_dict(self.msg[-1][1])
        self.model_opti.step()

    def test_data_eval(self, client):
        self.model.eval()
        test_data = self.test_data
        y = self.model(test_data)
        self.send(client, {self.c_id: y})
