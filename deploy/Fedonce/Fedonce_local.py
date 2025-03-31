import torch
from scipy.optimize import linear_sum_assignment
from torch import optim

from deploy.local_client import local_client
from model.LR1 import LR1_model_global
from tools import DataLoader


class Fedonce_local(local_client):
    def __init__(self, c_id, train_data, test_data, model, lr):
        super().__init__(c_id=c_id, train_data=train_data, test_data=test_data
                         , model=model)
        self.model_opti = None
        self.lr = lr
        self.N = None
        self.P = None

    def init(self, nat_out):
        self.model_opti = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.C = torch.empty((self.train_data.shape[0], nat_out), dtype=torch.float).uniform_(0.001, 1)
        # 做一个全局映射cosP，用于batch训练
        self.M = torch.arange(self.train_data.shape[0])

    def local_update_extraction(self):
        total_loss = 0.0
        n = 0
        DL = DataLoader.DataLoader(len=self.train_data.shape[0], batch_size=32, shuffle=True)
        for i, batch in enumerate(DL):
            self.model_opti.zero_grad()
            C_batch = self.C[batch, :]
            R_batch = self.model.forward(self.train_data[batch, :])
            loss = torch.mean((C_batch - R_batch) ** 2) / 2
            loss.backward()
            self.model_opti.step()
            total_loss += loss.item()
            n = n + 1
        return total_loss / n

    def local_update_p(self):
        with torch.no_grad():
            R = self.model.forward(self.train_data)
            cost = torch.cdist(R, self.C, p=2).pow(2).numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            x = torch.arange(self.train_data.shape[0])
            self.M[x] = x[col_ind]
            # print(self.M)

    def model_trans(self, client):
        a = self.model.state_dict()
        self.send(client, {self.c_id: a})
