import torch.optim
from torch import nn

from deploy.global_client import global_client


class Fedonce_global(global_client):
    def __init__(self, data, data_val, model, classifier, lr):
        super().__init__(data=data, data_val=data_val, model=model)
        # model:四个emb
        # classifier：全局分类器
        self.classifier = classifier
        self.model_opti = None
        self.classifier_opti = None
        self.lr = lr
        self.criterion = nn.BCELoss(reduction="mean")
        self.cal_label = torch.ones(data_val.shape)
        self.lamda = 0.5

    def init(self, config, model_list):
        self.model_opti = []
        for i in config:
            model_list[i].load_state_dict(self.msg[i])
            a = torch.optim.Adam(model_list[i].parameters(), lr=self.lr)
            self.model_opti.append(a)
        self.model = model_list
        self.classifier_opti = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss(reduction="mean")

    def global_update(self, batch, config):
        train_data = self.data[batch, :]
        list_temp = []
        for i in config:
            self.model_opti[i].zero_grad()
            data = train_data[:, config[i]]
            data = self.model[i].forward(data)
            list_temp.append(data)
        data = torch.cat(list_temp, dim=1)
        self.classifier_opti.zero_grad()
        y = self.classifier(data)
        m = nn.Sigmoid()
        y = y.squeeze()
        y = m(y)
        loss = self.criterion(y, self.data_val[batch])
        loss.backward()
        return loss.item()
