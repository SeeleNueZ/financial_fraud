import torch.optim
from torch import nn

from deploy.global_client import global_client


class LR1_global(global_client):
    def __init__(self, data, data_val, model, classifier, lr):
        super().__init__(data=data, data_val=data_val, model=model)
        # model:四个emb
        # classifier：全局分类器
        self.classifier = classifier
        self.model_opti = None
        self.classifier_opti = None
        self.lr = lr
        self.criterion = nn.BCELoss(reduction="mean")
        self.cal_label = torch.tensor(data_val.shape)

    def init(self):
        self.classifier_opti = torch.optim.Adam(self.classifier.parameters(), betas=(0.9, 0.999), lr=self.lr, eps=1e-08,
                                                weight_decay=0)
        list_temp = []
        for i in self.model:
            a = torch.optim.Adam(i.parameters(), betas=(0.9, 0.999), lr=self.lr, eps=1e-08,
                                 weight_decay=0)
            list_temp.append(a)
        self.model_opti = list_temp

    def pre_train(self, batch, config):
        list_tensor = []
        for i in config:
            self.model[i].train()
            self.model_opti[i].zero_grad()
            # print(batch, config[i])
            # print(self.data.shape)
            train_data = self.data[batch, :]
            train_data = train_data[:, config[i]]
            # print(train_data.shape)
            train_data = self.model[i].forward(train_data)
            list_tensor.append(train_data)

        train_data = torch.cat(list_tensor, dim=1)
        # print(train_data.shape)
        self.classifier_opti.zero_grad()
        y = self.classifier(train_data)
        # print(y.shape, self.data_val[batch].shape)
        y = y.squeeze()
        m = nn.Sigmoid()
        y = m(y)
        y_copy = y.clone().detach()
        loss = self.criterion(y, self.data_val[batch])
        return_loss = loss.clone().detach()
        loss.backward()
        self.classifier_opti.step()
        for i in config:
            self.model_opti[i].step()
        print(self.cal_label)
        return return_loss
