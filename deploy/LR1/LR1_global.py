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
        self.cal_label = torch.ones(data_val.shape)
        self.lamda = 0.5

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
        y_copy = (y_copy >= 0.5).float()
        loss = self.criterion(y, self.data_val[batch])
        return_loss = loss.clone().detach()
        loss.backward()
        self.classifier_opti.step()
        for i in config:
            self.model_opti[i].step()
        # print(self.cal_label.shape)
        for i, ba in enumerate(batch):
            # print(i ,ba)
            self.cal_label[ba] = y_copy[i]
        return return_loss

    def global_acc_cal(self):
        # print(self.cal_label)
        # print(self.data_val)
        correct = torch.eq(self.cal_label, self.data_val)
        correct_count = correct.long().sum()
        accuracy = correct_count.item() / self.cal_label.shape[0]
        return accuracy

    def model_trans(self, config):
        for i in config:
            list_temp = []
            a = self.model[i].state_dict()
            b = self.model_opti[i].state_dict()
            list_temp.append(a)
            list_temp.append(b)
            self.send(self.connected_clients[i], {self.c_id: list_temp})

    def train_update_local_1(self, config):
        self.classifier_opti.zero_grad()
        # 来自local的输出x2
        y = []
        y_noise = []
        for i in config:
            y.append(self.msg[i][0])
            y_noise.append(self.msg[i][1])
        y = torch.cat(y, dim=1)
        y_noise = torch.cat(y_noise, dim=1)
        y = self.classifier(y)
        y_noise = self.classifier(y_noise)
        m = nn.Sigmoid()
        y = m(y)
        y_noise = m(y_noise)
        loss_local = ((y - y_noise) ** 2).mean()
        # print("loss_local")
        # pr_loss_local = loss_local.clone().detach()
        # print(pr_loss_local)

        # global 数据的noise
        with torch.no_grad():
            noise = torch.empty(self.data.shape, dtype=torch.float).uniform_(0.001, 0.05)
            global_train_data = self.data + noise
            # print(global_train_data.shape)
            list_tensor = []
            for i in config:
                self.model[i].train()
                self.model_opti[i].zero_grad()
                train_data = global_train_data[:, config[i]]
                train_data = self.model[i].forward(train_data)
                list_tensor.append(train_data)
            global_train_data = torch.cat(list_tensor, dim=1)
        y_global = self.classifier(global_train_data)
        # print(y.shape, self.data_val[batch].shape)
        y_global = y_global.squeeze()
        y_global = m(y_global)
        loss_global = self.criterion(y_global, self.data_val)
        # print("loss_global")
        # pr_loss_global = loss_global.clone().detach()
        # print(pr_loss_global)
        loss = loss_local + loss_global * self.lamda
        return_loss = loss.clone().detach()
        # print("loss", return_loss)
        loss.backward()
        return return_loss

    def train_update_local_2(self, config):
        for i in config:
            for param, grad in zip(self.model[i].parameters(), self.msg[i][0]):
                param.grad = grad.clone() if grad is not None else None
            self.model_opti[i].load_state_dict(self.msg[i][1])
            self.model_opti[i].step()
        self.classifier_opti.step()

    def train_update_global_1(self, config):
        self.classifier_opti.zero_grad()
        # 来自local的输出x2
        with torch.no_grad():
            y = []
            y_noise = []
            for i in config:
                y.append(self.msg[i][0])
                y_noise.append(self.msg[i][1])
            y = torch.cat(y, dim=1)
            y_noise = torch.cat(y_noise, dim=1)
        y = self.classifier(y)
        y_noise = self.classifier(y_noise)
        m = nn.Sigmoid()
        y = m(y)
        y_noise = m(y_noise)
        loss_local = ((y - y_noise) ** 2).mean()
        # print("loss_local")
        # pr_loss_local = loss_local.clone().detach()
        # print(pr_loss_local)

        # global 数据的noise
        noise = torch.empty(self.data.shape, dtype=torch.float).uniform_(0.001, 0.05)
        global_train_data = self.data + noise
        # print(global_train_data.shape)
        list_tensor = []
        for i in config:
            self.model[i].train()
            self.model_opti[i].zero_grad()
            train_data = global_train_data[:, config[i]]
            train_data = self.model[i].forward(train_data)
            list_tensor.append(train_data)
        global_train_data = torch.cat(list_tensor, dim=1)
        y_global = self.classifier(global_train_data)
        # print(y.shape, self.data_val[batch].shape)
        y_global = y_global.squeeze()
        y_global = m(y_global)
        loss_global = self.criterion(y_global, self.data_val)
        # print("loss_global")
        # pr_loss_global = loss_global.clone().detach()
        # print(pr_loss_global)
        loss = loss_local + loss_global * self.lamda
        return_loss = loss.clone().detach()
        # print("loss", return_loss)
        loss.backward()

        # update parameters
        for i in config:
            a = [p.grad.clone().detach() for p in self.model[i].parameters()]
            b = self.model_opti[i].state_dict()
            self.send(self.connected_clients[i], {self.c_id: [a, b]})
            self.model_opti[i].step()
        self.classifier_opti.step()
        return return_loss

    def test_data_eval(self, config):
        y = []
        for i in config:
            y.append(self.msg[i])
        y = torch.cat(y, dim=1)
        y = self.classifier(y)
        m = nn.Sigmoid()
        y = m(y)
        y = y.clone().detach()
        y = (y >= 0.5).float()
        # correct = torch.eq(self.cal_label, self.data_val)
        # correct_count = correct.long().sum()
        # accuracy = correct_count.item() / self.cal_label.shape[0]
        return y


