import torch.optim

from deploy.global_client import global_client


class LR1_global(global_client):
    def __init__(self, data, data_val,model, classifier, lr):
        super().__init__(data=data, data_val=data_val, model=model)
        # model:四个emb
        # classifier：全局分类器
        self.classifier = classifier
        self.model_opti = None
        self.classifier_opti = None
        self.lr = lr

    def init(self):
        self.classifier_opti = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=self.lr, eps=1e-08,
                                                weight_decay=0)
        list_temp={}
        for _ in self.model:
            pass

