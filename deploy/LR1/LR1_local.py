from deploy.local_client import local_client


class LR1_local(local_client):
    def __init__(self, c_id, train_data, test_data, lr):
        super().__init__(c_id=c_id, train_data=train_data, test_data=test_data
                         , model=None)
        self.cluster_center = None
        self.model_opti = None
        self.lr = lr
