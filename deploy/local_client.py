class local_client:

    def __init__(self, c_id, train_data,  test_data, model):
        self.c_id = c_id
        self.train_data = train_data
        self.test_data = test_data
        self.device = None
        self.model = model
        self.connected_clients = {}
        self.msg = {}

    def connect(self, c_id, client):
        self.connected_clients[c_id] = client

    def send(self, client, msg: dict):
        # guest/host clients send message
        client.receive(msg)

    def receive(self, msg: dict):
        # guest/host clients receive message
        self.msg.update(msg)
