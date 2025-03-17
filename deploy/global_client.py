class global_client:

    def __init__(self, data, model, data_val):
        self.c_id = -1
        self.data = data
        self.data_val = data_val
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
