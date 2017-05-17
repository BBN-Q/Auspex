import zmq


class SimpleMatplotClient(object):

    TIMEOUT = 5000

    def __init__(self, status_port = 7771, data_port = 7772):
        super(SimpleMatplotClient, self).__init__()
        self.name = "Dumb Client"
        self.context = zmq.Context()
        self.status_socket = self.context.socket(zmq.DEALER)
        self.status_socket.identity = self.name.encode()
        self.status_socket.connect("tcp://localhost:%s" % status_port)
        self.poller = zmq.Poller()
        self.poller.register(self.status_socket)
        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.connect("tcp://localhost:%s" % data_port)
        self.data_socket.setsockopt_string(zmq.SUBSCRIBE, 'server')

    def query_server(self):
        print("sending...")
        self.status_socket.send(b"WHATSUP")
        print("polling...")
        evts = dict(self.poller.poll(self.TIMEOUT))
        if self.status_socket in evts:
            reply = self.status_socket.recv()
            print("Got {} from server.".format(reply.decode()))
        else:
            print("Server is dead!")

    def get_some_data(self):
        for i in range(10):
            print(self.data_socket.recv())

client = SimpleMatplotClient()
client.query_server()
client.get_some_data()
