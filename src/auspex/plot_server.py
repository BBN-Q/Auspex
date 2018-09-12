import json
import zmq
import os
import sys

client_desc_port = 7771
auspex_desc_port = 7761
client_data_port = 7772
auspex_data_port = 7762

context = zmq.Context()
client_desc_sock = context.socket(zmq.ROUTER)
auspex_desc_sock = context.socket(zmq.ROUTER)
client_data_sock = context.socket(zmq.PUB)
auspex_data_sock = context.socket(zmq.ROUTER)
client_desc_sock.bind("tcp://*:%s" % client_desc_port)
auspex_desc_sock.bind("tcp://*:%s" % auspex_desc_port)
client_data_sock.bind("tcp://*:%s" % client_data_port)
auspex_data_sock.bind("tcp://*:%s" % auspex_data_port)
poller = zmq.Poller()
poller.register(client_desc_sock, zmq.POLLIN)
poller.register(auspex_desc_sock, zmq.POLLIN)
poller.register(client_data_sock, zmq.POLLIN)
poller.register(auspex_data_sock, zmq.POLLIN)

# Should be empty by default
plot_descriptors = {}

# Loop and accept messages
try:
    while True: 
        try:
            socks = dict(poller.poll(100))

            if socks.get(client_desc_sock) == zmq.POLLIN:
                ident, uid, msg = client_desc_sock.recv_multipart()
                if msg == b"WHATSUP":
                    client_desc_sock.send_multipart([ident, b"HI!", json.dumps(plot_descriptors[uid]).encode('utf8')])

            if socks.get(auspex_desc_sock) == zmq.POLLIN:
                msg = auspex_desc_sock.recv_multipart()
                ident, uid, plot_desc = msg
                plot_descriptors[uid] = json.loads(plot_desc)

                auspex_desc_sock.send_multipart([ident, b"ACK"])

            if socks.get(auspex_data_sock) == zmq.POLLIN:
                # The expected data order is [msg, name, json.dumps(metadata), np.ascontiguousarray(dat)]
                # We assume that this is true and merely pass along the message
                msg = auspex_data_sock.recv_multipart()
                client_data_sock.send_multipart(msg[1:])
        except Exception as e:
            print("Plot server generated exception", e)

except KeyboardInterrupt:
    print("Server manually terminated.")
finally:
    print("Shutting down Auspex plot server")
    client_desc_sock.close()
    auspex_desc_sock.close()
    client_data_sock.close()
    auspex_data_sock.close()
    context.term()