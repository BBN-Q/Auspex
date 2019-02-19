#!/usr/bin/env python

import json
import zmq
import os, os.path
import sys
import subprocess


client_desc_port = 7771
auspex_desc_port = 7761
client_data_port = 7772
auspex_data_port = 7762

launch_client = True

if __name__ == '__main__':
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
    
    print("Welcome to the Auspex plot server!")
    print("Waiting for auspex to connect on ports 7761/7762")
    print("Waiting for plot client to connect on ports 7771/7772")

    if '--no-launch' in sys.argv[1:]:
        launch_client = False

    # Loop and accept messages
    try:
        while True: 
            try:
                socks = dict(poller.poll(50))

                if socks.get(client_desc_sock) == zmq.POLLIN:
                    ident, uid, msg = client_desc_sock.recv_multipart()
                    print(f"Sending plot descriptor for session {uid} to client {ident}")
                    if msg == b"WHATSUP":
                        client_desc_sock.send_multipart([ident, b"HI!", json.dumps(plot_descriptors[uid]).encode('utf8')])
                if socks.get(auspex_desc_sock) == zmq.POLLIN:
                    msg = auspex_desc_sock.recv_multipart()
                    ident, uid, plot_desc = msg
                    plot_descriptors[uid] = json.loads(plot_desc)
                    print(f"Received auspex plot descriptor for new session {uid} from {ident}")
                    auspex_desc_sock.send_multipart([ident, b"ACK"])
                    if launch_client:
                        client_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"auspex-plot-client.py")
                        preexec_fn  = os.setsid if hasattr(os, 'setsid') else None
                        subprocess.Popen(['python', client_path, 'localhost', uid], env=os.environ.copy(), preexec_fn=preexec_fn)

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