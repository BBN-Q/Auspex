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

launch_client = False

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
    uids             = []
    client_ident     = None
    
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

                # A new client has connected. Send the most recent information:
                if socks.get(client_desc_sock) == zmq.POLLIN:
                    ident, msg = client_desc_sock.recv_multipart()
                    if msg == b"new_client":
                        print(f"Sending plot descriptor for session to client {ident}")
                        if len(uids) > 0:
                            client_desc_sock.send_multipart([ident, b"new", uids[-1], json.dumps(plot_descriptors[uids[-1]]).encode('utf8')])
                        else:
                            print("No current plots availiable. Waiting for auspex.")
                        client_ident = ident
                # A new auspex data run has started!
                if socks.get(auspex_desc_sock) == zmq.POLLIN:
                    msg = auspex_desc_sock.recv_multipart()
                    ident, uid, plot_desc = msg
                    uids.append(uid)
                    plot_descriptors[uid] = json.loads(plot_desc)
                    print(f"Received auspex plot descriptor for new plotter {uid} from {ident}")
                    auspex_desc_sock.send_multipart([ident, b"ACK"])
                    
                    # Contact any connected clients and tell them to make a new plotter
                    client_desc_sock.send_multipart([client_ident, b"new", uid, json.dumps(plot_descriptors[uid]).encode('utf8')])

                    if launch_client:
                        client_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"auspex-plot-client.py")
                        preexec_fn  = os.setsid if hasattr(os, 'setsid') else None
                        subprocess.Popen(['python', client_path, 'localhost'], env=os.environ.copy(), preexec_fn=preexec_fn)

                if socks.get(auspex_data_sock) == zmq.POLLIN:
                    # The expected data order is [uid, msg, plot name, json.dumps(metadata), np.ascontiguousarray(dat)]
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