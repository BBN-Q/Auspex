import zmq
from random import randrange
import time
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

def send_array(socket, A, flags=0, copy=False, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_string(f"buq123 Plot{np.random.randint(3)} 111", flags|zmq.SNDMORE) # Session name, plot name, subplot number
    socket.send_json(md, flags|zmq.SNDMORE) # Array metadata
    return socket.send(A, flags, copy=copy, track=track) # Array data

while True:
    send_array(socket, np.random.random(50))

    time.sleep(1)