import zmq
from random import randrange
import time
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

context2 = zmq.Context()
socket_sessions = context2.socket(zmq.PUB)
socket_sessions.bind("tcp://*:5557")


def send_array(socket, A, session="buq123", flags=0, copy=False, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_string(f"{session} Plot{np.random.randint(3)} 111", flags|zmq.SNDMORE) # Session name, plot name, subplot number
    socket.send_json(md, flags|zmq.SNDMORE) # Array metadata
    return socket.send(A, flags, copy=copy, track=track) # Array data

time.sleep(0.1)
socket_sessions.send_string("session buq123 started")
socket_sessions.send_string("session buq234 started")
for i in range(50):
    send_array(socket, np.random.random(50))
    time.sleep(0.1)
socket_sessions.send_string("session buq123 stopped")
socket_sessions.send_string("session buq234 stopped")
