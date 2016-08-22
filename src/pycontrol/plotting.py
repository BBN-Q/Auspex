import threading
import subprocess
import psutil
import os
import sys

from pycontrol.logging import logger

def in_notebook():
    # From http://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False

class BokehServerThread(threading.Thread):
    def __init__(self):
        super(BokehServerThread, self).__init__()
        self.daemon = True
        self.run_in_notebook = in_notebook()

    def __del__(self):
        self.join()

    def run(self):
        args = ["bokeh", "serve"]
        if self.run_in_notebook:
            args.append("--allow-websocket-origin=localhost:8888")
        self.p = subprocess.Popen(args, env=os.environ.copy())

    def join(self, timeout=None):
        if self.p:
            print("Killing bokeh server thread {}".format(self.p.pid))
            for child_proc in psutil.Process(self.p.pid).children():
                print("Killing child process {}".format(child_proc.pid))
                child_proc.kill()
            self.p.kill()
            self.p = None
            super(BokehServerThread, self).join(timeout=timeout)
