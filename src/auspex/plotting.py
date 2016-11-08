# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import threading
import subprocess
import psutil
import os
import sys

from auspex.log import logger

class BokehServerThread(threading.Thread):
    def __init__(self, notebook=False):
        super(BokehServerThread, self).__init__()
        self.daemon = True
        self.run_in_notebook = notebook

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
