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
import tempfile
import time

from auspex.log import logger

class BokehServerProcess(object):
    def __init__(self, notebook=False):
        super(BokehServerProcess, self).__init__()
        self.run_in_notebook = notebook
        self.pid_filename = os.path.join(tempfile.gettempdir(), "auspex_bokeh.pid")

    def run(self):
        # start a Bokeh server if one is not already running
        pid = self.read_session_pid()
        if pid:
            self.p = psutil.Process(pid)
            logger.info("Using existing Bokeh server")
            return
        logger.info("Starting Bokeh server")
        args = ["bokeh", "serve", "--port", "5006", "--allow-websocket-origin=localhost:8888", "--allow-websocket-origin=localhost:8889", "--allow-websocket-origin=localhost:8890"]
        self.p = subprocess.Popen(args, env=os.environ.copy())
        self.write_session_pid()
        # sleep to give the Bokeh server a chance to start
        # TODO replace this with some bokeh client API call that
        # verifies that the server is running
        time.sleep(3)

    def terminate(self):
        if self.p:
            print("Killing bokeh server process {}".format(self.p.pid))
            try:
                for child_proc in psutil.Process(self.p.pid).children():
                    print("Killing child process {}".format(child_proc.pid))
                    child_proc.terminate()
            except:
                print("Couldn't kill child processes.")
            self.p.terminate()
            self.p = None
            os.remove(self.pid_filename)

    def write_session_pid(self):
        with open(self.pid_filename, "w") as f:
            f.write("{}\n".format(self.p.pid))

    def read_session_pid(self):
        # check if pid file exists
        if not os.path.isfile(self.pid_filename):
            return None
        with open(self.pid_filename) as f:
            pid = int(f.readline())
        # check that a process is running on that PID
        if not psutil.pid_exists(pid):
            return None
        # check that the process is a Bokeh server
        cmd = psutil.Process(pid).cmdline()
        if any('bokeh' in item for item in cmd):
            return pid
        return None
