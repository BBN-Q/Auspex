from pycontrol.instruments.tektronix import DPO72004C
import numpy as np

if __name__ == '__main__':
    scope = DPO72004C("Test Scope", "192.168.5.107")
    print(scope.interface.query("*IDN?"))
