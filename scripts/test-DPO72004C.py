from pycontrol.instruments.tektronix import DPO72004C
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    scope = DPO72004C("Test Scope", "192.168.5.107")
    print(scope.interface.query("*IDN?"))
    curve = scope.get_curve()
    print(curve.shape)
    # curve = np.sum(curve, axis=0)
    plt.plot(curve.T)
    plt.show()
