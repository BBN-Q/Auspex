# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from pycontrol.instruments.tektronix import DPO72004C
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    scope = DPO72004C("192.168.5.107")
    print(scope.interface.query("*IDN?"))
    curve = scope.get_curve()
    print(curve.shape)
    # curve = np.sum(curve, axis=0)
    plt.plot(curve.T)
    plt.show()
