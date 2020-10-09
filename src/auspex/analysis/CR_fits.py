# Copyright 2019 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from auspex.log import logger
from enum import Enum
from .fits import AuspexFit

class CR_cal_type(Enum):
    LENGTH = 1
    PHASE = 2
    AMP = 3


class SineFit(AuspexFit):

    def __init__(self, xpts, ypts, initial_phase, initial_freq):

        self.initial_phase = initial_phase
        self.initial_freq = initial_freq
        super().__init__(xpts, ypts, make_plots=False)

    def _model(self, x, *p):
        return p[1]*np.sin(2*np.pi*p[0]*x + p[2]) + p[3]

    def _initial_guess(self):
        return [self.initial_freq, 1.0, self.initial_phase, 0.0]

    def _fit_dict(self, p):
        return {"f": p[0],
                "A": p[1],
                "phi": p[2],
                "y0": p[3]}

    def __str__(self):
        return "A*sin(2*pi*f*x + phi) + y0"


def fit_CR(xpoints, data, cal_type):

    data0 = data[:len(data)//2]
    data1 = data[len(data)//2:]
    xpoints = [xp if len(xp)>1 else xp[0] for xp in xpoints]

    if cal_type == CR_cal_type.LENGTH:
        return fit_CR_length(xpoints, data0, data1)
    elif cal_type == CR_cal_type.PHASE:
        return fit_CR_phase(xpoints, data0, data1)
    elif cal_type == CR_cal_type.AMP:
        return fit_CR_amp(xpoints, data0, data1)

def fit_CR_length(xpoints, data0, data1):

    xpoints = xpoints[0]
    x_fine = np.linspace(min(xpoints), max(xpoints), 1001)

    fit0 = SineFit(xpoints, data0, np.pi/2.0, 1/(2.0*xpoints[-1]))
    fit1 = SineFit(xpoints, data1, np.pi/2.0, 1/(2.0*xpoints[-1]))

    #find the first zero crossing
    delta = 2*(x_fine[1] - x_fine[0])
    idx0 = int(1.0 / np.abs(fit0.fit_params["f"]) / delta)
    idx1 = int(1.0 / np.abs(fit1.fit_params["f"]) / delta)

    yfit0 = fit0.model(x_fine[:idx0])
    yfit1 = fit1.model(x_fine[:idx1])
    #average between the two qc states, rounded to 10 ns
    xopt = round((x_fine[np.argmin(abs(yfit0))] + x_fine[np.argmin(abs(yfit1))])/2/10e-9)*10e-9
    logger.info('CR length = {} ns'.format(xopt*1e9))

    return xopt, fit0.fit_params, fit1.fit_params

def fit_CR_phase(xpoints, data0, data1):

    xpoints = xpoints[1]
    x_fine = np.linspace(min(xpoints), max(xpoints), 1001)

    fit0 = SineFit(xpoints, data0, np.pi, 1.0/xpoints[-1])
    fit1 = SineFit(xpoints, data1, np.pi, 1.0/xpoints[-1])
    #find the phase for maximum contrast
    contrast = (fit0.model(x_fine) - fit1.model(x_fine))/2.0
    logger.info(f"CR Contrast = {np.min(contrast)}")
    xopt = x_fine[np.argmin(contrast)] % (2*np.pi)

    logger.info(f"CR phase = {xopt}")

    return xopt, fit0.fit_params, fit1.fit_params


def fit_CR_amp(xpoints, data0, data1):
    xpoints = xpoints[2]
    x_fine = np.linspace(min(xpoints), max(xpoints), 1001)
    popt0 = np.polyfit(xpoints, data0, 1) # tentatively linearize
    popt1 = np.polyfit(xpoints, data1, 1)
    #average between optimum amplitudes
    xopt = -(popt0[1]/popt0[0] + popt1[1]/popt1[0])/2
    logger.info('CR amplitude = {}'.format(xopt))
    return xopt, popt0, popt1
