# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:21:36 2025

@author: Titan.Hartono
"""

import time
from  keithley2600 import Keithley2600, ResultTable
import matplotlib.pyplot as plt

# k = Keithley2600('TCPIP0::192.168.2.121::INSTR')
k = Keithley2600('TCPIP0::169.254.0.1::inst0::INSTR')

# create ResultTable with two columns
rt = ResultTable(
    column_titles=['Voltage', 'Current'],
    units=['V', 'A'],
    params={'recorded': time.asctime(), 'sweep_type': 'iv'},
)

# create live plot which updates as data is added
rt.plot(live=True)

# measure some currents
for v in range(0, 240):
    k.apply_voltage(k.smua, v/100)
    time.sleep(0.01)
    i = k.smua.measure.i()
    rt.append_row([v/100, i])

# save the data
# rt.save('~/iv_curve.txt')

data = rt.data
plt.plot(data[:,0], data[:,1])