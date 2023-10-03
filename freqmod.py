import matplotlib.pyplot as plt
import numpy as np
from math import pi

plt.close('all')

Fs = 2000
t = np.arange(0, 0.2, 1 / Fs)

fc = 100  # carrier wave
fm = 15  # signal frequency

# modulating index
b = 3
frm = np.cos(2 * pi * fc * t + b * np.sin(2 * pi * fm * t))
m = np.cos(2 * pi * fm * t)

plt.plot(t, frm)
plt.plot(t, m)
plt.title('Frequency Modulated Signal')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.legend(['FM signal', 'Message Signal'])
plt.show()