import matplotlib.pyplot as plt
import math
from scipy.fft import rfft, rfftfreq
import numpy as np

accuracy = 1000


def count_cos(A, freq, dur, x0):
    t = []
    x = []
    i = 0
    while i <= dur:
        x.append(A * math.cos(2 * math.pi * freq * i) + x0)
        t.append(i)
        i = i + 1 / accuracy
    return t, x


def count_meander(A, freq, dur):
    t = []
    x = []
    i = 0
    while i <= dur:
        x.append(1 if A * math.sin(2 * math.pi * freq * i) > 0 else 0)
        # x.append(np.sign(A * math.sin(2 * math.pi * freq * i)))
        t.append(i)
        i = i + 1 / accuracy
    return t, x


def count_spectrum(signal, dur, acc):
    t, x = signal
    xf = rfft(x)
    tf = rfftfreq(int(dur * acc), 1 / acc)
    return tf, xf


t1, x1 = count_meander(1.0, 1, 5)
t2, x2 = count_cos(1.0, 5, 5, 0)

x3 = [x * y for x, y in zip(x1, x2)]
sig = t2, x3
t4, x4 = count_spectrum(sig, 5, accuracy)


plt.figure(figsize=(12, 10))
plt.subplot(411)
plt.plot(t1, x1)
plt.subplot(412)
plt.plot(t2, x2)
plt.subplot(413)
plt.plot(t1, x3)
plt.subplot(414)
plt.xlim(0, 40)
plt.plot(t4, abs(x4))
plt.show()
