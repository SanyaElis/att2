import math

import numpy as np
from scipy.fft import rfft, rfftfreq, irfft
import matplotlib.pyplot as plt
import scipy.signal as signal


class Signal:
    accuracy = 1000

    def __init__(self, frequency, duration, amplitude):
        self.frequency = frequency
        self.duration = duration
        self.amplitude = amplitude
        self.t = []
        self.x = []
        self.tf = None
        self.xf = None

    def count_signal(self):
        pass

    def count_spectrum(self):
        self.xf = rfft(self.x)
        self.xf[0] = 0
        self.tf = rfftfreq(int(self.duration * self.accuracy), 1 / self.accuracy)

    def plot_graphic(self):
        self.count_spectrum()
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        plt.title(self.get_name())
        plt.xlabel('t, c')
        plt.ylabel('x')
        plt.grid(True)
        plt.plot(self.t, self.x)
        plt.subplot(2, 1, 2)
        plt.title('Спектр сигнала')
        plt.grid(True)
        plt.xlabel('Частота, Гц')
        plt.ylabel('Мощность')
        plt.xlim(0, 100)
        plt.plot(self.tf, abs(self.xf))  # модуль из-за того, что значения комплексные
        plt.show()

    def get_name(self):
        return "Error"


class HarmonicSignal(Signal):

    def __init__(self, frequency, duration, amplitude):
        super().__init__(frequency, duration, amplitude)
        self.count_signal()

    def count_signal(self):
        i = 0
        while i <= self.duration:
            self.x.append(self.amplitude * math.cos(2 * math.pi * self.frequency * i))
            self.t.append(i)
            i = i + 1 / self.accuracy

    def get_name(self):
        return f'График гармонического сигнала (косинуса), частота {self.frequency} гц'


class MeanderSignal(Signal):

    def __init__(self, frequency, duration, amplitude):
        super().__init__(frequency, duration, amplitude)
        self.count_signal()

    def count_signal(self):
        i = 0
        while i <= self.duration:
            self.x.append(1 if self.amplitude * math.sin(2 * math.pi * self.frequency * i) > 0 else 0)
            self.t.append(i)
            i = i + 1 / self.accuracy

    def get_name(self):
        return f'График цифрового (однополярного меандра) сигнала, частота {self.frequency} гц'


class ModulatedSignal(Signal):

    def __init__(self, frequency, duration, amplitude, carrier_signal):
        super().__init__(frequency, duration, amplitude)
        self.carrier_signal = carrier_signal
        self.t = self.carrier_signal.t


class AmplitudeSCModulatedSignal(ModulatedSignal):

    def __init__(self, frequency, duration, amplitude, carrier_signal):
        super().__init__(frequency, duration, amplitude, carrier_signal)
        self.recovery_xf = None
        self.synthesized_xf = None
        self.positive_envelope = None
        self.count_signal()

    def count_signal(self):
        for i in range(len(self.t)):
            self.x.append(
                self.amplitude * math.cos(2 * math.pi * self.frequency * self.t[i]) * self.carrier_signal.x[i])

    def get_name(self):
        return f'График Ампитудной модуляции однополярным меандром с частотой {self.carrier_signal.frequency} ' \
               f'\nгармонического сигнала (косинуса) с частотой {self.frequency} гц '

    def cut_high_and_low_frequencies(self):
        max_value_index = 0
        filter_kf = 0.5
        for index, value in enumerate(self.xf):
            if abs(self.xf[max_value_index]) < value:
                max_value_index = index
        for index, value in enumerate(self.xf):
            if abs(value) < filter_kf * abs(self.xf[max_value_index]):
                self.xf[index] = 0

    def synthesized_signal(self):
        self.synthesized_xf = irfft(self.xf)

    def envelope_signal(self):
        envelope = np.abs(signal.hilbert(self.synthesized_xf))
        self.positive_envelope = np.maximum(0, envelope)

    def meander_recovery(self):
        self.recovery_xf = []
        filter_kf = 0.5
        for value in self.positive_envelope:
            self.recovery_xf.append(1 if value > filter_kf else 0)

    def plot_recovery_meander(self):
        self.cut_high_and_low_frequencies()
        self.synthesized_signal()
        self.envelope_signal()
        self.meander_recovery()
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        plt.title("Синтезированный сигнал и его огибающая")
        plt.xlabel('t, c')
        plt.ylabel('x')
        plt.grid(True)
        plt.plot(self.t, self.synthesized_xf)
        plt.plot(self.t, self.positive_envelope)
        plt.legend(['Синтезированный', 'Огибающая'])
        plt.subplot(2, 1, 2)
        plt.title('Восстановленный меандр')
        plt.grid(True)
        plt.xlabel('t, c')
        plt.ylabel('x')
        plt.plot(self.t, self.recovery_xf)
        plt.show()


class FrequencySCModulatedSignal(ModulatedSignal):

    def __init__(self, frequency, duration, amplitude, carrier_signal):
        super().__init__(frequency, duration, amplitude, carrier_signal)
        self.count_signal()

    def count_signal(self):
        for i in range(len(self.t)):
            self.x.append(
                self.amplitude * math.cos(
                    2 * math.pi * (self.frequency + self.carrier_signal.x[i] * self.frequency / 2) *
                    self.t[i]))

    def get_name(self):
        return f'График Частотной модуляции однополярным меандром с частотой {self.carrier_signal.frequency} ' \
               f'\nгармонического сигнала (косинуса) с частотой {self.frequency} гц '


class PhaseSCModulatedSignal(ModulatedSignal):

    def __init__(self, frequency, duration, amplitude, carrier_signal):
        super().__init__(frequency, duration, amplitude, carrier_signal)
        self.count_signal()

    def count_signal(self):
        for i in range(len(self.t)):
            self.x.append(
                self.amplitude * math.cos(
                    2 * math.pi * self.frequency *
                    self.t[i] + math.pi * self.carrier_signal.x[i]))

    def get_name(self):
        return f'График Фазовой модуляции однополярным меандром с частотой {self.carrier_signal.frequency} ' \
               f'\nгармонического сигнала (косинуса) с частотой {self.frequency} гц '


ASCM = AmplitudeSCModulatedSignal(30, 1, 1, MeanderSignal(2, 1, 1))
ASCM.plot_graphic()
ASCM.plot_recovery_meander()
FSCM = FrequencySCModulatedSignal(32, 1, 1, MeanderSignal(2, 1, 1))
FSCM.plot_graphic()
PSCM = PhaseSCModulatedSignal(32, 1, 1, MeanderSignal(2, 1, 1))
PSCM.plot_graphic()
