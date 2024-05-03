# Wyatt
import smbus
import time
import matplotlib.pyplot as plt
import numpy as np

def get_ecg_point(bus, address=0x48, channel=0x42):
    bus.write_byte(address, channel)
    value = bus.read_byte(address)
    normalized_value = value / 255.0  # Normalize the value
    return normalized_value

# Initialize plot
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlim(0, 3600)  # Assuming duration of 10s at 360Hz
ax.set_ylim(0, 1)  # Normalized ECG values are between 0 and 1

# Initialize bus
bus = smbus.SMBus(1)

# Get and plot ECG data in real-time
freq = 360
duration = 10
ecg_signals = []
for i in range(int(duration * freq)):
    new_data = get_ecg_point(bus)  # Get one data point
    ecg_signals.append(new_data)
    line.set_ydata(ecg_signals)
    line.set_xdata(np.arange(len(ecg_signals)))
    plt.draw()
    plt.pause(1.0 / freq)

bus.close()
plt.ioff()
plt.show()
