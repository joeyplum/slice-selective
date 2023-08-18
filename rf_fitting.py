# %%
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Load RF vector from Philips software
try:
    rf = np.load("rf_default.npy")
except:
    rf = np.array([66, -138, -359, -595, -845, -1107, -1378, -1658, -1943, -2230, -2517, -2800, -3076, -3341, -3592, -3824, -4034, -4217, -4370, -4488, -4567, -4602, -4591, -4530, -4414, -4240, -4007, -3710, -3348, -2918, -2421, -1853, -1216, -510, 266, 1110, 2019, 2991, 4024, 5114, 6256, 7446, 8678, 9947, 11247, 12571, 13911, 15262, 16614, 17960,
                   19293, 20604, 21885, 23129, 24326, 25471, 26554, 27570, 28511, 29371, 30144, 30826, 31410, 31894, 32274, 32547, 32712, 32767, 32712, 32547, 32274, 31894, 31410, 30826, 30144, 29371, 28511, 27570, 26554, 25471, 24326, 23129, 21885, 20604, 19293, 17960, 16614, 15262, 13911, 12571, 11247, 9947, 8678, 7446, 6256, 5114, 4024, 2991, 2019, 1110, 266])
    np.save("rf_default.npy", rf)

try:
    gz = np.load("gz_default.npy")
except:
    print("Could not load the g(z) pulse. Run and save the pulse shape from rf_bloch_simulation.py.")

# Normalize gz to slice thickness
TH = 15
z = np.linspace(-1.5*TH, 1.5*TH, len(gz))

# Plot
plt.figure(1, figsize=(10, 14))
plt.subplot(211)
plt.plot(rf)
plt.grid(True)
plt.xlabel('time (ms)')
plt.ylabel('RF (mT)')
plt.title('RF Pulse')

plt.subplot(212)
plt.plot(z, gz, 'r.')
plt.grid(True)
plt.xlabel('time (ms)')
plt.ylabel('RF (mT)')
