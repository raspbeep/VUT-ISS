import numpy as np
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt

Fs, data = wav.read('/home/pavel/iss-project/audio/xkrato61.wav')

print(f"Sample rate: {Fs} Hz")  # 16 kHz
print(f"Length: {len(data)/Fs} s")  # 0.5 seconds
print(f"Length: {np.size(data)} samples")
print(f"Maximum in samples: {np.amax(data)}")
print(f"Minimum in samples: {np.amin(data)}")

# Plot the data in the time domain
'''
time = np.linspace(0, len(data)/Fs, num=len(data))
plt.figure(figsize=(10, 4))
plt.plot(time, data, label='Signal')
plt.legend()
plt.show()
'''

# Normalize the data
data = data - np.mean(data)
data = data / np.abs(data).max()

# +1 to fit the remaining samples into the last column of matrix
matrix = np.zeros((1024, (np.size(data)//512) + 1))

for i in range(0, (np.size(data)//512) + 1):
    if i*512 + 1024 > np.size(data):
        print(data[i*512:].reshape(1024, 1))
        print(np.size(data[i*512:].reshape(1024, 1)))
    else:
        print(data[i*512: i*512 + 1024].reshape(1024, 1))
        print(np.size(data[i*512: i*512 + 1024].reshape(1024, 1)))

    #matrix[:, [i]] = data[i*512: i*512 + 1024].reshape((1024, 1))
