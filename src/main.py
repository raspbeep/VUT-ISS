import numpy as np
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
import copy

Fs, data = wav.read('/home/pavel/iss-project/audio/xkrato61.wav')

print(f"Sample rate: {Fs} Hz")  # 16 kHz
print(f"Length: {len(data)/Fs} s")  # 0.5 seconds
print(f"Length: {np.size(data)} samples")
print(f"Maximum in samples: {np.amax(data)}")
print(f"Minimum in samples: {np.amin(data)}")

# Plot the data in the time domain

time = np.linspace(0, len(data)/Fs, num=len(data))
plt.figure(figsize=(10, 4))
plt.stem(time, data, label='Signal')
plt.legend()
plt.show()


# Normalize the data
orig_data = copy.deepcopy(data)
data = data - np.mean(data)
data = data / np.abs(data).max()

# +1 to fit the remaining samples into the last column of matrix
# using 90 frames instead of possible (uncomplete) 92 frames
matrix = np.zeros((1024, np.size(data)//512 - 1))
print(np.size(data)//512)


for i in range(0, 90):
    matrix[:, [i]] = data[i*512: i*512 + 1024].reshape((1024, 1))

# Prettyprint for matrix (for debugging)
# np.set_printoptions(precision=3)

# Plot one frame in the time domain
# 45 je nice
frame_number = 18
""" frame = matrix[:, 45]
time = np.linspace(0, len(frame)/Fs, num=len(frame))
plt.figure(figsize=(10, 4))
plt.plot(time, frame, label='Signal')
plt.legend()
plt.show()
 """


frame = orig_data[frame_number*512: 45*512 + 1024]
fram_fft = np.fft.fft(frame)
fft_fre = np.fft.fftfreq(n=np.size(frame), d=1/Fs)
plt.figure(figsize=(10, 4))
plt.plot(fft_fre, fram_fft.real, label='Signal')
plt.plot(fft_fre, fram_fft.imag, label='Signal')
plt.legend()
plt.show()
