import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter

def correlate(sig, sig1):
    return np.array([np.sum(sig[i:]*sig1[:len(sig1)-i]) for i in range(len(sig))])

def pitch(frame, thresh):
    lag = np.argmax(frame[thresh:]) + thresh
    return Fs / lag

x, Fs = librosa.load("xkrato61.wav")
print(Fs)
plt.figure(figsize=(15,15))
plt.plot(x)
plt.show()

frame_len = int(0.05 * Fs)
shift = int(0.05 * Fs)

frames = np.array([x[i*shift:i*shift + frame_len] for i in range(len(x) // shift - frame_len // shift + 1)])

filtered_X = np.array([np.abs(np.fft.fft(x, n=4096)) for x in frames])

f_axis = np.arange(2048)
plt.figure(figsize=(15,15))
plt.plot(f_axis/4096 * Fs, 10*np.log(filtered_X[3][:2048]**2))
plt.xlabel('frekvence')
plt.show()


# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a


def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

spectras = []
frame = filtered_X[3][:2048]
spectras.append(frame)
plt.figure(figsize=(10,10))
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html
w, h = signal.freqz(*butter_bandstop(10000-50, 10000+50, Fs))
plt.plot(w/np.pi*16000, 20 * np.log10(abs(h)), 'b')  # freqz vrati w v rozmezi 0 - pi
plt.title('Frekvenční charakteristika filtru, typu pásmová zádrž, kt. odstraní frekvence kolem 10 kHz')
plt.xlabel('Frekvence [Hz]')
plt.ylabel('Intenzita')
plt.show()
xfilts = []
xfilts.append(x)
x_filt = x
# vytvorime si filtry pro frekvence na nasobcich 1 kHz
for band_f in range(1000, 16000, 1000):
    #band_f = band / 2048 * Fs
    x_filt = butter_bandstop_filter(x_filt, band_f-50, band_f+50, Fs)
    xfilts.append(x_filt)
    frame = np.abs(np.fft.fft(x_filt[3*shift:3*shift + frame_len], n=4096)[:2048])
    #print(frame)
    spectras.append(frame)

f, ax = plt.subplots(len(spectras), figsize=(15, len(spectras)*5))
for s in range(len(spectras)):
    ax[s].set_title(f'Spektrum signalu po odstraneni {s} nasobku 1000 Hz')
    ax[s].plot(np.arange(2048)/2048 * 16000, spectras[s].T)
plt.tight_layout()
plt.show()