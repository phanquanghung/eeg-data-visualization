import streamlit as st
import time
import numpy as np
import mne
from scipy import signal
import matplotlib.pyplot as plt


def rawplot():
    raw.plot()
    st.pyplot()


def topo():
    mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info)
    st.pyplot()


def stem():
    plt.semilogy(f, Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    plt.stem(f, Pxx_den)
    plt.show()
    st.pyplot()


def bar():
    # creating the bar plot
    plt.bar(courses, values, color='maroon',
            width=0.4)
    plt.ylim(0, 2e-9)
    # plt.xlabel("Courses offered")
    # plt.ylabel("No. of students enrolled")
    plt.title("PSD")
    plt.show()
    st.pyplot()


st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('EEG Data Visualization')

uploaded_file = st.sidebar.file_uploader("Choose a file", type='vhdr')
buttonPlot = st.sidebar.button('Raw Data')
st.sidebar.write('Visualization of raw data.')
buttonActi = st.sidebar.button('Activation Region')
st.sidebar.write('Plot a topographic map as image.')
buttonStem = st.sidebar.button('Stem')
st.sidebar.write('Create a stem plot.')
buttonPSD = st.sidebar.button('Power Spectral Density')
st.sidebar.write('Return PSD estimation of EEG signal over alpha, beta, theta and delta bands.')
# Load raw data
data_path = 'data/101_AgrLexAux_s1.vhdr'
raw = mne.io.read_raw_brainvision(data_path, preload=True, verbose=False)
raw.info['line_freq'] = 50.
st.write(raw.n_times)
raw.crop(tmin=5., tmax=10.)


biosemi_montage = mne.channels.make_standard_montage('biosemi32')
fake_info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=1000,
                            ch_types='eeg')

data, times = raw[:]
fake_evoked = mne.EvokedArray(data, fake_info)
fake_evoked.set_montage(biosemi_montage)

f, Pxx_den = signal.welch(data[:, 3500], fs=60)

delta = Pxx_den[0] * Pxx_den[0]
for i in range(1, 5):
    delta += Pxx_den[i] * Pxx_den[i]

delta *= 1e11
theta = Pxx_den[5] * Pxx_den[5]
for i in range(6, 8):
    theta += Pxx_den[i] * Pxx_den[i]
theta *= 1e11
alpha = Pxx_den[8] * Pxx_den[8]
for i in range(9, 13):
    alpha += Pxx_den[i] * Pxx_den[i]
alpha *= 1e11
beta = Pxx_den[13] * Pxx_den[13]
for i in range(13, 17):
    beta += Pxx_den[i] * Pxx_den[i]
beta *= 1e11

data_bar = {'delta': delta, 'theta': theta, 'alpha': alpha,
            'beta': beta}
courses = list(data_bar.keys())
values = list(data_bar.values())

fig = plt.figure(figsize=(10, 5))

if buttonPlot:
    rawplot()

if buttonActi:
    topo()

if buttonStem:
    stem()

if buttonPSD:
    bar()

st.write(raw.n_times)