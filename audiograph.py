import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
#wfrom tkinter import filedialog
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer")

        # Load GIF (will load as a png)
        gif_path = "music.gif" 
        self.gif = tk.PhotoImage(file=gif_path)

        self.gif_label = tk.Label(root, image=self.gif)
        self.gif_label.pack()



        self.file_path = ""
        self.audio_data = None
        self.frame_rate = None
        self.channels = 1  # mono channel

        self.load_button = tk.Button(root, text="Load Audio", command=self.load_audio)
        self.load_button.pack(pady=10)

        self.process_button = tk.Button(root, text="Process Audio", command=self.process_audio)
        self.process_button.pack(pady=10)
        self.process_button.config(state=tk.DISABLED)

        self.plot_button = tk.Button(root, text="Plot", command=self.plot_audio)
        self.plot_button.pack(pady=10)
        self.plot_button.config(state=tk.DISABLED)

        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()

    def load_audio(self):
        self.file_path = filedialog.askopenfilename(title="Select Audio File",
                                                    filetypes=[("Audio Files", "*.wav;*.mp3;*.aac")])
        if self.file_path:
            self.load_button.config(text=f"Loaded: {os.path.basename(self.file_path)}")
            self.audio_data, self.frame_rate = sf.read(self.file_path, always_2d=True)
            self.channels = self.audio_data.shape[1]
            self.process_button.config(state=tk.NORMAL)

    def process_audio(self):
        # Checks if the audio file is in wav format
        if not self.file_path.lower().endswith('.wav'):
            sf.write(self.file_path, self.audio_data[:, 0], self.frame_rate)
            self.file_path = self.file_path[:-4] + '.wav'

        self.plot_button.config(state=tk.NORMAL)

    def plot_audio(self):
        time_seconds = len(self.audio_data) / self.frame_rate
        print(f"Time: {time_seconds:.2f} seconds")

        # finds the frequency of the greatest amplitude
        samples = self.audio_data[:, 0]
        frequencies, times, amplitudes = spectrogram(samples, fs=self.frame_rate, nperseg=1024)
        max_amplitude_freq = frequencies[np.argmax(np.mean(amplitudes, axis=1))]
        print(f"Frequency of Greatest Amplitude: {max_amplitude_freq:.2f} Hz")

        # Compute RT60 for Low, Mid, High frequencies
        freq_range = [125, 500, 2000]  # Low, Mid, High frequencies
        for freq in freq_range:
            rt60 = self.compute_rt60(samples, self.frame_rate, freq)
            print(f"RT60 for {freq} Hz: {rt60:.2f} seconds")

        # Additional plot (frequency spectrum)
        plt.figure()
        plt.pcolormesh(times, frequencies, 10 * np.log10(amplitudes + 1e-10), shading='auto', cmap='viridis')
        plt.title("Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Power/Frequency (dB/Hz)")
        plt.show()

        # shows differences in RT60 values
        rt60_diff_low_mid = abs(self.compute_rt60(samples, self.frame_rate, freq_range[0])
                                - self.compute_rt60(samples, self.frame_rate, freq_range[1]))
        rt60_diff_mid_high = abs(self.compute_rt60(samples, self.frame_rate, freq_range[1])
                                 - self.compute_rt60(samples, self.frame_rate, freq_range[2]))
        rt60_diff_high_low = abs(self.compute_rt60(samples, self.frame_rate, freq_range[2])
                                 - self.compute_rt60(samples, self.frame_rate, freq_range[0]))
        print(
            f"RT60 Differences: Low-Mid: {rt60_diff_low_mid:.2f}s, Mid-High: {rt60_diff_mid_high:.2f}s, High-Low: {rt60_diff_high_low:.2f}s")

    @staticmethod
    def compute_rt60(samples, frame_rate, frequency):
        fft_size = 1024
        signal_fft = np.fft.fft(samples, fft_size)
        signal_power = np.abs(signal_fft) ** 2
        frequency_bins = np.fft.fftfreq(fft_size, d=1 / frame_rate)
        index = np.argmin(np.abs(frequency_bins - frequency))
        power_at_frequency = signal_power[index]

        threshold = power_at_frequency / 1000.0  # Adjust the threshold 
        decay_curve = np.argmax(signal_power[index:] < threshold)
        rt60 = decay_curve / frame_rate
        return rt60
# The main function after all the code
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()