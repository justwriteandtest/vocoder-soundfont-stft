import sys
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
import fluidsynth
import subprocess
import tempfile
import traceback

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                                 QFileDialog, QSpinBox, QDoubleSpinBox, QTextEdit, 
                                 QGroupBox, QMessageBox, QCheckBox)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
except ImportError:
    print("PyQt6 not found. Please install it using: pip install PyQt6")
    sys.exit(1)

# --- Core Logic (Same as before) ---

def get_pitch_contour(audio, sr, hop_size=512, frame_size=2048):
    """
    Estimates the pitch contour (MIDI notes) of the audio frame-by-frame.
    Returns an array of MIDI note values (float).
    """
    # STFT
    # STFT
    win = hann(frame_size)
    sft = ShortTimeFFT(win, hop=hop_size, fs=sr, scale_to='magnitude')
    Zxx = sft.stft(audio)
    magnitude = np.abs(Zxx)
    
    n_frames = magnitude.shape[1]
    pitch_contour = np.zeros(n_frames)
    
    # HPS Parameters
    n_fft = frame_size
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    for i in range(n_frames):
        frame_spec = magnitude[:, i]
        
        # Simple HPS per frame
        hps_spec = frame_spec.copy()
        for h in range(2, 4): # 2, 3 harmonics
            decimated = frame_spec[::h]
            hps_spec = hps_spec[:len(decimated)] * decimated
            
        # Ignore low freq noise
        ignore_bins = int(50 * n_fft / sr)
        hps_spec[:ignore_bins] = 0
        
        peak_bin = np.argmax(hps_spec)
        f0 = peak_bin * sr / n_fft
        
        if f0 > 50:
            midi_val = 69 + 12 * np.log2(f0 / 440)
            pitch_contour[i] = midi_val
        else:
            pitch_contour[i] = 0 # Unvoiced/Silence
            
    # Simple smoothing (median filter)
    from scipy.signal import medfilt
    pitch_contour = medfilt(pitch_contour, kernel_size=5)
    
    return pitch_contour

def render_soundfont_carrier(sf_path, duration, sample_rate, midi_note=48, bank_num=0, preset_num=0, pitch_contour=None, hop_size=512):
    fs = fluidsynth.Synth()
    fs.setting('audio.period-size', 64)
    fs.setting('synth.sample-rate', float(sample_rate))
    
    sfid = fs.sfload(sf_path)
    if sfid == -1:
        raise RuntimeError(f"Gagal memuat SoundFont: {sf_path}")
    
    fs.program_select(0, sfid, bank_num, preset_num)
    
    total_samples = int(sample_rate * duration)
    
    if pitch_contour is None:
        # Static Note Mode
        fs.noteon(0, midi_note, 100)
        raw_audio = fs.get_samples(total_samples)
        audio_data = np.array(raw_audio).astype(np.float32)
        audio_mono = audio_data[::2]
        
    else:
        # Dynamic Pitch Mode
        # 1. Set Pitch Bend Range to +/- 24 semitones (RPN 0,0)
        fs.cc(0, 101, 0) # RPN MSB
        fs.cc(0, 100, 0) # RPN LSB
        fs.cc(0, 6, 24)  # Data Entry MSB (24 semitones)
        fs.cc(0, 38, 0)  # Data Entry LSB
        
        # Base note is the median of the contour (ignoring 0s)
        valid_notes = pitch_contour[pitch_contour > 0]
        if len(valid_notes) > 0:
            base_note = int(np.median(valid_notes))
        else:
            base_note = midi_note
            
        fs.noteon(0, base_note, 100)
        
        audio_mono = np.zeros(total_samples, dtype=np.float32)
        current_sample = 0
        
        # Interpolate contour to sample rate or process in chunks?
        # Processing in chunks matching the STFT hop size is easier
        
        samples_per_hop = hop_size
        
        for i, target_note in enumerate(pitch_contour):
            if current_sample >= total_samples:
                break
                
            # Calculate Pitch Bend
            if target_note > 0:
                bend_semitones = target_note - base_note
                # Clamp to range
                bend_semitones = max(-24, min(24, bend_semitones))
                
                # Map -24..24 to 0..16383 (8192 is center)
                # 8192 + (bend / 24) * 8192
                bend_val = int(8192 + (bend_semitones / 24) * 8192)
                bend_val = max(0, min(16383, bend_val))
                
                fs.pitch_bend(0, bend_val)
            
            # Render chunk
            chunk_len = min(samples_per_hop, total_samples - current_sample)
            raw_chunk = fs.get_samples(chunk_len)
            chunk_data = np.array(raw_chunk).astype(np.float32)
            
            # Stereo to Mono
            audio_mono[current_sample:current_sample+chunk_len] = chunk_data[::2]
            current_sample += chunk_len
            
    fs.delete()
    return audio_mono

def vocoder(modulator_signal, carrier_signal, sample_rate, frame_size=2048, hop_size=512, threshold=0.01):
    min_len = min(len(modulator_signal), len(carrier_signal))
    modulator_signal = modulator_signal[:min_len]
    carrier_signal = carrier_signal[:min_len]

    win = hann(frame_size)
    sft = ShortTimeFFT(win, hop=hop_size, fs=sample_rate, scale_to='magnitude')

    Zxx_modulator = sft.stft(modulator_signal)
    Zxx_carrier = sft.stft(carrier_signal)

    modulator_envelope = np.abs(Zxx_modulator)
    if np.max(modulator_envelope) > 0:
        modulator_envelope /= np.max(modulator_envelope)

    if threshold > 0:
        modulator_envelope[modulator_envelope < threshold] = 0

    output_spectrum = Zxx_carrier * modulator_envelope
    output_signal = sft.istft(output_spectrum)
    
    # Trim to original length
    if len(output_signal) > min_len:
        output_signal = output_signal[:min_len]
    elif len(output_signal) < min_len:
        output_signal = np.pad(output_signal, (0, min_len - len(output_signal)))
        
    return output_signal

def spectral_subtraction(vocal_signal, noise_signal, sr, frame_size=2048, hop_size=512, alpha=2.0):
    win = hann(frame_size)
    sft = ShortTimeFFT(win, hop=hop_size, fs=sr, scale_to='magnitude')
    
    # 1. Analyze Noise to get average magnitude profile
    Zxx_noise = sft.stft(noise_signal)
    noise_mag = np.abs(Zxx_noise)
    noise_profile = np.mean(noise_mag, axis=1, keepdims=True) # Average over time
    
    # 2. Analyze Vocal
    Zxx_vocal = sft.stft(vocal_signal)
    vocal_mag = np.abs(Zxx_vocal)
    vocal_phase = np.angle(Zxx_vocal)
    
    # 3. Subtract
    cleaned_mag = vocal_mag - (alpha * noise_profile)
    cleaned_mag = np.maximum(cleaned_mag, 0.0) 
    
    # 4. Reconstruct
    Zxx_cleaned = cleaned_mag * np.exp(1j * vocal_phase)
    cleaned_signal = sft.istft(Zxx_cleaned)
    
    # Trim to original length
    if len(cleaned_signal) > len(vocal_signal):
        cleaned_signal = cleaned_signal[:len(vocal_signal)]
    elif len(cleaned_signal) < len(vocal_signal):
        cleaned_signal = np.pad(cleaned_signal, (0, len(vocal_signal) - len(cleaned_signal)))
        
    return cleaned_signal

def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mp3':
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_wav_path = tmp.name
        try:
            subprocess.run(['ffmpeg', '-i', path, '-y', tmp_wav_path, '-loglevel', 'error'], check=True)
            sr, data = wavfile.read(tmp_wav_path)
            return sr, data
        except Exception as e:
            raise RuntimeError(f"MP3 conversion failed: {e}")
        finally:
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)
    else:
        return wavfile.read(path)

def get_presets(sf_path):
    if not os.path.exists(sf_path):
        return []
    presets = []
    try:
        fs = fluidsynth.Synth()
        sfid = fs.sfload(sf_path)
        if sfid != -1:
            for bank in range(129):
                for preset in range(128):
                    name = fs.sfpreset_name(sfid, bank, preset)
                    if name:
                        presets.append((bank, preset, name))
        fs.delete()
    except Exception:
        pass
    return presets

def detect_pitch(audio_path, sample_rate=44100):
    """
    Estimates the median pitch (MIDI note) of the audio file using HPS.
    """
    try:
        sr, audio = load_audio(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        
        # Simple HPS (Harmonic Product Spectrum)
        # 1. Windowing and FFT
        n_fft = 8192 # High resolution
        if len(audio) < n_fft:
            n_fft = len(audio)
            
        # Use a few chunks from the center of the file to avoid silence at start/end
        center = len(audio) // 2
        chunk_size = n_fft
        start = max(0, center - chunk_size // 2)
        end = min(len(audio), start + chunk_size)
        chunk = audio[start:end]
        
        if len(chunk) < n_fft:
            return 48 # Fallback
            
        window = np.hanning(len(chunk))
        spectrum = np.abs(np.fft.rfft(chunk * window))
        
        # 2. Downsampling and Multiplication
        hps_spec = spectrum.copy()
        for h in range(2, 5): # 2, 3, 4 harmonics
            decimated = spectrum[::h]
            hps_spec = hps_spec[:len(decimated)] * decimated
            
        # 3. Find Peak
        # Ignore very low frequencies (< 50Hz)
        ignore_bins = int(50 * n_fft / sr)
        hps_spec[:ignore_bins] = 0
        
        peak_bin = np.argmax(hps_spec)
        f0 = peak_bin * sr / n_fft
        
        if f0 <= 0:
            return 48
            
        # 4. Convert to MIDI
        # MIDI = 69 + 12 * log2(f0 / 440)
        midi_note = 69 + 12 * np.log2(f0 / 440)
        return int(round(midi_note))
        
    except Exception as e:
        print(f"Pitch detection failed: {e}")
        return 48

# --- Worker Thread ---

class WorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str) # success, message

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params
            self.log_signal.emit(f"Processing: {os.path.basename(p['vocal_path'])} + {os.path.basename(p['sf_path'])}")
            
            sr, modulator = load_audio(p['vocal_path'])
            if len(modulator.shape) > 1:
                modulator = modulator[:, 0]
            modulator = modulator.astype(np.float32)

            # Load Noise
            if os.path.exists(p['noise_path']):
                self.log_signal.emit("Loading noise profile...")
                _, noise = load_audio(p['noise_path'])
                if len(noise.shape) > 1:
                    noise = noise[:, 0]
                noise = noise.astype(np.float32)
                
                self.log_signal.emit("Applying Spectral Subtraction...")
                modulator = spectral_subtraction(modulator, noise, sr)
            else:
                self.log_signal.emit("Warning: Noise file not found, skipping subtraction.")
            
            duration = len(modulator) / sr
            self.log_signal.emit(f"Duration: {duration:.2f}s")
            
            self.log_signal.emit("Analyzing pitch contour...")
            pitch_contour = get_pitch_contour(modulator, sr)
            
            carrier = render_soundfont_carrier(
                p['sf_path'], duration, sr, 
                bank_num=p['bank'], preset_num=p['preset'],
                pitch_contour=pitch_contour
            )
            
            self.log_signal.emit("Applying Vocoder...")
            vocoded = vocoder(modulator, carrier, sr, threshold=p['threshold'])
            
            if np.max(np.abs(vocoded)) > 0:
                vocoded = vocoded / np.max(np.abs(vocoded))
            
            output_int16 = (vocoded * 32767).astype(np.int16)
            wavfile.write(p['output_path'], sr, output_int16)
            
            self.finished_signal.emit(True, f"Saved to {p['output_path']}")
            
        except Exception as e:
            self.finished_signal.emit(False, str(e))
            traceback.print_exc()


# --- Scan Thread ---

class ScanThread(QThread):
    result_signal = pyqtSignal(str)
    def __init__(self, sf_path):
        super().__init__()
        self.sf_path = sf_path
    
    def run(self):
        presets = get_presets(self.sf_path)
        if not presets:
            self.result_signal.emit("No presets found.")
            return
        
        msg = [f"{'Bank':<6} {'Preset':<8} {'Name'}"]
        msg.append("-" * 40)
        for b, p, n in presets:
            msg.append(f"{b:<6} {p:<8} {n}")
        self.result_signal.emit("\n".join(msg))

# --- Pitch Detect Thread ---

class PitchDetectThread(QThread):
    result_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)
    
    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path
        
    def run(self):
        try:
            note = detect_pitch(self.audio_path)
            self.result_signal.emit(note)
        except Exception as e:
            self.error_signal.emit(str(e))

# --- Main Window ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Talking Instrument (SciPy ShortTimeFFT)")
        self.setFixedSize(600, 700)
        
        # Audio Player Setup
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # File Selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        # Vocal Row (Custom with Play Button)
        vocal_row = QHBoxLayout()
        vocal_lbl = QLabel("Vocal File:")
        vocal_lbl.setFixedWidth(80)
        vocal_row.addWidget(vocal_lbl)
        
        self.vocal_edit = QLineEdit()
        self.vocal_edit.textChanged.connect(self.on_vocal_changed)
        vocal_row.addWidget(self.vocal_edit)
        
        btn_browse_vocal = QPushButton("Browse")
        btn_browse_vocal.clicked.connect(self.browse_vocal)
        vocal_row.addWidget(btn_browse_vocal)
        
        btn_play_vocal = QPushButton("▶")
        btn_play_vocal.setFixedWidth(40)
        btn_play_vocal.setToolTip("Play Vocal")
        btn_play_vocal.clicked.connect(self.play_vocal)
        vocal_row.addWidget(btn_play_vocal)
        
        file_layout.addLayout(vocal_row)

        # Noise Row
        noise_row = QHBoxLayout()
        noise_lbl = QLabel("Noise File:")
        noise_lbl.setFixedWidth(80)
        noise_row.addWidget(noise_lbl)
        
        self.noise_edit = QLineEdit()
        noise_row.addWidget(self.noise_edit)
        
        btn_browse_noise = QPushButton("Browse")
        btn_browse_noise.clicked.connect(self.browse_noise)
        noise_row.addWidget(btn_browse_noise)
        
        btn_play_noise = QPushButton("▶")
        btn_play_noise.setFixedWidth(40)
        btn_play_noise.setToolTip("Play Noise")
        btn_play_noise.clicked.connect(self.play_noise)
        noise_row.addWidget(btn_play_noise)
        
        file_layout.addLayout(noise_row)
        
        # Other rows
        self.sf_edit = self.create_file_row(file_layout, "SoundFont:", "Browse", self.browse_sf)
        self.out_edit = self.create_file_row(file_layout, "Output File:", "Save As", self.browse_out, "vocoder_output.wav")
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()
        
        # Bank/Preset Row
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Bank:"))
        self.bank_spin = QSpinBox()
        self.bank_spin.setRange(0, 128)
        row1.addWidget(self.bank_spin)
        
        row1.addWidget(QLabel("Preset:"))
        self.preset_spin = QSpinBox()
        self.preset_spin.setRange(0, 127)
        row1.addWidget(self.preset_spin)
        
        btn_list = QPushButton("List Presets")
        btn_list.clicked.connect(self.list_presets)
        row1.addWidget(btn_list)
        param_layout.addLayout(row1)
        
        # Note/Threshold Row
        row2 = QHBoxLayout()
        
        row2.addWidget(QLabel("Threshold:"))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.0, 1.0)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(0.01)
        row2.addWidget(self.thresh_spin)
        param_layout.addLayout(row2)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Run Button
        self.run_btn = QPushButton("RUN VOCODER")
        self.run_btn.setMinimumHeight(50)
        self.run_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.run_btn)
        
        # Result Play Button
        self.btn_play_result = QPushButton("▶ Play Result")
        self.btn_play_result.setEnabled(False) # Initially disabled
        self.btn_play_result.clicked.connect(self.play_result)
        layout.addWidget(self.btn_play_result)
        
        # Log Area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)
        
    def create_file_row(self, parent_layout, label_text, btn_text, btn_callback, default_text=""):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(80)
        row.addWidget(lbl)
        
        edit = QLineEdit(default_text)
        row.addWidget(edit)
        
        btn = QPushButton(btn_text)
        btn.clicked.connect(btn_callback)
        row.addWidget(btn)
        
        parent_layout.addLayout(row)
        return edit

    def browse_vocal(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Vocal", "", "Audio Files (*.wav *.mp3)")
        if path: self.vocal_edit.setText(path)

    def browse_noise(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Noise Sample", "", "Audio Files (*.wav *.mp3)")
        if path: self.noise_edit.setText(path)

    def browse_sf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SoundFont", "", "SoundFonts (*.sf2 *.sfz)")
        if path: self.sf_edit.setText(path)

    def browse_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Output", "vocoder_output.wav", "WAV Files (*.wav)")
        if path: self.out_edit.setText(path)

    def log(self, text):
        self.log_area.append(text)

    def list_presets(self):
        sf_path = self.sf_edit.text()
        if not os.path.exists(sf_path):
            self.log("Error: SoundFont file not found.")
            return
            
        self.log(f"Scanning {os.path.basename(sf_path)}...")
        self.scan_thread = ScanThread(sf_path)
        self.scan_thread.result_signal.connect(self.log)
        self.scan_thread.start()



    def start_processing(self):
        params = {
            'vocal_path': self.vocal_edit.text(),
            'noise_path': self.noise_edit.text(),
            'sf_path': self.sf_edit.text(),
            'output_path': self.out_edit.text(),
            'bank': self.bank_spin.value(),
            'preset': self.preset_spin.value(),
            'threshold': self.thresh_spin.value(),
        }
        
        if not os.path.exists(params['vocal_path']) or not os.path.exists(params['sf_path']):
            QMessageBox.warning(self, "Error", "Please check input file paths.")
            return
            
        self.run_btn.setEnabled(False)
        self.btn_play_result.setEnabled(False) # Disable play result during run
        self.log_area.clear()
        
        self.worker = WorkerThread(params)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, success, message):
        self.run_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", message)
            self.log("Done.")
            self.btn_play_result.setEnabled(True)
            self.btn_play_result.setText("▶ Play Result")
        else:
            QMessageBox.critical(self, "Error", message)
            self.log(f"Failed: {message}")

    def on_vocal_changed(self):
        # Grey out result play button when vocal changes
        self.btn_play_result.setEnabled(False)
        self.btn_play_result.setText("▶ Play Result (Process first)")

    def play_audio(self, path):
        if not path or not os.path.exists(path):
            self.log(f"File not found: {path}")
            return
        
        url = QUrl.fromLocalFile(os.path.abspath(path))
        self.player.setSource(url)
        self.audio_output.setVolume(1.0)
        self.player.play()
        self.log(f"Playing: {os.path.basename(path)}")

    def play_vocal(self):
        self.play_audio(self.vocal_edit.text())

    def play_noise(self):
        self.play_audio(self.noise_edit.text())

    def play_result(self):
        self.play_audio(self.out_edit.text())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
