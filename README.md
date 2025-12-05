# vocoder-soundfont-stft
A simple vocoder project that utilizes Short Time Fourier Transform for noise reduction (Spectral Subtraction) and Voice Modulation. Uses a human vocal .wav audio, a noise sample .wav audio, and an sf2 soundfont.

## Features
1. Add a `wav` audio file of your own voice. 
2. Soundfont-based modulation: Load an `sf2` soundfont (preferably with dedicated audio sample for each note, like Saint James Orchestra Soundfonts' Violin Detache in the repository linked below) to modulate your voice.
3. Sample-based noise reduction: Load a noise sample `wav` file to activate the Spectral Subtraction-based noise reduction. Helps if your voice audio file is noisy.
4. Play the samples to review.
5. Listen to the vocoder result with a simple click, without many hassles.

## Requirements 
- scipy
- numpy
- fluidsynth
- PyQt6

## Instructions
1. Clone this repository
2. Install [fluidsynth](https://www.fluidsynth.org/)
3. (OPTIONAL BUT RECOMMENDED) Create a virtual environtment: `python -m venv [env-name]`
4. Install the required python packages: `pip install numpy scipy pyfluidsynth PyQt6`
5. Run the file `python talking_instrument_scipy_stft_gui.py`

## Useful resources
[Saint James Orchestra Soundfonts](https://github.com/open-soundfonts/Saint_James_Orchestra_soundfonts) (Contains rich samples for many pitches)
[FluidR3 Soundfonts](https://github.com/Jacalz/fluid-soundfont) (Contains only one sample of one tone for each instrument)
