
# Spectrogram Generator

This project contains a Python script to convert `.wav` audio files into spectrogram images in `.png` format. It uses the `librosa` library for audio processing and `matplotlib` for generating and saving the spectrogram images.

## Requirements

- Python 3.x
- NumPy
- librosa
- matplotlib

## Installation

1. **Clone the Repository**

   ```
   git clone https://github.com/sarthakmishra459/audio-classifier.git
   cd audio-classifier
   ```

2. **Install the Required Packages**

   You can install the required packages using `pip`:

   ```
   pip install numpy librosa matplotlib
   ```
3. **Download Dataset**

   You can download the dataset from  `kaggle`:

   ```
   kaggle datasets download -d sarthak7654654/sounds-dataset
   ```

## Usage

### Function: `create_spectrogram`

This function takes an audio file and generates a spectrogram image.

#### Parameters:

- `audio_file` (str): Path to the input `.wav` file.
- `image_file` (str): Path to the output `.png` file where the spectrogram image will be saved.

### Function: `create_pngs_from_wavs`

This function processes all `.wav` files in the specified input directory and generates corresponding spectrogram images in the specified output directory.

#### Parameters:

- `input_path` (str): Path to the directory containing `.wav` files.
- `output_path` (str): Path to the directory where `.png` spectrogram images will be saved.

### Example

1. **Create a Spectrogram for a Single File**

   ```python
   import numpy as np
   import librosa.display
   import matplotlib.pyplot as plt
   %matplotlib inline

   def create_spectrogram(audio_file, image_file):
       fig = plt.figure()
       ax = fig.add_subplot(1, 1, 1)
       fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

       y, sr = librosa.load(audio_file)
       ms = librosa.feature.melspectrogram(y=y, sr=sr)
       log_ms = librosa.power_to_db(ms, ref=np.max)
       librosa.display.specshow(log_ms, sr=sr)

       fig.savefig(image_file)
       plt.close(fig)

   create_spectrogram('input.wav', 'output.png')
   ```

2. **Create Spectrograms for All Files in a Directory**

   ```python
   import os

   def create_spectrogram(audio_file, image_file):
       fig = plt.figure()
       ax = fig.add_subplot(1, 1, 1)
       fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

       y, sr = librosa.load(audio_file)
       ms = librosa.feature.melspectrogram(y=y, sr=sr)
       log_ms = librosa.power_to_db(ms, ref=np.max)
       librosa.display.specshow(log_ms, sr=sr)

       fig.savefig(image_file)
       plt.close(fig)

   def create_pngs_from_wavs(input_path, output_path):
       if not os.path.exists(output_path):
           os.makedirs(output_path)

       dir = os.listdir(input_path)

       for i, file in enumerate(dir):
           input_file = os.path.join(input_path, file)
           output_file = os.path.join(output_path, file.replace('.wav', '.png'))
           create_spectrogram(input_file, output_file)

   create_pngs_from_wavs('input_directory', 'output_directory')
   ```
