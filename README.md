# Audio Classification using Neural Networks

## Overview
This project classifies audio signals by converting them into spectrogram images using the `librosa` library and training a neural network on these images. The approach enables high-accuracy classification by leveraging deep learning techniques for image recognition.

## Features
- Converts audio signals into spectrogram images.
- Uses `librosa` for feature extraction.
- Applies a Convolutional Neural Network (CNN) for classification.
- Supports multiple audio categories, such as background noise and specific sound events.

## Installation
### Prerequisites
Ensure you have Python installed along with the required dependencies:

```bash
pip install numpy librosa matplotlib tensorflow keras
```

## Usage
1. Convert `.wav` files into spectrogram images:
   ```python
   from spectrogram_generator import create_pngs_from_wavs
   create_pngs_from_wavs('path_to_audio', 'path_to_spectrograms')
   ```
2. Train the CNN model:
   ```python
   from model import train_model
   train_model('path_to_spectrograms')
   ```
3. Predict on new audio data:
   ```python
   from predict import classify_audio
   result = classify_audio('new_audio.wav')
   print("Predicted Class:", result)
   ```

## Model Architecture
- **Feature Extraction:** Convert audio to Mel spectrograms.
- **Neural Network:** A CNN model trained on spectrogram images.
- **Classification:** Predicts the category of the given audio signal.

## Results
- Achieved high accuracy with CNN on spectrogram images.
- Improved classification performance by leveraging audio pre-processing techniques.

## Future Improvements
- Integrate transformer-based audio models.
- Expand dataset for better generalization.
- Optimize training using transfer learning techniques.

