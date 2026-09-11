# Lao Air-Writing Using Deep Learning

## About The Project

This project is a Bachelor's thesis in Computer Science that implements a Lao character recognition system using air hand writing gestures. The system uses Computer Vision for hand tracking and gesture recognition and Deep Learning (CNN) for character recognition.

## System Requirements

**Operating System**: Linux (64-bit) - This project is specifically designed for Linux environments and requires Linux-specific dependencies.

### Key Features

- Real-time hand gesture tracking for air writing
- Lao character recognition using Convolutional Neural Networks (CNN)
- Support for both Lao vowels and consonants
- User-friendly GUI interface built with Tkinter

### Demo

![Lao Air-Writing Demo](src/assets/application_demo.gif)

## Project Structure

```md
/
├── datasets/              # Training and testing datasets (download separately)
├── model/                 # Trained models and related files
├── src/
│   ├── assets/            # Static assets (fonts, demo files, MediaPipe models)
│   ├── augment_image/     # Data augmentation utilities
│   ├── collect_data/      # Data collection interface
│   ├── lao_air_writting/  # Core application modules
│   └── utils/             # Utility modules and helper functions
```

## Installation

### Prerequisites

- uv

### Setup

1. Install uv if you haven't already:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Clone the repository:

    ```bash
    git clone https://github.com/sila1404/lao_air_writting.git
    cd lao_air_writting
    ```

3. Install dependencies using uv:

    ```bash
    uv sync --extra local
    ```

    `--extra local` adds mediapipe/seaborn/scikit-learn/albumentations, needed for data collection, augmentation, training, and evaluation. If you only want to run the API server, use `--extra api` instead (and `--extra postgres` if using Postgres for feedback storage) — it skips the desktop/training-only dependencies.

4. Download the dataset (see [Dataset](#dataset) section below)

All required dependencies are managed in pyproject.toml:

```toml
dependencies = [
"certifi",
"tensorflow>=2.19.0,<3",
"python-dotenv>=1.1.0,<2",
"torch>=2.7.1,<3",
"transformers>=4.52.4,<5",
"opencv-python-headless>=4.11.0,<5",
"numpy<2",
"pillow>=11.1.0,<12",
]
```

## Dataset

The dataset for training and testing the model can be downloaded from the following sources:

- **Hugging Face**: <https://huggingface.co/datasets/silamany/lao-character-images>
- **Kaggle**: <https://www.kaggle.com/datasets/silamany/lao-characters>

After downloading, extract the dataset into the `datasets/` folder in the project root directory.

## Usage

The project includes seven main commands for different stages of the process:

### Data Collection and Augmentation

- Collect Data

    ```bash
    uv run python src/collect_data/main.py
    ```

  - Launches the data collection interface
  - Use hand gestures to write Lao characters
  - Characters are saved in respective vowel/consonant folders

- Augment Data

    ```bash
    uv run python src/augment_image/main.py
    ```

  - Performs data augmentation on collected images
  - Increases dataset size through various transformations
  - Helps improve model robustness

### Model Training

- Split Dataset

    ```bash
    uv run python src/augment_image/split_data.py
    ```

  - Splits the collected data into training and testing sets
  - Prepares data for model training

- Train Model

    ```bash
    uv run python src/lao_air_writting/train_model.py
    ```

  - Initiates the CNN model training process
  - Uses the prepared training dataset
  - Saves the trained model

- Evaluate Model

    ```bash
    uv run python src/lao_air_writting/evaluate_model.py
    ```

  - Evaluates the trained model's performance
  - Generates performance metrics and reports

- Test Model

    ```bash
    uv run python src/lao_air_writting/test_app.py
    ```

  - Launches the main application interface
  - Allows real-time character writing and recognition

### API Server

- Start API Server

    ```bash
    uv run --extra api uvicorn lao_air_writting.api:app
    ```

  - Launches the API server for Lao character recognition
  - Provides endpoints for text recognition
  - Server runs on localhost (default port: 8000)

## Troubleshooting

If you encounter the following error:

_ModuleNotFoundError: No module named 'certifi'_

You can resolve it by removing the virtual environment:

```bash
rm -rf .venv
```

Then, reinstall the dependencies:

```bash
uv sync --extra local
```

## How It Works

- **Hand Tracking**: Uses MediaPipe for real-time hand landmark detection
- **Character Drawing**: Tracks index finger movement to create character drawings
- **Recognition**: Processes drawings through a trained CNN model

## Model Architecture

The character recognition model uses a Convolutional Neural Network (CNN) architecture:

- Input layer for processing character images
- Multiple convolutional and pooling layers
- Dense layers for classification
- Output layer for Lao character recognition

## Authors

Silamany HOMPHASATHANE & Phongsavanh SENGOKPADITH
Computer Science Department  
Faculty of Natural Sciences  
National University of Laos

## Acknowledgements

- Academic advisors

  - Somsack INTHASONE, Ph.D.
  - Ms. Sommany LOUSAVONG

- Data Collection Volunteers  
    We extend our sincere gratitude to all volunteers who contributed their time and effort in providing handwriting samples for our dataset:

  - Students from the Computer Programming, Soutsaka Institute of Technology
  - Members of the Computer Science Department

    Their contributions were essential in creating a diverse and comprehensive dataset for training our model.
