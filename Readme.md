# Lao Air-Writing and Text-to-Speech Using Deep Learning

## About The Project

This project is a Bachelor's thesis in Computer Science that implements a Lao character recognition system using air hand writing gestures and converts the recognized text to speech. The system uses Computer Vision for hand tracking and gesture recognition, Deep Learning (CNN) for character recognition, and integrates with a Text-to-Speech API for voice output.

### Key Features

-   Real-time hand gesture tracking for air writing
-   Lao character recognition using Convolutional Neural Networks (CNN)
-   Support for both Lao vowels and consonants
-   Text-to-Speech conversion through API integration
-   User-friendly GUI interface built with Tkinter

## Installation

### Prerequisites

-   pixi

### Setup

1. Install pixi if you haven't already:

    - Windows:

        ```bash
        powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
        ```

    - Linux & MacOS:
        ```bash
        curl -fsSL https://pixi.sh/install.sh | sh
        ```

2. Clone the repository:

    ```bash
    git clone https://github.com/sila1404/lao_air_writting.git
    cd lao_air_writting
    ```

3. Install dependencies using pixi:
    ```bash
    pixi install
    ```

All required dependencies are managed in pyproject.toml:

-   pypi dependency
    ```toml
    dependencies = [
    "certifi",
    "mediapipe>=0.10.14,<0.11",
    "tensorflow>=2.19.0,<3",
    "python-dotenv>=1.1.0,<2",
    "torch>=2.7.1,<3",
    "transformers>=4.52.4,<5",
    "accelerate>=1.7.0,<2",
    ]
    ```
-   conda dependency
    ```toml
    [tool.pixi.dependencies]
    opencv = ">=4.11.0,<5"
    numpy = "<2"
    pillow = ">=11.1.0,<12"
    seaborn = ">=0.13.2,<0.14"
    scikit-learn = ">=1.6.1,<2"
    albumentations = ">=2.0.5,<3"
    ```

## Usage

The project includes seven main commands for different stages of the process:

### Data Collection and Augmentation

-   Collect Data

    ```bash
    pixi run collect
    ```

    -   Launches the data collection interface
    -   Use hand gestures to write Lao characters
    -   Characters are saved in respective vowel/consonant folders

-   Augment Data
    ```bash
    pixi run augment
    ```
    -   Performs data augmentation on collected images
    -   Increases dataset size through various transformations
    -   Helps improve model robustness

### Model Training

-   Split Dataset

    ```bash
    pixi run split
    ```

    -   Splits the collected data into training and testing sets
    -   Prepares data for model training

-   Train Model

    ```bash
    pixi run train
    ```

    -   Initiates the CNN model training process
    -   Uses the prepared training dataset
    -   Saves the trained model

-   Evaluate Model

    ```bash
    pixi run eval
    ```

    -   Evaluates the trained model's performance
    -   Generates performance metrics and reports

-   Test Model
    ```bash
    pixi run test
    ```
    -   Launches the main application interface
    -   Allows real-time character writing and recognition
    -   Includes text-to-speech functionality

### API Server

-   Start API Server
    ```bash
    pixi run api
    ```
    -   Launches the API server for Lao character recognition
    -   Provides endpoints for text recognition and text-to-speech conversion
    -   Server runs on localhost (default port: 8000)

## Troubleshooting

If you encounter the following error:

_ModuleNotFoundError: No module named 'certifi'_

You can resolve it by running the following command:

```bash
pixi clean
```

Then, reinstall the dependencies:

```bash
pixi install
```

## How It Works

-   **Hand Tracking**: Uses MediaPipe for real-time hand landmark detection
-   **Character Drawing**: Tracks index finger movement to create character drawings
-   **Recognition**: Processes drawings through a trained CNN model
-   **Text-to-Speech**: Converts recognized characters to speech using API

## Model Architecture

The character recognition model uses a Convolutional Neural Network (CNN) architecture:

-   Input layer for processing character images
-   Multiple convolutional and pooling layers
-   Dense layers for classification
-   Output layer for Lao character recognition

## Future Improvements

-   Enhance recognition accuracy
-   Add support for continuous writing
-   Implement local text-to-speech processing
-   Improve GUI interface

## Authors

Silamany HOMPHASATHANE & Phongsavanh SENGOKPADITH   
Computer Science Department  
Faculty of Natural Sciences  
National University of Laos

## Acknowledgements

-   Academic advisors

    -   Somsack INTHASONE, Ph.D.
    -   Ms. Sommany LOUSAVONG

-   Data Collection Volunteers  
    We extend our sincere gratitude to all volunteers who contributed their time and effort in providing handwriting samples for our dataset:

    -   Students from the Computer Programming, Soutsaka Institute of Technology
    -   Members of the Computer Science Department

    Their contributions were essential in creating a diverse and comprehensive dataset for training our model.
