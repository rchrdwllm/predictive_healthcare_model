# SymbiPredict - Predictive Healthcare Model

![SymbiPredict](https://img.shields.io/badge/SymbiPredict-v1.0-blue)

## Overview

SymbiPredict is a comprehensive predictive healthcare system designed to identify diseases based on symptoms, track disease outbreaks, and forecast future health trends. This intelligent system combines machine learning with data visualization to provide valuable insights for healthcare professionals and organizations.

## Features

- **Symptom-Based Disease Prediction**: Interactive GUI interface that allows users to input symptoms and predicts potential diseases
- **Disease Outbreak Detection**: Analyzes historical data to identify current disease outbreaks
- **Future Outbreak Prediction**: Forecasts potential disease outbreaks up to 7 days in advance
- **Visualized Health Trends**: Graphical representation of historical and forecasted disease cases
- **Data-Driven Decision Making**: Leverages machine learning to assist healthcare planning and resource allocation

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Dependencies

- scikit-learn
- pandas
- numpy
- matplotlib
- prophet
- tkinter
- joblib

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/rchrdwllm/predictive_healthcare_model.git
   cd predictive-healthcare-model
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Application

Execute the main application which includes symptom checking, outbreak detection, and forecasting:

```bash
python app.py
```

This will:

1. Launch the symptom checker GUI
2. Add your input to the dataset
3. Calculate disease counts
4. Detect current outbreaks
5. Forecast future outbreaks
6. Display visualization of the data

### Train the Model

If you need to retrain the disease prediction model with updated data:

```bash
python disease_detect_train.py
```

This will train a Random Forest model using the dataset in `dataset/symbipredict_2022.csv` and save the model and label encoder to the `models` and `encoders` directories.

## Project Structure

```
predictive_healthcare_model/
├── app.py                     # Main application entry point
├── disease_detect.py          # Disease prediction GUI and logic
├── disease_detect_train.py    # Training script for the disease prediction model
├── disease_forecast.py        # Time series forecasting for disease trends
├── disease_outbreak.py        # Outbreak detection algorithms
├── disease_preprocess.py      # Data preprocessing utilities
├── flow.txt                   # Process flow documentation
├── requirements.txt           # Project dependencies
├── dataset/                   # Data directory
│   ├── sample_user_data.csv   # User symptom entries and predictions
│   └── symbipredict_2022.csv  # Training dataset for disease prediction
├── encoders/                  # Directory for saved encoders
│   └── label_encoder.pkl      # Label encoder for disease classes
└── models/                    # Directory for saved models
    └── disease_prediction_model.pkl  # Trained Random Forest model
```

## How It Works

### Application Flow

1. **User Input**: The user checks symptoms through a GUI interface
2. **Disease Prediction**: The system predicts diseases based on the symptoms
3. **Data Recording**: The entry is recorded with symptoms, date, and predicted disease
4. **Outbreak Analysis**: Current outbreaks are detected based on historical data
5. **Forecasting**: Future outbreaks are predicted for the next 7 days
6. **Visualization**: Results are displayed as interactive charts

### System Components

#### 1. Disease Prediction

The system uses a Random Forest classifier to predict diseases based on user-inputted symptoms. The model is trained on a comprehensive dataset that maps symptoms to diagnoses.

#### 2. Data Collection

User interactions with the system are recorded in `sample_user_data.csv`, creating a growing database of symptoms, dates, and predicted diseases. This data serves as the foundation for outbreak detection and forecasting.

#### 3. Outbreak Detection

The system defines an outbreak when the number of cases for a specific disease exceeds twice the rolling average of previous days. This helps identify unusual spikes in disease prevalence.

#### 4. Forecasting

Using Facebook's Prophet library, the system analyzes historical disease data to predict future cases up to 7 days in advance. This forecasting capability includes:

- Daily, weekly, and yearly seasonality patterns
- Change point detection for trend shifts
- Uncertainty intervals

#### 5. Visualization

The system generates plots for each disease showing:

- Historical case counts
- Forecasted future cases
- Highlighted outbreak points
- Trend lines

## Model Training

The disease prediction model uses scikit-learn's RandomForestClassifier with hyperparameter tuning through GridSearchCV. The training process includes:

- Feature selection from the symptom dataset
- Label encoding for disease categories
- Train/test splitting (80/20)
- Cross-validation for performance assessment
- Hyperparameter optimization
- Model evaluation using classification reports

The model is trained to recognize patterns in over 100 different symptoms and associate them with specific diseases. This allows for accurate prediction even with partial symptom information.

## Contributing

Contributions to SymbiPredict are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


