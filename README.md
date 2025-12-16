# SymbiPredict - Predictive Healthcare Model

![SymbiPredict](https://img.shields.io/badge/SymbiPredict-v1.0-blue)

## 🏥 Overview

SymbiPredict is an advanced predictive healthcare system that leverages machine learning and artificial intelligence to revolutionize disease diagnosis and outbreak prediction. This innovative platform combines symptom-based disease classification with time-series forecasting to provide healthcare professionals with valuable insights for early detection and prevention strategies.

Built with Python and powered by cutting-edge ML algorithms, SymbiPredict enables accurate disease prediction from patient symptoms while forecasting potential disease outbreaks up to 7 days in advance. The system is designed for healthcare providers, epidemiologists, and public health officials seeking data-driven decision-making tools.

## ✨ Key Features

### 🩺 Symptom-Based Disease Classification

- Interactive symptom checker with 120+ medical conditions
- Real-time disease prediction based on user-input symptoms
- Comprehensive patient data recording (name, age, gender, location, symptoms)

### 📊 Outbreak Detection & Forecasting

- Automated outbreak detection using statistical thresholds
- Time-series forecasting with Prophet algorithm for 7-day projections
- Visual trend analysis with historical and predictive data
- Location-based outbreak mapping for geographic insights

### 🗺️ Geographic Visualization

- Interactive heat maps showing disease hotspots
- Location-based case aggregation and analysis
- Real-time monitoring of disease distribution patterns

### 🔄 Data-Driven Insights

- Continuous learning from new patient records
- Dynamic updating of outbreak detection parameters
- Integration of historical and real-time data for improved accuracy

## 🛠️ Technical Architecture

### Machine Learning Models

#### Disease Classification Model

- **Algorithm**: LinearSVC (Linear Support Vector Classification)
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Training Data**: Comprehensive dataset mapping 100+ symptoms to various diseases
- **Preprocessing**: Text-based symptom aggregation and normalization
- **Performance**: Optimized for high precision and recall in medical diagnostics

#### Time-Series Forecasting Model

- **Algorithm**: Facebook Prophet
- **Parameters**: Yearly and weekly seasonality, changepoint detection, uncertainty intervals
- **Prediction Horizon**: 7-day future forecasts
- **Input**: Historical disease case counts aggregated by date and disease type
- **Features**: Automatic detection of seasonal patterns and trend shifts

### Core Components

1. **`ui.py`** - Streamlit-based web interface for user interaction
2. **`disease_detect_train.py`** - Model training pipeline using LinearSVC
3. **`disease_forecast.py`** - Prophet-based time-series forecasting engine
4. **`disease_outbreak.py`** - Outbreak detection algorithms with statistical thresholds
5. **`disease_preprocess.py`** - Data preprocessing and aggregation utilities

### Data Pipeline

- **Input**: Symptom selection, patient demographics, geographic location
- **Processing**: Symptom-to-disease classification, case aggregation by date/location
- **Analysis**: Outbreak detection using 2x rolling average thresholds
- **Forecasting**: 7-day projections using time-series modeling
- **Output**: Visualizations, outbreak alerts, geographic heat maps

## 📋 Prerequisites

- Python 3.8 or higher
- Stable internet connection for package installation

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/rchrdwllm/predictive_healthcare_model.git
cd predictive_healthcare_model
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Dataset Availability

Ensure the following files exist in the appropriate directories:

- `dataset/dataset.csv` (training data)
- `dataset/sample_user_data.csv` (user data storage)
- Models directory with pre-trained models

If models are missing, retrain using:

```bash
python disease_detect_train.py
```

## 🎯 Usage

### Running the Application

Launch the SymbiPredict dashboard:

```bash
streamlit run ui.py
```

The application opens in your default browser at `http://localhost:8501` with three main tabs:

#### 1. 🩺 Disease Prediction Tab

- Enter patient information (name, age, gender, location)
- Select relevant symptoms from the extensive symptom list
- View predicted disease outcome with confidence indicators
- Record patient data for future analysis

#### 2. 📊 Outbreak Forecasting Tab

- Visualize historical disease trends
- Access 7-day outbreak forecasts
- Monitor outbreak detection alerts
- Explore geographic disease distribution

#### 3. 📝 Patient Log Tab

- Browse historical patient records
- Track diagnostic patterns over time
- Analyze location-based trends

### Training Custom Models

To retrain the disease classification model with updated data:

```bash
python disease_detect_train.py
```

This updates the `disease_prediction_model.joblib` with the latest training dataset.

## 🧠 Algorithms & Technologies

### Machine Learning Stack

- **Scikit-learn**: LinearSVC for disease classification
- **Prophet**: Time-series forecasting for outbreak prediction
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Static visualizations

### Web Interface

- **Streamlit**: Interactive web application framework
- **PyDeck**: Geographic visualization and heat mapping
- **Altair**: Statistical visualizations

### Outbreak Detection Methodology

- Statistical threshold: Outbreak declared when cases exceed 2x rolling average
- Rolling window: 3-day moving average for baseline comparison
- Continuous monitoring: Real-time calculation of outbreak status

## 📊 Dataset Information

### Training Data

- Source: `dataset/dataset.csv`
- Contains mappings between symptoms and diseases
- Features: 17 symptom columns per patient record
- Labels: Disease classifications across multiple medical categories

### User Data Storage

- Location: `dataset/sample_user_data.csv`
- Format: Patient demographics, symptoms, predictions, and timestamps
- Purpose: Training enhancement and trend analysis

## 🎨 User Interface Components

### Symptom Checker Interface

- Intuitive multi-select dropdown for symptom selection
- Real-time prediction display
- Patient data validation and recording
- Disease-specific outbreak visualization

### Geographic Heat Map

- Location-based disease intensity visualization
- Interactive map with hover details
- Geographic clustering analysis
- Real-time updating with new data

### Forecasting Charts

- Historical vs forecasted case comparisons
- Outbreak indicator overlays
- Disease-specific trend analysis
- Confidence interval displays

## 🔐 Data Privacy & Security

SymbiPredict handles sensitive health information with care:

- Local data storage ensures privacy
- No external data transmission by default
- Secure data handling protocols
- HIPAA-compliant data structure considerations

## 🧪 Model Performance

### Disease Classification

- Training methodology: TF-IDF vectorization + LinearSVC
- Feature engineering: Aggregated symptom text representation
- Validation: Hold-out test sets with performance metrics
- Accuracy: Optimized for high clinical relevance

### Time-Series Forecasting

- Model evaluation: Cross-validation on historical data
- Seasonal patterns: Captures weekly and annual trends
- Uncertainty quantification: Probabilistic forecasts with confidence intervals
- Sensitivity analysis: Robust performance across different diseases

## 🤝 Contributing

Contributions to SymbiPredict are welcome! Potential areas for improvement include:

- Enhanced feature extraction methods for symptom classification
- Advanced ensemble techniques for improved prediction accuracy
- Additional visualization options and dashboard improvements
- Integration with electronic health record systems
- Mobile application development
- Multilingual support for global healthcare applications

### Development Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes following the existing code style
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request with clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions, issues, or feature requests:

- Open an issue in the repository for bug reports
- Contact the development team for implementation support
- Check the project documentation for troubleshooting guides

## 🚀 Future Enhancements

- Integration with clinical decision support systems
- Advanced natural language processing for free-text symptom input
- Deep learning approaches for improved classification accuracy
- Real-time integration with health department reporting systems
- Multi-city expansion capabilities
- API development for third-party integrations
