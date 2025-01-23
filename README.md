# Disease Prediction Using Machine Learning

A machine learning-based disease prediction system that uses symptoms to predict possible diseases. The model utilizes an ensemble approach combining Support Vector Classifier (SVC), Gaussian Naive Bayes, and Random Forest Classifier for accurate predictions.

# Features

### Disease Prediction
- Data preprocessing and cleaning
- Multiple machine learning models implementation
- Model evaluation using k-fold cross-validation
- Ensemble learning with voting classifier
- Feature importance analysis
- Symptom-based disease classification

### Treatment Plan Generation
- Integration with BioGPT language model
- Detailed treatment plan generation
- Backup treatment plan system
- Structured medical recommendations
- Comprehensive patient care guidelines

### System Integration
- Unified medical diagnosis system
- Combined disease prediction and treatment planning
- Interactive command-line interface
- Robust error handling and validation
- Extensible architecture for adding new diseases

### Technical Features
- Modular code structure
- Git LFS support for large model files
- Efficient symptom processing
- Scalable model deployment
- Comprehensive documentation

![display](assets/disease-prediction-web.gif)
![display](assets/disease-prediction-app.gif)

# Dataset
The dataset used in this project is available in the data directory and includes:

training.csv: Training data with symptoms and corresponding diseases.
testing.csv: Testing data for model evaluation.

132 different symptoms as features
Multiple disease categories as target variables
Balanced distribution of disease cases
Source: Kaggle Disease Prediction Dataset

# Model Performance

Cross-validation accuracy: 99%
Test set accuracy: 99%
Ensemble model combines predictions from:
    Support Vector Classifier
    Gaussian Naive Bayes
    Random Forest Classifier 

# License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

