# Fake News Classification Project

## Project Overview
This project aims to classify news articles as 'fake' or 'real' using various machine learning and deep learning techniques. The goal is to build and compare multiple models to achieve high accuracy in distinguishing between fake and real news.

## Project Structure
The project is organized into several steps:
1. **Data Preprocessing**: Cleaning and processing the text data.
2. **Feature Extraction**: Converting text data into numerical features using techniques like TF-IDF.
3. **Model Training**:
   - Classical machine learning models: Naive Bayes, SVM.
   - Deep learning models: RNN, LSTM.
   - Transfer learning: BERT.
4. **Model Evaluation**: Comparing model performance using metrics like accuracy, precision, recall, and F1-score.
5. **Deployment**: Deploying the best model using Flask to create a web application.

## Installation
To run this project, ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- NLTK
- PyTorch
- Transformers (Hugging Face)
- Flask

You can install the required packages using pip:
```bash
pip install numpy pandas scikit-learn nltk torch transformers flask
```

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/SandyHedia/fake_news_classification.git
   cd fake_news_classification
   ```

2. **Run the Jupyter Notebook**:
   Open the `fake_news_classification.ipynb` file in Jupyter Notebook or Jupyter Lab and run the cells to execute the project.

3. **Run the Flask App**:
   After training the models, you can run the Flask app to deploy the model.
   ```bash
   export FLASK_APP=app.py
   flask run
   ```
   This will start the web application, and you can interact with the model through the web interface.

## Models and Techniques
- **Classical Machine Learning Models**:
  - Naive Bayes
  - Support Vector Machine (SVM)
- **Deep Learning Models**:
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
- **Transfer Learning**:
  - BERT (Bidirectional Encoder Representations from Transformers)

## Results
- Achieved high accuracy with both classical and deep learning models.
- BERT model provided the best performance with an accuracy of 99.6%.

## Conclusion
This project demonstrates the effectiveness of various machine learning and deep learning techniques in classifying fake news. The use of transfer learning with BERT significantly improved the accuracy of the model.

## Future Work
- Explore other advanced models and techniques for further improvement.
- Implement additional feature engineering methods.
- Enhance the web application with more features and better user interface.

## Acknowledgements
- The dataset used in this project is publicly available and sourced from [Kaggle](https://www.kaggle.com/).
- The BERT model is provided by the Hugging Face library.
