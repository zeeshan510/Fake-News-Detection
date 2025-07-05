📰 Fake News Detector using Machine Learning
A Streamlit-powered web application to classify news as Fake, Real, or Humor.
Table of Contents
Introduction

Features

Demo

How to Run Locally

Dataset

Model Details

Future Enhancements

Contributing

License

Contact

Introduction
In the age of information, distinguishing between genuine news and misinformation is crucial. This project presents a simple yet effective Fake News Detector built using machine learning techniques and deployed as an interactive web application with Streamlit. It aims to help users quickly classify news content into three categories: Fake, Real, or Humor.

Features
Interactive Web UI: Powered by Streamlit for a user-friendly experience.

Real-time Classification: Paste any news content and get an instant prediction.

Machine Learning Pipeline: Utilizes TfidfVectorizer for text feature extraction and LogisticRegression for classification.

Robust Training: Trained on a combined dataset from MediaEval 2015 to enhance accuracy.

Handles Data Imbalance: Employs class_weight='balanced' in the model to improve performance across different news categories.

Demo
Here's a quick look at the application in action, demonstrating both real and fake news detection:

1. Detecting Real News:
(Place your screenshot of the app classifying real news as real_news_screenshot.png inside the assets folder.)

2. Detecting Fake News:
(Place your screenshot of the app classifying fake news as fake_news_screenshot.png inside the assets folder.)

How to Run Locally
To get a local copy up and running, follow these simple steps.

Prerequisites
Python 3.8+

pip (Python package installer)

Installation
Clone the repository:

git clone https://github.com/your-username/Fake-News-Detection.git

(Replace your-username with your actual GitHub username, which is zeeshan510 in your case)

Navigate to the project directory:

cd Fake-News-Detection

Create a virtual environment (recommended):

python -m venv venv

On Windows:

.\venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Ensure data files are in place:
Make sure the mediaeval-2015-trainingset.csv and mediaeval-2015-testset.csv files are located inside the data/ subdirectory within your Fake-News-Detection project folder.

Running the Application
Once dependencies are installed and data files are in place, run the Streamlit app:

streamlit run app.py

This will open the application in your default web browser.

Dataset
The model is trained using a combined dataset from the MediaEval 2015 Fake News Detection Challenge. Specifically, the mediaeval-2015-trainingset.csv and mediaeval-2015-testset.csv files are concatenated to provide a comprehensive training base for the machine learning model. The dataset contains tweets labeled as 'fake', 'real', or 'humor'.

Model Details
The core of the detection system is a machine learning pipeline:

Text Vectorization: TfidfVectorizer is used to convert raw text data (tweet content) into numerical feature vectors. It weighs words based on their frequency in a document relative to their frequency in the entire corpus, highlighting important terms.

Classification: LogisticRegression is employed as the classification algorithm. It's a robust and interpretable model suitable for binary and multi-class classification tasks.

Class Weight Balancing: To address potential imbalances in the dataset (where one class might have significantly more samples than others), class_weight='balanced' is used. This automatically adjusts weights inversely proportional to class frequencies, helping the model learn more effectively from underrepresented classes.

Future Enhancements
Advanced Models: Experiment with more sophisticated NLP models (e.g., BERT, RoBERTa) for potentially higher accuracy.

Expanded Dataset: Integrate more diverse and larger datasets to improve generalization.

Feature Engineering: Explore additional features beyond TF-IDF, such as sentiment scores or linguistic features.

Deployment: Deploy the application to a cloud platform (e.g., Streamlit Community Cloud, AWS, GCP) for public access.

User Feedback: Implement a mechanism for users to provide feedback on predictions to further refine the model.

Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
Your GitHub Profile: https://github.com/zeeshan510

Email: mohdzeeshan0626@gmail.com
