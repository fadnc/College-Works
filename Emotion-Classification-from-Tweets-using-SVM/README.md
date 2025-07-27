# SVM was used for the task of emotion classification.
For this project, we employed SVM as part of a pipeline that classifies the emotion in tweets. The process consists of cleaning data, using TF-IDF to extract top features, training all possible SVM kernels, evaluating and saving away the top model.

## Features
-Removes URLs, mentions, hashtags, punctuation, numbers and stopwords from raw tweet text.

-Applies the process of stemming to bring all words to their root.

-Transforms emotion labels into numbers.

-Changes text into a list of TF-IDF feature vectors.

-Builds and examines SVM classifiers for four types of kernels (linear, poly, rbf, sigmoid).

-Presents both classification scores and heat maps showing the confusion matrix.

-Keeps the best trained model, TF-IDF vectorizer and label encoder for next use in inferenc

## Installation
It is important to have Python 3.7 or higher installed before continuing. After that, install the necessary libraries.

You can install pandas, numpy, scikit-learn, matplotlib, seaborn, nltk and joblib by typing 
 pip install <name> for each.
 
## Setup
Download the NLTK data only one time before starting the code.

python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('punkt')
Usage
Upload your labeled CSV to the working directory (file name: emotions.csv)

In the script, be sure to update the place where the dataset is located.

You can read the file with the code: df = pd.read_csv('/path/to/emotions.csv')
Open and run the Python script in your terminal.

Clean all your collected tweets before working with them.

Encode labels

Devide the data into groups for training and for testing.

Turn your text into a vector using TF-IDF.

Test SVM models by using various kernels.

Draw confusion matrices and show how they differ in performance.

Save your best model along with all the things that help it.

Files Saved
The contacts.svm model is the best performing SVM classifier.

tfidf_vectorizer.pkl: This file is used to make TF-IDF vectors for feature extraction

This file contains a label encoder that lets you understand labeled predictions.

Adding to the Pipeline
To gather live tweets, either use the snscrape package or the Twitter API.

KMeans is a type of unsupervised clustering, so use it on the TF-IDF features to explore your data set.

Your application should be set up to warn users (using emails, for example) once emotions are spotted during interactions.

You should use grid or random search to find the best settings for SVM parameters.

