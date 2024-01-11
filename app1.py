from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
count_vector = None
rf_model = None

# Load the Amazon data and preprocess it (as shown in your original code)
path = r"C:/Users/Anjana/team 13(Sentiment Analysis on user feedback data)/Amazon_Unlocked_Mobile.csv"
amazon_data = pd.read_csv(path)
amazon_data = amazon_data.dropna(axis=0)
amazon_data = amazon_data[["Reviews", "Rating"]]
amazon_data_pos = amazon_data[amazon_data["Rating"].isin([4, 5])]
amazon_data_neg = amazon_data[amazon_data["Rating"].isin([1, 2])]
amazon_data_filtered = pd.concat([amazon_data_pos[:20000], amazon_data_neg[:20000]])
amazon_data_filtered["Sentiment"] = amazon_data_filtered["Rating"].apply(lambda x: 1 if x > 3 else 0)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    amazon_data_filtered["Reviews"],
    amazon_data_filtered["Sentiment"],
    test_size=0.2,
    random_state=42
)

# Preprocess text data using CountVectorizer
count_vector = CountVectorizer(stop_words="english")
count_vector.fit(X_train)
X_train_transformed = count_vector.fit_transform(X_train)
X_test_transformed = count_vector.transform(X_test)

def train_and_predict_svm(X_train, Y_train, X_test):
    svm_model = SVC()
    svm_model.fit(X_train, Y_train)
    predictions = svm_model.predict(X_test)
    return predictions
def train_and_predict_logistic_regression(X_train, Y_train, X_test):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, Y_train)
    predictions = lr_model.predict(X_test)
    return predictions
def train_and_predict_multinomial_nb(X_train, Y_train, X_test):
    nb_model = MultinomialNB()
    nb_model.fit(X_train, Y_train)
    predictions = nb_model.predict(X_test)
    return predictions

def train_and_predict_bernoulli_nb(X_train, Y_train, X_test):
    nb_model = BernoulliNB()
    nb_model.fit(X_train, Y_train)
    predictions = nb_model.predict(X_test)
    return predictions

def train_and_predict_knn(X_train, Y_train, X_test):
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(X_train, Y_train)
    predictions = knn_model.predict(X_test)
    return predictions

def train_and_predict_random_forest(X_train, Y_train, X_test):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, Y_train)
    predictions = rf_model.predict(X_test)
    return predictions

def train_and_predict_decision_tree(X_train, Y_train, X_test):
    tree_model = tree.DecisionTreeClassifier()
    tree_model.fit(X_train, Y_train)
    predictions = tree_model.predict(X_test)
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST','GET'])
def analyze():
    global count_vector, rf_model
    
    selected_model = request.form.get('model')
    selected_metric = request.form.get('metric')
    result = None

    predictions = []
    # Analyze the selected model with the selected metric
    if selected_model == 'SVM':
        predictions = train_and_predict_svm(X_train_transformed, y_train, X_test_transformed)
    elif selected_model == 'Logistic Regression':
        predictions = train_and_predict_logistic_regression(X_train_transformed, y_train, X_test_transformed)
    elif selected_model == 'Multinomial Naive Bayes':
        predictions = train_and_predict_multinomial_nb(X_train_transformed, y_train, X_test_transformed)
    elif selected_model == 'Bernoulli Naive Bayes':
        predictions = train_and_predict_bernoulli_nb(X_train_transformed, y_train, X_test_transformed)
    elif selected_model == 'k-NN':
        predictions = train_and_predict_knn(X_train_transformed, y_train, X_test_transformed)
    elif selected_model == 'Ensemble':
        predictions = train_and_predict_random_forest(X_train_transformed, y_train, X_test_transformed)
    elif selected_model == 'Decision Tree':
        predictions = train_and_predict_decision_tree(X_train_transformed, y_train, X_test_transformed)
   
    # Calculate the selected metric
    if selected_metric == 'Accuracy':
        result = accuracy_score(y_test, predictions)
    elif selected_metric == 'ROC Curve':
    
        fpr, tpr, _ = roc_curve(y_test, predictions)
        roc_auc = auc(fpr, tpr)

        # Plot ROC Curve 
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
        result = "ROC Curve"
    elif selected_metric == 'Precision,Recall and F-Measure':
        result = classification_report(y_test, predictions, target_names=["POSITIVE", "NEGATIVE"])

   
    if 'Content-Type' in request.headers and (request.headers['Content-Type'] == 'application/json' or request.headers['content-type'] == 'application/json'):
        return jsonify({'result': result})
    else:
        return render_template('analyze.html', result=result)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global count_vector, rf_model 

    if request.method == 'POST':
        review = request.form.get('review')
        if not review:
            return jsonify({'error': 'Please enter a review text.'})

        # Transform the user's input review
        new_review_transformed = count_vector.transform([review])

        # Predict the rating (0 for Bad, 1 for Good)
        rating_prediction = rf_model.predict(new_review_transformed)[0]

        # Convert the prediction to "Good" or "Bad"
        predicted_rating = "Good" if rating_prediction == 1 else "Bad"

        return render_template('predict.html', review=review, predicted_rating=predicted_rating)
    file_path = r"C:/Users/Anjana/team 13(Sentiment Analysis on user feedback data)/Amazon_Unlocked_Mobile.csv"
    # Data cleaning and preprocessing
    def load_data(file_path):
        amazon_data = pd.read_csv(file_path)
        amazon_data = amazon_data.dropna(axis=0)
        amazon_data = amazon_data[["Reviews", "Rating"]]
        amazon_data_pos = amazon_data[amazon_data["Rating"].isin([4, 5])]
        amazon_data_neg = amazon_data[amazon_data["Rating"].isin([1, 2])]
        amazon_data_filtered = pd.concat([amazon_data_pos[:20000], amazon_data_neg[:20000]])
        amazon_data_filtered["r"] = 1
        amazon_data_filtered["r"][amazon_data_filtered["Rating"].isin([1, 2])] = 0

        # Splitting Train and Test Data
        X_train_data, x_test_data, Y_train_data, y_test_data = train_test_split(
            amazon_data_filtered["Reviews"],
            amazon_data_filtered["r"],
            test_size=0.2
        )

        return X_train_data, x_test_data, Y_train_data, y_test_data

    # Load data
    X_train_data, x_test_data, Y_train_data, y_test_data = load_data(file_path)

    # Text Transformation using Count Vectorization Technique
    count_vector = CountVectorizer(stop_words="english")
    count_vector.fit(X_train_data)
    X_train_data_new = count_vector.transform(X_train_data)

    # Train the Random Forest Classifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_data_new, Y_train_data)

    return render_template('predict.html')  # You might want to render a form here

def compare_models():
    # List of model names and their corresponding instances
    models = [
        ('Logistic Regression', LogisticRegression()),  # Create an instance
        ('SVM', SVC()),
        ('Multinomial Naive Bayes', MultinomialNB()),
        ('Bernoulli Naive Bayes', BernoulliNB()),
        ('k-NN', KNeighborsClassifier()),
    ]

    # Create an empty dictionary to store comparison results
    comparison_results = {}

    # Iterate through each model
    for model_name, model_instance in models:
        # Calculate accuracy of the current model
        model_instance.fit(X_train_transformed, y_train)
        accuracy = accuracy_score(y_test, model_instance.predict(X_test_transformed))

        # Store the accuracy in the comparison_results dictionary
        comparison_results[model_name] = accuracy

    return comparison_results

@app.route('/compare')
def compare():
    # Perform comparison of models
    comparison_results = compare_models()
    return render_template('compare.html', comparison_results=comparison_results)

if __name__ == '__main__':
    app.run(debug=True)
