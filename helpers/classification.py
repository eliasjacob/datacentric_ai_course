import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import ndarray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, hamming_loss,
                             jaccard_score, matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from xgboost import XGBClassifier
from typing import Tuple
import numpy as np
from numpy import ndarray
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf


def train_and_evaluate_classification_models(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[Tuple[str, str, np.ndarray]]]:
    """
    Train multiple classification models and evaluate their performance.

    Args:
        X (pd.DataFrame): The training data.
        y (pd.Series): The training labels.

    Returns:
        Tuple[pd.DataFrame, List[Tuple[str, str, np.ndarray]]]: A DataFrame with the performance metrics of each model and a list of classification reports.
    """
    random_state = 271828
    
    # Define the models to be trained
    models = [
        ('Calibrated-LSVC', CalibratedClassifierCV(LinearSVC(random_state=random_state, class_weight='balanced', dual='auto'))),
        ('Logistic Regression', LogisticRegression(random_state=random_state, n_jobs=-1, class_weight='balanced')),
        ('Random Forest', RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced')),
        ('XGBoost', XGBClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced', verbosity=0)),
        ('SGD', SGDClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced')),
        ('Naive Bayes', MultinomialNB()),
        # ('Linear SVC', LinearSVC(random_state=random_state, class_weight='balanced')),
        ('K-Nearest Neighbors', KNeighborsClassifier(n_jobs=-1)),
        ('Decision Tree', DecisionTreeClassifier(random_state=random_state, class_weight='balanced')),
        ('Extra Trees', ExtraTreesClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced'))
    ]
    
    performance_results = []
    classification_reports = []
    
    # StratifiedKFold cross-validator to ensure each fold has the same proportion of classes
    cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    for model_name, model in models:
        start_time = time.time()

        try:
            # Perform cross-validated predictions
            predicted_labels = cross_val_predict(estimator=model, X=X, y=y, cv=cross_validation, method="predict", n_jobs=2)
        except Exception as e:
            print(f'Error {model_name} - {e}')
            continue 

        # Calculate performance metrics
        f1 = f1_score(y, predicted_labels, average='micro')
        balanced_accuracy = balanced_accuracy_score(y, predicted_labels)
        accuracy = accuracy_score(y, predicted_labels)
        classification_report_str = classification_report(y, predicted_labels)
        matthews_corr_coeff = matthews_corrcoef(y, predicted_labels)
        confusion_mat = confusion_matrix(y, predicted_labels)
        
        # Append the classification report and confusion matrix to the list
        classification_reports.append((model_name, classification_report_str, confusion_mat))

        elapsed_time = time.time() - start_time
        
        # Append the performance metrics to the results list
        performance_results.append([
            model_name, f1, balanced_accuracy, accuracy, matthews_corr_coeff, elapsed_time, confusion_mat, classification_report_str
        ])

        # Print the performance metrics
        print(f'Model: {model_name} - F1: {f1:.4f} - Balanced Accuracy: {balanced_accuracy:.4f} - Accuracy: {accuracy:.4f} - Matthews Correlation Coefficient: {matthews_corr_coeff:.4f} - Elapsed time: {elapsed_time:.2f}s')
        print(classification_report_str)
        print(confusion_mat)
        print('*' * 20, '\n')

    # Create a DataFrame to store the performance results
    results_df = pd.DataFrame(performance_results, columns=[
        'Model', 'F1', 'Balanced Accuracy', 'Accuracy', 'Matthews Correlation Coefficient', 'Elapsed Time', 'Confusion Matrix', 'Classification Report'
    ])
    
    # Convert the confusion matrix to a string for better readability in the DataFrame
    results_df['Confusion Matrix'] = results_df['Confusion Matrix'].apply(lambda x: str(x))

    return results_df, classification_reports




def build_neural_network(input_dimension: int, hidden_layer1_size: int, hidden_layer2_size: int, 
                         hidden_layer3_size: int, output_size: int, dropout_rate: float, learning_rate: float) -> Model:
    """
    Build a neural network model with three hidden layers and dropout.

    Args:
        input_dimension (int): The size of the input layer.
        hidden_layer1_size (int): The size of the first hidden layer.
        hidden_layer2_size (int): The size of the second hidden layer.
        hidden_layer3_size (int): The size of the third hidden layer.
        output_size (int): The size of the output layer.
        dropout_rate (float): The dropout rate to be used after each hidden layer.
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        Model: The compiled Keras model.
    """
    # Define the input layer
    input_layer = Input(shape=(input_dimension,))
    
    # Define the first hidden layer
    hidden_layer1 = Dense(hidden_layer1_size)(input_layer)
    
    # Apply dropout after the first hidden layer
    dropout_layer1 = Dropout(dropout_rate)(hidden_layer1)
    
    # Define the second hidden layer with ReLU activation
    hidden_layer2 = Dense(hidden_layer2_size, activation='relu')(dropout_layer1)
    
    # Apply dropout after the second hidden layer
    dropout_layer2 = Dropout(dropout_rate)(hidden_layer2)
    
    # Define the third hidden layer with ReLU activation
    hidden_layer3 = Dense(hidden_layer3_size, activation='relu')(dropout_layer2)
    
    # Apply dropout after the third hidden layer
    dropout_layer3 = Dropout(dropout_rate)(hidden_layer3)
    
    # Define the output layer with softmax activation
    output_layer = Dense(output_size, activation='softmax')(dropout_layer3)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model with Adam optimizer and Kullback-Leibler divergence loss
    model.compile(optimizer=optimizers.Adam(learning_rate), loss=losses.KLDivergence(reduction='sum_over_batch_size'), metrics=['accuracy'])
    
    return model

def train_neural_network_keras(X_train: ndarray, y_train: ndarray, X_dev: ndarray, y_dev: ndarray, verbose: int = 1) -> tf.keras.Model:
    """
    Train a neural network model using Keras with early stopping and model checkpointing.

    Args:
        X_train (ndarray): The training data.
        y_train (ndarray): The training labels.
        X_dev (ndarray): The development data.
        y_dev (ndarray): The development labels.
        verbose (int): Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).

    Returns:
        tf.keras.Model: The trained Keras model.
    """
    # Define the dimensions of the input and hidden layers
    input_dimension = X_train.shape[1]
    hidden_layer1_size = 384
    hidden_layer2_size = 192
    hidden_layer3_size = 96
    output_size = y_train.shape[1]
    
    # Define the dropout rate, learning rate, number of epochs, and batch size
    dropout_rate = 0.1
    learning_rate = 0.0003
    num_epochs = 20
    batch_size = 256
    
    # Define the early stopping and model checkpointing callbacks
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('best_model.weights.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True)

    # Train the model on the CPU
    with tf.device('/cpu:0'): 
        model = build_neural_network(input_dimension, hidden_layer1_size, hidden_layer2_size, hidden_layer3_size, output_size, dropout_rate, learning_rate)
        
        model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, shuffle=True, validation_data=(X_dev, y_dev), callbacks=[early_stopping_callback, model_checkpoint_callback], verbose=verbose)

    # Load the weights from the best model
    model.load_weights('best_model.weights.h5')

    # Remove the best model file
    os.remove('best_model.weights.h5')

    return model


def print_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Display various classification metrics in a formatted manner.

    Args:
        y_true (np.ndarray): True labels, can be 1D or 2D array.
        y_pred (np.ndarray): Predicted labels, can be 1D or 2D array.

    Returns:
        None

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> print_classification_metrics(y_true, y_pred)
    """

    # Convert 2D arrays to 1D by taking the index of the maximum value in each row
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Print the header for the metrics table
    print(f"{'Metric':<35} {'Score':>10}")
    print("=" * 45)
    
    # Print accuracy score
    print(f"{'Accuracy Score:':<35} {accuracy_score(y_true, y_pred):>10.5f}")
    
    # Print balanced accuracy score
    print(f"{'Balanced Accuracy Score:':<35} {balanced_accuracy_score(y_true, y_pred):>10.5f}")
    
    # Print F1 score with weighted average
    print(f"{'F1 Score (weighted):':<35} {f1_score(y_true, y_pred, average='weighted'):>10.5f}")
    
    # Print Cohen's kappa score
    print(f"{'Cohen Kappa Score:':<35} {cohen_kappa_score(y_true, y_pred):>10.5f}")
    
    # Print Matthews correlation coefficient
    print(f"{'Matthews Correlation Coefficient:':<35} {matthews_corrcoef(y_true, y_pred):>10.5f}")
    
    # Print the classification report
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))
    
    # Print the confusion matrix
    print("\nConfusion Matrix:\n")
    # Use pd.crosstab to create a confusion matrix with row and column labels
    # Format the numbers with commas for better readability
    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).map("{:,}".format))

