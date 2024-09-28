import os
import shutil
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from numpy import ndarray
from sklearn.metrics import (accuracy_score, classification_report, f1_score, hamming_loss,
                             jaccard_score)
from sklearn.model_selection import (KFold, train_test_split)
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)


class MultilabelDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Dictionary of input_ids, attention_masks, etc.
        self.labels = labels  # Numpy array of labels

    def __getitem__(self, idx):
        # Convert each item to a PyTorch tensor
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Labels need to be floats for BCEWithLogitsLoss
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)


def get_tokenizer(model_name: str = "bert-base-uncased") -> AutoTokenizer:
    """
    Get the tokenizer for the specified pre-trained model.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        AutoTokenizer: The tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def build_transformer_model_multilabel(
    num_labels: int, model_name: str = "bert-base-uncased"
) -> AutoModelForSequenceClassification:
    """
    Build a Huggingface Transformer model for multilabel classification.

    Args:
        num_labels (int): The number of labels for multilabel classification.
        model_name (str): The name of the pre-trained model to use.

    Returns:
        AutoModelForSequenceClassification: The Huggingface Transformer model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type="multi_label_classification"
    )
    return model


def train_transformer_multilabel(
    texts_train: List[str],
    labels_train: np.ndarray,
    texts_val: List[str],
    labels_val: np.ndarray,
    model_name: str = "bert-base-uncased",
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 2,
    output_dir: str = "./results",
    fp16: bool = True,
    bf16: bool = False,
    verbose: bool = True,
) -> AutoModelForSequenceClassification:
    """
    Train a Huggingface Transformer model for multilabel classification.

    Args:
        texts_train (List[str]): The training texts.
        labels_train (np.ndarray): The training labels.
        texts_val (List[str]): The validation texts.
        labels_val (np.ndarray): The validation labels.
        model_name (str): The name of the pre-trained model.
        batch_size (int): The batch size.
        num_epochs (int): The number of epochs.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay.
        early_stopping_patience (int): Early stopping patience.
        output_dir (str): Directory to save model checkpoints.
        fp16 (bool): Whether to use mixed precision training.
        bf16 (bool): Whether to use bfloat16 precision training.
        verbose (bool): Whether to print training progress.

    Returns:
        AutoModelForSequenceClassification: The trained model.
    """
    # Load tokenizer and model
    tokenizer = get_tokenizer(model_name)
    model = build_transformer_model_multilabel(labels_train.shape[1], model_name)

    # Tokenize texts
    train_encodings = tokenizer(
        texts_train, truncation=True, padding=True, max_length=512
    )
    val_encodings = tokenizer(
        texts_val, truncation=True, padding=True, max_length=512
    )

    # Create datasets
    train_dataset = MultilabelDataset(train_encodings, labels_train)
    val_dataset = MultilabelDataset(val_encodings, labels_val)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=fp16,
        bf16=bf16,
        metric_for_best_model="eval_loss",
        logging_steps=1,
        logging_first_step=True,
        report_to="none",
        disable_tqdm=not verbose,
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # Train the model
    trainer.train()

    return model


def cross_val_predict_transformer_multilabel(
    texts: List[str],
    labels: np.ndarray,
    n_splits: int = 5,
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 2,
    fp16: bool = True,
    bf16: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Perform cross-validation and get out-of-sample predicted probabilities for all instances
    using a Huggingface Transformer model designed for multilabel classification.

    Args:
        texts (List[str]): The list of texts.
        labels (np.ndarray): The label matrix (binary matrix for multilabel classification).
        n_splits (int): Number of folds for cross-validation.
        model_name (str): The name of the pre-trained model.
        batch_size (int): The batch size.
        num_epochs (int): The number of epochs.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay.
        early_stopping_patience (int): Early stopping patience.
        fp16 (bool): Whether to use mixed precision training.
        bf16 (bool): Whether to use bfloat16 precision training.
        verbose (bool): Whether to print training progress.

    Returns:
        np.ndarray: Predicted probabilities for each instance, obtained via cross-validation.
    """
    # Initialize KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=271828)

    # Initialize an array to hold the predicted probabilities
    y_pred = np.zeros_like(labels, dtype=float)

    # Loop over each fold
    for fold, (train_index, test_index) in enumerate(kf.split(texts)):
        print(f"Processing fold {fold + 1}/{n_splits}")

        # Split the data into training and test sets for this fold
        texts_train_fold = [texts[i] for i in train_index]
        texts_test_fold = [texts[i] for i in test_index]
        labels_train_fold = labels[train_index]
        labels_test_fold = labels[test_index]

        # Further split the training fold into training and validation sets for early stopping
        (
            texts_train_subfold,
            texts_dev_subfold,
            labels_train_subfold,
            labels_dev_subfold,
        ) = train_test_split(
            texts_train_fold, labels_train_fold, test_size=0.1, random_state=fold
        )

        # Define output directory for this fold
        output_dir = f"./results_fold_{fold}"

        # Train the model on the training subfold
        model = train_transformer_multilabel(
            texts_train_subfold,
            labels_train_subfold,
            texts_dev_subfold,
            labels_dev_subfold,
            model_name=model_name,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            output_dir=output_dir,
            fp16=fp16,
            bf16=bf16,
            verbose=verbose,
        )

        # Tokenize the test texts
        tokenizer = get_tokenizer(model_name)
        test_encodings = tokenizer(
            texts_test_fold, truncation=True, padding=True, max_length=512
        )
        test_dataset = MultilabelDataset(test_encodings, labels_test_fold)

        # Get the predicted probabilities on the test fold
        trainer = Trainer(model=model)
        predictions = trainer.predict(test_dataset)
        y_pred_fold = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()

        # Store the predicted probabilities in the correct positions
        y_pred[test_index] = y_pred_fold

        # Clean up the output directory
        shutil.rmtree(output_dir)

    # Return the predicted probabilities for all instances
    return y_pred


def build_neural_network_multilabel(input_dimension: int, hidden_layer1_size: int, hidden_layer2_size: int, 
                         hidden_layer3_size: int, output_size: int, dropout_rate: float, learning_rate: float) -> Model:
    """
    Build a neural network model with three hidden layers and dropout for multilabel classification.

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
    
    # Define the output layer with sigmoid activation for multilabel classification
    output_layer = Dense(output_size, activation='sigmoid')(dropout_layer3)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model with Adam optimizer and binary cross-entropy loss
    model.compile(
        optimizer=optimizers.Adam(learning_rate), 
        loss='binary_crossentropy', 
        metrics=['binary_accuracy']
    )
    
    return model

def train_neural_network_keras_multilabel(X_train: ndarray, y_train: ndarray, X_dev: ndarray, y_dev: ndarray, verbose: int = 1) -> tf.keras.Model:
    """
    Train a neural network model using Keras with early stopping and model checkpointing for multilabel classification.

    Args:
        X_train (ndarray): The training data.
        y_train (ndarray): The training labels (binary matrix for multilabel).
        X_dev (ndarray): The development data.
        y_dev (ndarray): The development labels (binary matrix for multilabel).
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
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'best_model.weights.h5', 
        monitor='val_binary_accuracy', 
        save_best_only=True, 
        save_weights_only=True
    )

    # Train the model
    model = build_neural_network_multilabel(
        input_dimension, 
        hidden_layer1_size, 
        hidden_layer2_size, 
        hidden_layer3_size, 
        output_size, 
        dropout_rate, 
        learning_rate
    )
    
    model.fit(
        X_train, 
        y_train, 
        epochs=num_epochs, 
        batch_size=batch_size, 
        shuffle=True, 
        validation_data=(X_dev, y_dev), 
        callbacks=[early_stopping_callback, model_checkpoint_callback], 
        verbose=verbose
    )

    # Load the weights from the best model
    model.load_weights('best_model.weights.h5')

    # Remove the best model file
    os.remove('best_model.weights.h5')

    return model


def cross_val_predict_neural_network_keras_multilabel(
    X: ndarray, 
    y: ndarray, 
    n_splits: int = 5, 
    verbose: int = 1
) -> ndarray:
    """
    Perform cross-validation and get out-of-sample predicted probabilities for all instances
    using a neural network model designed for multilabel classification.

    Args:
        X (ndarray): The dataset features.
        y (ndarray): The label matrix (binary matrix for multilabel classification).
        n_splits (int): Number of folds for cross-validation.
        verbose (int): Verbosity mode for model training (0 = silent, 1 = progress bar, 2 = one line per epoch).

    Returns:
        ndarray: Predicted probabilities for each instance, obtained via cross-validation.
    """
    # Initialize KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=271828)
    
    # Initialize an array to hold the predicted probabilities
    y_pred = np.zeros_like(y, dtype=float)
    
    # Loop over each fold
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Processing fold {fold + 1}/{n_splits}")
        
        # Split the data into training and test sets for this fold
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        # Further split the training fold into training and validation sets for early stopping
        X_train_subfold, X_dev_subfold, y_train_subfold, y_dev_subfold = train_test_split(
            X_train_fold, y_train_fold, test_size=0.1, random_state=fold
        )
        
        # Train the model on the training subfold
        model = train_neural_network_keras_multilabel(
            X_train_subfold, y_train_subfold, X_dev_subfold, y_dev_subfold, verbose=verbose
        )
        
        # Get the predicted probabilities on the test fold
        y_pred_fold = model.predict(X_test_fold)
        
        # Store the predicted probabilities in the correct positions
        y_pred[test_index] = y_pred_fold
        
    # Return the predicted probabilities for all instances
    return y_pred


def print_multilabel_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Display various classification metrics for multilabel classification in a formatted manner.

    Args:
        y_true (np.ndarray): True labels, expected to be a binary matrix of shape (n_samples, n_classes).
        y_pred (np.ndarray): Predicted labels, expected to be a binary matrix of shape (n_samples, n_classes).

    Returns:
        None

    Example:
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]])
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]])
        >>> print_multilabel_classification_metrics(y_true, y_pred)
    """
    # Ensure y_true and y_pred are binary matrices
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be binary matrices of shape (n_samples, n_classes)")
    
    # Print the header for the metrics table
    print(f"{'Metric':<35} {'Score':>10}")
    print("=" * 45)
    
    # Subset accuracy (exact match ratio)
    subset_accuracy = accuracy_score(y_true, y_pred)
    print(f"{'Subset Accuracy:':<35} {subset_accuracy:>10.5f}")
    
    # Hamming loss
    hamming = hamming_loss(y_true, y_pred)
    print(f"{'Hamming Loss:':<35} {hamming:>10.5f}")

    # Micro-average F1 score
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    print(f"{'F1 Score (micro-average):':<35} {f1_micro:>10.5f}")

    # Macro-average F1 score
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"{'F1 Score (macro-average):':<35} {f1_macro:>10.5f}")

    # Weighted-average F1 score
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"{'F1 Score (weighted-average):':<35} {f1_weighted:>10.5f}")

    # Jaccard similarity coefficient
    jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    print(f"{'Jaccard Score (samples):':<35} {jaccard:>10.5f}")

    # Print the classification report
    print("\nClassification Report (per class):\n")
    class_names = [f'Class {i}' for i in range(y_true.shape[1])]
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Print the confusion matrix per class
    print("\nConfusion Matrix (per class):\n")
    for i in range(y_true.shape[1]):
        print(f"Class {i}:\n")
        cm = pd.crosstab(
            y_true[:, i],
            y_pred[:, i],
            rownames=['True'],
            colnames=['Predicted'],
            margins=True
        )
        print(cm)
        print("\n")

