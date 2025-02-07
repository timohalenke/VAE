import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt



class LogisticRegressionModel:
    def __init__(self, label_column="patient_category"):
        self.label_column = label_column
        self.best_model = None
        self.test_set_roc = None
        self.test_set_metrics = None
        self.real_data_report = None
    def fit(self, df):
        # Splitting the data into training (70%) and testing (30%) sets
        X_train, X_test, y_train, y_test = train_test_split(
            df.iloc[:, :1101], df[self.label_column].astype(int), test_size=0.3, random_state=42
        )
        # Define the logistic regression model
        model = LogisticRegression(max_iter=int(1e6), penalty="l2")
        # Define the grid of hyperparameters to search
        param_grid = {'C': [10]}
        # Perform cross-validation using GridSearchCV
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            error_score='raise',
            verbose=3
        )
        grid_search.fit(X_train, y_train)
        # Store the best model
        self.best_model = grid_search.best_estimator_
        # Display the best parameters and cross-validation accuracy
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Validation Accuracy:", grid_search.best_score_)
        # Evaluate the best model on the test set
        y_pred = self.best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        print("\nClassification Report for Test Set:\n", classification_report(y_test, y_pred))
        # Store ROC and metrics
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=self.best_model.classes_[1])
        roc_auc = auc(fpr, tpr)
        self.test_set_roc = (fpr, tpr, roc_auc)
        self.test_set_metrics = report
        return self.best_model
    def evaluate(self, df, label="Gen Data"):
        y_true = df[self.label_column]
        y_pred = self.predict(df)
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        print(f"\nClassification Report for {label}:\n", classification_report(y_true, y_pred))
        # Calculate ROC curve
        y_prob = self.best_model.predict_proba(df.iloc[:, :1101].values)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=self.best_model.classes_[1])
        roc_auc = auc(fpr, tpr)
        # Plot ROC curves
        self.plot_roc_curves(self.test_set_roc, (fpr, tpr, roc_auc), self.test_set_metrics, report, label)
        return fpr, tpr, roc_auc
    def predict(self, df):
        if self.best_model is None:
            raise ValueError("Model has not been fit. Please fit the model first.")
        X = df.iloc[:, :1101].values
        y_pred = self.best_model.predict(X)
        return y_pred
    def plot_roc_curves(self, first_roc, second_roc, first_report, second_report, second_label):
        plt.figure(figsize=(14, 10))
        # Plot first ROC curve
        fpr1, tpr1, roc_auc1 = first_roc
        plt.plot(fpr1, tpr1, label=f"First Test Set ROC Curve (AUC = {roc_auc1:.2f})")
        # Plot second ROC curve
        fpr2, tpr2, roc_auc2 = second_roc
        plt.plot(fpr2, tpr2, label=f"{second_label} ROC Curve (AUC = {roc_auc2:.2f})")
        # Plot settings
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves with Metrics")
        plt.legend(loc="lower right")
        plt.grid()
        # Extract metrics for display
        def format_metrics(report):
            return (f"Class 0:\n"
                    f"  Precision: {report['0']['precision']:.2f}\n"
                    f"  Recall:      {report['0']['recall']:.2f}\n"
                    f"  F1-score:  {report['0']['f1-score']:.2f}\n\n"
                    f"Class 1:\n"
                    f"  Precision: {report['1']['precision']:.2f}\n"
                    f"  Recall:      {report['1']['recall']:.2f}\n"
                    f"  F1-score:  {report['1']['f1-score']:.2f}\n\n"
                    f"Accuracy:   {report['accuracy']:.2f}\n\n"
                    f"Avg:\n"
                    f"  Precision: {report['macro avg']['precision']:.2f}\n"
                    f"  Recall:      {report['macro avg']['recall']:.2f}\n"
                    f"  F1-score:  {report['macro avg']['f1-score']:.2f}\n"
                    #f"Weighted Avg:\n"
                    #f"  Precision: {report['weighted avg']['precision']:.2f}\n"
                    #f"  Recall: {report['weighted avg']['recall']:.2f}\n"
                    #f"  F1-score: {report['weighted avg']['f1-score']:.2f}"
                   )
        # Annotate metrics
        plt.text(0.92, 0.1, f"Test Set Metrics:\n\n{format_metrics(first_report)}",
                 bbox=dict(facecolor='lightblue', alpha=0.5), fontsize=10)
        plt.text(0.75, 0.1, f"{second_label} Metrics:\n\n{format_metrics(second_report)}",
                 bbox=dict(facecolor='lightblue', alpha=0.5), fontsize=10)
        plt.show()
# Initialize the model
logistic_model = LogisticRegressionModel()
# Fit the model to the first dataset
lr_model = logistic_model.fit(df_train_set_real_and_sim)
fpr2, tpr2, roc_auc2 = logistic_model.evaluate(df_real_test)