import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, roc_auc_score
import time
import numpy as np
import matplotlib.pyplot as plt


TARGET = 'new_child'
def load_data(filepath, cols_num=None):
    """Load the dataset from a specified filepath."""
    if cols_num is not None:
        temp = pd.read_csv(filepath, nrows=1)
        print(f"# of columns total = {len(temp.columns)}")
        # Generate the list of first 1000 columns
        cols_to_use = temp.columns.tolist()[:cols_num]
        cols_to_use.append(TARGET)
        data = pd.read_csv(filepath, usecols=cols_to_use)
    else:
        data = pd.read_csv(filepath)
        
    X = data.drop(columns=[TARGET])  # Adjust column name as necessary
    y = data[TARGET]  # Adjust column name as necessary
    return X, y

def perform_grid_search(X_train, y_train):
    """Perform grid search to find the best hyperparameters."""
    return {
        'max_depth': 3,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'subsample': 0.8
    }

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 0.9]
    }
    model = xgb.XGBClassifier(
        #use_label_encoder=False,
        tree_method= 'hist',
        #device='cuda'
        #gpu_id=0,  # Starting GPU, multi-GPU handled internally
    )
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=5, 
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def train_final_model(X, y, best_params):
    """Train the final model on the entire dataset using the best hyperparameters."""
    model = xgb.XGBClassifier(
        **best_params,  
        #use_label_encoder=False
    )
    # accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    # f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
    # precision = cross_val_score(model, X, y, cv=5, scoring='precision')
    # recall = cross_val_score(model, X, y, cv=5, scoring='recall')
    # roc_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    # print({
    #     'cv_accuracy': np.mean(accuracy),
    #     'cv_f1': np.mean(f1),
    #     'cv_precision': np.mean(precision),
    #     'cv_recall': np.mean(recall),
    #     'cv_roc_auc': np.mean(roc_auc)
    # })
    model.fit(X, y)
    return model

def train_final_model_early_stopping(X, y, best_params):
    pass



def save_model(model, filename):
    """Save the model to disk."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    """Load a model from disk."""
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def evaluate_model(y_true, y_pred, y_prob):
    """Evaluate the model using multiple metrics, including AUC-ROC."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)  # y_prob is the probability estimates of the positive class
    r_squared = r2_score(y_true, y_prob)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'r2': r_squared,
    }

def plot_feature_importance(model):
    """Plot feature importance of the model."""
    # Get feature importance
    importance = model.get_booster().get_score(importance_type='gain')

    # Sort the feature importance in descending order
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    sorted_importance = sorted_importance[:10] + sorted_importance[-5:]
    # Prepare data for plotting
    features = [item[0] for item in sorted_importance]
    scores = [item[1] for item in sorted_importance]

    # Create a bar plot
    plt.figure(figsize=(10, 8))
    plt.barh(features, scores)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Invert the Y-axis to show the highest importance on top
    plt.savefig("feature_importance.png", format='png', bbox_inches='tight')
    plt.close()

def main():
    # Load data
    load_model('models/final_model_full.pkl')
    plot_feature_importance(final_model_full)
    exit(0)
    st = time.time()
    X, y = load_data('data/train_data_combined_after_imputing.csv', 10000)
    print(f"loading time: {time.time()-st} seconds")
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=41
    )
    
    # Find best hyperparameters
    best_params = perform_grid_search(X_train, y_train)
    print("Best parameters:", best_params)
    
    # Train final model on the full training data
    # final_model = train_final_model(X_train, y_train, best_params)
    
    # Validate on the test set
    # y_pred = final_model.predict(X_test)
    # y_prob = final_model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # print("pred")
    # print(y_pred)
    # print("probbb")
    # print(y_prob)
    # print("test")
    # print(y_test)
    # print(evaluate_model(y_test, y_pred, y_prob))
    
    # Optionally, re-train the final model on the entire dataset
    final_model_full = train_final_model(X, y, best_params)
    plot_feature_importance(final_model_full)
    # save_model(final_model, 'models/final_model.pkl')
    save_model(final_model_full, 'models/final_model_full.pkl')
    
if __name__ == '__main__':
    main()
