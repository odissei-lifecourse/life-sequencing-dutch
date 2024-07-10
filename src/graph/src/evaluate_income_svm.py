import numpy as np
import pandas as pd
import pickle
import random
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
    
print("Loading training and testing data...", flush=True)
with open("income_training_by_year.pkl", "rb") as pkl_file:
    train_grouped_by_year = dict(pickle.load(pkl_file))
    
with open("income_testing_by_year.pkl", "rb") as pkl_file:
    test_grouped_by_year = dict(pickle.load(pkl_file))
  
print("Loading mappings...", flush=True)
# Load mapping
with open("mappings/family_2010.pkl", "rb") as pkl_file:
    mappings = dict(pickle.load(pkl_file))
 
fig, _ = plt.subplots(figsize=(5,3), dpi=150)

# First plot baseline
with open("baseline_2011.pkl", "rb") as pkl_file:
    baseline_list = list(pickle.load(pkl_file))
baseline_dict = {}
for pair in baseline_list:
    rin = pair[0]
    income = pair[1]
    if rin in mappings:
        mapped_id = mappings[rin]
    
        baseline_dict[mapped_id] = income
        
yearly_differences = []
years = list(train_grouped_by_year.keys())
    
for year in years:
    differences = []
    
    testing_data = test_grouped_by_year[year]
    for pair in testing_data:
        rin = pair[0]
        income = pair[1]
            
        if rin not in mappings:
            continue
        embedding_id = mappings[rin]
        
        if embedding_id in baseline_dict:
            baseline = baseline_dict[embedding_id]
            difference = np.abs(baseline - income)
            differences.append(difference)

    yearly_differences.append(np.mean(differences))

plt.plot(years, yearly_differences, 'g--', label="baseline", color='black')  

 
emb_files = ["10", "20", "30", "40"]
for emb in emb_files:

    # Load embeddings
    with open("embeddings/family_2010_" + emb + ".emb", "rb") as pkl_file:
        embeddings = pickle.load(pkl_file)
        
    yearly_differences = []
    
    print("Training classifiers...", flush=True)
    # Train a separate classifier per year
    for year in years:
        training_data = train_grouped_by_year[year]
        testing_data = test_grouped_by_year[year]
        
        train_embeddings = []
        train_labels = []
        
        test_embeddings = []
        test_labels = []
        
        for pair in training_data:
            rin = pair[0]
            income = pair[1]
            
            if rin not in mappings:
                continue
            embedding_id = mappings[rin]
            embedding = embeddings[embedding_id]
            
            train_embeddings.append(embedding)
            train_labels.append(income)
            
        for pair in testing_data:
            rin = pair[0]
            income = pair[1]
            
            if rin not in mappings:
                continue
            embedding_id = mappings[rin]
            embedding = embeddings[embedding_id]
            
            test_embeddings.append(embedding)
            test_labels.append(income)
            
        model = SGDRegressor()
        model.fit(train_embeddings, train_labels)
        
        predictions = model.predict(test_embeddings)
        differences = []
        
        for i in range(len(predictions)):
            difference = np.abs(predictions[i] - test_labels[i])
            differences.append(difference)
            
        yearly_average = np.mean(differences)
        yearly_differences.append(yearly_average)
        print(year, ":", yearly_average, flush=True)
    
    plt.plot(years, yearly_differences, label=emb)
    
plt.title("SVM on 2010 Family Embeddings")
plt.ylabel("Average absolute deviation (Euros)")
plt.xlabel("Prediction Year")
plt.legend()
plt.savefig("results/svm_family_2010.png", bbox_inches="tight")
plt.show()
        
        
    
