import numpy as np
import pandas as pd
import pickle
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
    
print("Loading training and testing data...", flush=True)
with open("income_training_by_year.pkl", "rb") as pkl_file:
    train_grouped_by_year = dict(pickle.load(pkl_file))
    
with open("income_testing_by_year.pkl", "rb") as pkl_file:
    test_grouped_by_year = dict(pickle.load(pkl_file))
  
print("Loading mappings...", flush=True)
# Load mapping
with open("mappings/gron_2020_mappings.pkl", "rb") as pkl_file:
    mappings = dict(pickle.load(pkl_file))
temp_mappings = {}
for key, value in mappings.items():
    temp_mappings[int(key)] = int(value)
mappings = temp_mappings
    
inflation_rates = {2012: .0234,
                    2013: .0474,
                    2014: .0713,
                    2015: .0803,
                    2016: .0859,
                    2017: .0888,
                    2018: .1013,
                    2019: .1167,
                    2020: .1399,
                    2021: .1509}
 
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
yearly_medians = []
years = list(train_grouped_by_year.keys())[1:]
    
for year in years:
    differences = []
    incomes = []
    
    founds = 0
    total = 0
    testing_data = test_grouped_by_year[year]
    #print(year, len(testing_data), flush=True)
    for pair in testing_data:
        rin = pair[0]
        income = pair[1]
            
        if rin not in mappings:
            continue
        embedding_id = mappings[rin]
        
        total += 1
        if embedding_id in baseline_dict:
            founds += 1
            baseline = baseline_dict[embedding_id]
            adjusted_baseline = baseline + (baseline * inflation_rates[year])
            difference = np.abs(adjusted_baseline - income)
            differences.append(difference)
            incomes.append(income)
            
    print(year, total, flush=True)

    yearly_differences.append(np.mean(differences))
    yearly_medians.append(np.median(incomes))

#plt.plot(years, yearly_differences, linestyle='dashed', label="baseline - 2011 + inflation", color='black') 
#plt.plot(years, yearly_medians, linestyle='dotted', label='baseline - yearly median', color='gray')
 
#emb_files = ["random", "lr_steve_full_network_2010_0", "lr_steve_full_network_2010_10",
#            "lr_steve_full_network_2010_20", "lr_steve_full_network_2010_30"]
emb_root = "gron_full_network_2020_"
emb_nums = [str(i) for i in range(0, 100, 10)]
emb_files = [emb_root + num for num in emb_nums]

for emb in emb_files:
    print(emb, flush=True)
    # Load embeddings
    if emb == "random":
        with open("embeddings/gron_full_network_2020_0.emb", "rb") as pkl_file:
            embeddings = pickle.load(pkl_file)
        embeddings = np.random.rand(embeddings.shape[0], embeddings.shape[1])
    else:
        with open("embeddings/" + emb + ".emb", "rb") as pkl_file:
            embeddings = pickle.load(pkl_file)
		
    print("Training classifiers...", flush=True)
	# Train the big model
    train_embeddings = []
    train_labels = []
        
    yearly_differences = []
    yearly_scores = []
    
    for year in years:
        training_data = train_grouped_by_year[year]
        
        for pair in training_data:
            rin = pair[0]
            income = pair[1]
            
            if rin not in mappings:
                continue
            embedding_id = mappings[rin]
            embedding = embeddings[embedding_id]
            if embedding_id not in baseline_dict:
                continue
            embedding = normalize(embedding)
            #baseline = baseline_dict[embedding_id]
            #embedding = np.append(embedding, [baseline], axis=0)
            
            train_embeddings.append(embedding)
            train_labels.append(income)
            
    model = LinearRegression()
    model.fit(train_embeddings, train_labels)
    
    for year in years:
        
        differences = []
        test_embeddings = []
        test_labels = []
        
        testing_data = test_grouped_by_year[year]
        for pair in testing_data:
            rin = pair[0]
            income = pair[1]
            if rin not in mappings:
                continue
            embedding_id = mappings[rin]
            embedding = embeddings[embedding_id]
            if embedding_id not in baseline_dict:
                continue
            embedding = normalize(embedding)
            #baseline = baseline_dict[embedding_id]
            #embedding = np.append(embedding, [baseline], axis=0)
            
            test_embeddings.append(embedding)
            test_labels.append(income)
        
        predictions = model.predict(test_embeddings)
        score = model.score(test_embeddings, test_labels)
        differences = []
        
        for i in range(len(predictions)):
            difference = np.abs(predictions[i] - test_labels[i])
            differences.append(difference)
            
        yearly_average = np.mean(differences)
        yearly_differences.append(yearly_average)
        #print(year, ":", yearly_average, flush=True)
        yearly_scores.append(score)
        print(year, ":", score, flush=True)
    
    #plt.plot(years, yearly_differences, label=emb)
    plt.plot(years, yearly_scores, label=emb)
    
plt.title("Linear Regression on 2020 Groningen Network Embeddings\n (Single Model)")
#plt.ylabel("Average absolute deviation (Euros)")
plt.ylabel("R^2")
plt.xlabel("Prediction Year")
plt.legend(loc='upper center', fontsize='6', bbox_to_anchor=(1.19, 0.80), ncol=1)
plt.savefig("results/comparison_normal_r2_gron_2020.png", bbox_inches="tight")
#plt.show()
        
        
    
