from scipy.spatial import distance as dst
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from torch import Tensor
from sentence_transformers import util
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import random
import csv
import json
from nearest_neighbor import build_index, get_nearest_neighbor_e2e

# Computes/Loads any values that are used to evaluate all embedding sets, such as income at age 30 or marriage
def precompute_global(var_type):

    # Get dict of income at age 30, organized by year and RINPERSOONNUM
    if var_type == 'income':
        with open("data/processed/income_by_year.pkl", 'rb') as pkl_file:
            data = dict(pickle.load(pkl_file))

        return data
        
    #-----------------------------------------------------------------------------------------------------------------#
    # Get dict of marriage, organized by event year and subindexed by RINPERSOONNUM
    if var_type == 'marriage':
        with open("data/processed/marriages_by_year.pkl", "rb") as pkl_file:
            marriage_data = dict(pickle.load(pkl_file))
            
        with open("data/processed/partnerships_by_year.pkl", "rb") as pkl_file:
            partnership_data = dict(pickle.load(pkl_file))
            
        with open("data/processed/id_to_gender_map.pkl", "rb") as pkl_file:
            gender_map = dict(pickle.load(pkl_file))
            
        with open("data/processed/full_male_list.pkl", "rb") as pkl_file:
            full_male_list = list(pickle.load(pkl_file))
            
        with open("data/processed/full_female_list.pkl", "rb") as pkl_file:
            full_female_list = list(pickle.load(pkl_file))
                
        marriage_data_by_year = {}
        partnership_data_by_year = {}
        
        seen_marriages = set()
        seen_partnerships = set()
        
        for year in marriage_data:
            marriage_data_by_year[int(year)] = {}
            partnership_data_by_year[int(year)] = {}
        
            relevant_marriages = marriage_data[year]
            relevant_partnerships = partnership_data[year]
            
            for person in relevant_marriages:
                partner = relevant_marriages[person]
                
                if person in seen_marriages or partner in seen_marriages:
                    continue
                    
                seen_marriages.add(person)
                seen_marriages.add(partner)
                
                real_pair = (person, partner)
                
                partner_gender = gender_map[partner]
                
                if partner_gender == 1:
                    partner_list = full_male_list
                else:
                    partner_list = full_female_list
                    
                fake_partner = random.choice(partner_list)
                        
                fake_pair = (person, fake_partner)
                
                marriage_data_by_year[int(year)][real_pair] = 1
                marriage_data_by_year[int(year)][fake_pair] = 0
                
            for person in relevant_partnerships:
                partner = relevant_partnerships[person]
                
                if person in seen_partnerships or partner in seen_partnerships:
                    continue
                    
                seen_partnerships.add(person)
                seen_partnerships.add(partner)
                
                real_pair = (person, partner)
                
                partner_gender = gender_map[partner]
                
                if partner_gender == 1:
                    partner_list = full_male_list
                else:
                    partner_list = full_female_list
                    
                fake_partner = random.choice(partner_list)
                        
                fake_pair = (person, fake_partner)
                
                partnership_data_by_year[int(year)][real_pair] = 1
                partnership_data_by_year[int(year)][fake_pair] = 0
                        
        return marriage_data_by_year, partnership_data_by_year
        
    #------------------------------------------------------------------------------------------------------------------#
    if var_type == 'death':
        
        with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/death_years_by_person.pkl", 'rb') as pkl_file:
            death_years_by_person = dict(pickle.load(pkl_file))
            
        years = [x for x in range(2010, 2022)]
        full_person_set = set(death_years_by_person.keys())
        
        deaths_by_year = {}
        death_count_by_year = {}
        
        for year in years:
            deaths_by_year[year] = {}
            death_count_by_year[year] = 0
            
            for person in full_person_set:
                if year == death_years_by_person[person]:
                    deaths_by_year[year][person] = 1
                    death_count_by_year[year] += 1
                else:
                    deaths_by_year[year][person] = 0
        
        #for year in death_count_by_year:
        #    print(year, death_count_by_year[year], flush=True)
            
        return deaths_by_year
    

########################################################################################################################
# Computes/Loads any values that are used in multiple steps to evaluate a single embedding set, such as distance matrices
def precompute_local(embedding_set, top_k=100, only_embedding=False):
    root = embedding_set['root']
    url = embedding_set['url']
    emb_type = embedding_set['type']
    truth_type = embedding_set['truth']

    year = embedding_set['year']

    ##########################################################################################
    # Step 1: Load the embeddings into a dictionary indexed by ID
    embedding_dict = {}

    if emb_type == 'NET':
    
        mapping_url = root + embedding_set['mapping']
    
        # First get mappings
        with open(mapping_url, 'rb') as pkl_file:
            mappings = dict(pickle.load(pkl_file))

        inverse_mappings = {}
        for key, value in mappings.items():
            inverse_mappings[value] = key

        emb_url = root + url
        with open(emb_url, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        for i in range(len(data)):
            embedding = data[i]
            player_id = int(inverse_mappings[i])

            embedding_dict[player_id] = list(embedding)

    if emb_type == 'LLM':
        emb_url = root + url
        with open(emb_url, 'r') as json_file:
            embedding_dict = dict(json.load(json_file))
            
        # Need to typecast into int
        embedding_dict = {int(key):value for key, value in embedding_dict.items()}

    #         for person in data_dict:
    #             relevant = data_dict[person]
    #             key = subtype + "_emb"
    #             emb = relevant[key]

    #             embedding_dict[person] = emb

    if only_embedding:
        return embedding_dict

    ############################################################################################
    # Step 2: Load hops from pkl file
    
    # Overwrite root, hops and ground truth all live here
    hops_url = "/gpfs/ostor/ossc9424/homedir/Dakota_network/ground_truth_hops/" + truth_type + "_" + str(year) + '_hops.pkl'
    with open(hops_url, 'rb') as pkl_file:
        network_hops = dict(pickle.load(pkl_file))

    ############################################################################################
    # Step 3: Load binary connection ground truth
    truth_url = '/gpfs/ostor/ossc9424/homedir/Dakota_network/ground_truth_adjacency/' + truth_type + '_' + str(year) + '_adjacency.pkl' 
    
    with open(truth_url, 'rb') as pkl_file:
        ground_truth_dict = dict(pickle.load(pkl_file))

    ############################################################################################
    # Step 4: Get person x person distance matrix in embedding space

    distance_matrix = {}
    
    #for person in embedding_dict:
    #    distance_matrix[person] = {}

    #nn_dict = get_nearest_neighbor_e2e(corpus_embs_dict=embedding_dict,
    #                                   query_embs_dict=embedding_dict,
    #                                   check_pids=None,
    #                                   top_k=top_k,
    #                                   ignore_self=True,
    #                                   return_index=False)

    #nn_data = nn_dict['result']

    #for person in nn_data:
    #    person_results = nn_data[person]
    #    for result in person_results:
    #        other = result['pid']
    #        score = result['score']

    #        distance_matrix[person][other] = score

    #     embedding_array = []
    #     id_to_index = {}

    #     names = list(embedding_dict.keys())

    #     for i, person in enumerate(names):

    #         embedding = embedding_dict[person]
    #         embedding_array.append(embedding)
    #         id_to_index[person] = i

    #     inverse_mapping = {}
    #     for key, value in id_to_index.items():
    #         inverse_mapping[value] = key

    #     embedding_tensor = Tensor(np.array(embedding_array))
    #     raw_distance_matrix = util.cos_sim(embedding_tensor, embedding_tensor).numpy()

    #     for i in range(len(names)):

    #         person_id = inverse_mapping[i]

    #         for j in range(i, len(names)):

    #             other_id = inverse_mapping[j]

    #             distance = raw_distance_matrix[i][j].item()

    #             distance_matrix[person_id][other_id] = distance
    #             distance_matrix[other_id][person_id] = distance

    return embedding_dict, network_hops, ground_truth_dict, distance_matrix

########################################################################################################################
# Produces histograms for distance distributions, with cohorts defined by the shortest path distance within graph space

def plot_embedding_distances(embedding_dict, hop_dict, distance_matrix, num_samples,
                             title,
                             savename,
                             show=True
                             ):
    num_hops = 3

    distances = {}
    random_distances = []

    # Initialize the distance list for each length of hops
    for i in range(num_hops):
        distances[i + 1] = []
        
    # Subsample the person list to only 100 thousand, no fucking way we can run all of these
    person_list = random.sample(list(embedding_dict.keys()), 100000)

    for person in person_list:

        if person not in hop_dict:
            continue

        person_embedding = embedding_dict[person]

        for i in range(1, num_hops + 1):

            if len(hop_dict[person][i]) > num_samples:
                relevant_connections = random.sample(hop_dict[person][i], num_samples)
            else:
                relevant_connections = hop_dict[person][i]

            # Real connections
            for connection in relevant_connections:
                if connection not in embedding_dict:
                    continue

                distance = None

                if person in distance_matrix:
                    if connection in distance_matrix[person]:
                        distance = distance_matrix[person][connection]

                elif connection in distance_matrix:
                    if person in distance_matrix[connection]:
                        distance = distance_matrix[connection][person]

                if distance == None:
                    other_embedding = embedding_dict[connection]
                
                    person_tensor = Tensor(np.array(person_embedding))
                    other_tensor = Tensor(np.array(other_embedding))
                    
                    distance = util.cos_sim(person_tensor, other_tensor).numpy()[0][0]

                    # Add to lookup table so we don't waste time computing this again
                    if person not in distance_matrix:
                        distance_matrix[person] = {}

                    distance_matrix[person][connection] = distance
                    
                distances[i].append(distance)

        # Get random samples for person
        for i in range(num_samples):

            connection = random.choice(person_list)
            distance = None

            if person in distance_matrix:
                if connection in distance_matrix[person]:
                    distance = distance_matrix[person][connection]

            elif connection in distance_matrix:
                if person in distance_matrix[connection]:
                    distance = distance_matrix[connection][person]

            if distance == None:
                other_embedding = embedding_dict[connection]
                
                person_tensor = Tensor(np.array(person_embedding))
                other_tensor = Tensor(np.array(other_embedding))
                
                distance = util.cos_sim(person_tensor, other_tensor).numpy()[0][0]

                # Add to lookup table so we don't waste time computing this again
                if person not in distance_matrix:
                    distance_matrix[person] = {}
                #if connection not in distance_matrix:
                    #distance_matrix[connection] = {}

                distance_matrix[person][connection] = distance
                #distance_matrix[connection][person] = distance

            random_distances.append(distance)

    # Plot the distributions
    fig, _ = plt.subplots(figsize=(5, 3), dpi=150)
    colors = ['lime', 'blue', 'purple', 'red']

    min_num_samples = np.inf
    for i in range(1, num_hops + 1):
        num_samples = len(distances[i])
        if num_samples < min_num_samples:
            min_num_samples = num_samples

    for i in range(1, num_hops + 1):

        counts, bins = np.histogram(distances[i], bins=100, weights=np.ones_like(distances[i]) / len(distances[i]))
        clean_counts = []
        clean_bins = []
    
        bad_hop_count = 0
    
        for j in range(len(bins)):
            if j == len(counts):
                clean_bins.append(bins[j])
                break            
            
            if (counts[j]*len(distances[i])) > 10:
                clean_counts.append(counts[j])
                clean_bins.append(bins[j])
            else:
                bad_hop_count += 1
                
        print("Excluded", str(bad_hop_count), "hops of distance", str(i), flush=True)

        clean_bins = np.array(clean_bins)
        bin_centers = 0.5*(clean_bins[1:] + clean_bins[:-1])
    
        plt.plot(bin_centers, clean_counts, color=colors[i-1], label=str(i) + " hops away", alpha=0.8)
        #plt.hist(clean_bins[:-1], clean_bins, weights=clean_counts, color=colors[i-1], label=str(i) + " hops away", alpha=0.4)
        #plt.stairs(clean_counts, clean_bins, color=colors[i-1], label=str(i) + " hops away", alpha=0.4)
        #plt.hist(distances[i], bins=100, weights=np.ones_like(distances[i]) / len(distances[i]),
        #plt.hist(distances[i], bins=100, density=True,
        #         color=colors[i - 1], label=str(i) + " hops away", alpha=0.4)

    counts, bins = np.histogram(random_distances, bins=100, weights=np.ones_like(random_distances) / len(random_distances))
    clean_counts = []
    clean_bins = []
    
    bad_fake_count = 0
    
    for i in range(len(bins)):
        if i == len(counts):
            clean_bins.append(bins[i])
            break
            
        if (counts[i]*len(random_distances)) > 10:
            clean_counts.append(counts[i])
            clean_bins.append(bins[i])
        else:
            bad_fake_count += 1
            
    print("Excluded", str(bad_fake_count), "random connections", str(i), flush=True)
    
    clean_bins = np.array(clean_bins)
    bin_centers = 0.5*(clean_bins[1:] + clean_bins[:-1])
    
    plt.plot(bin_centers, clean_counts, color=colors[-1], label='random other', alpha=0.8)
    #plt.hist(clean_bins[:-1], clean_bins, weights=clean_counts, color=colors[-1], label='random other', alpha=0.4)
    #plt.stairs(clean_counts, clean_bins, color=colors[-1], label='random other', alpha=0.4)
    #plt.hist(random_distances, bins=100, weights=np.ones_like(random_distances) / len(random_distances),
    #plt.hist(random_distances, bins=100, density=True,
    #         color=colors[-1], label="random other", alpha=0.4)

    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency of Distance")
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(1.20, 0.8), ncol=1)

    plt.savefig(savename, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.clf()

########################################################################################################################
# Produces histograms for 2 distance distributions, with cohorts by real / fake connections as provided by the ground truth
# ground truth = dict containing real connection lists

def plot_distance_vs_ground_truth(embedding_dict, ground_truth, distance_matrix,
                                  num_samples,
                                  title,
                                  savename,
                                  show=True):
    real_distances = []
    fake_distances = []
    
    all_users = list(embedding_dict.keys())
    reduced_users = random.sample(all_users, 100000)

    for person in reduced_users:

        try:
            person_embedding = embedding_dict[person]
            real_connections = random.sample(ground_truth[person], num_samples)
        except:
            continue

        # Real connections
        for connection in real_connections:

            # ONly people we have embeddings for
            if connection not in embedding_dict:
                continue

            distance = None

            if person in distance_matrix:
                if connection in distance_matrix[person]:
                    distance = distance_matrix[person][connection]

            elif connection in distance_matrix:
                if person in distance_matrix[connection]:
                    distance = distance_matrix[connection][person]

            if distance == None:
                other_embedding = embedding_dict[connection]
                
                person_tensor = Tensor(np.array(person_embedding))
                other_tensor = Tensor(np.array(other_embedding))
                
                distance = util.cos_sim(person_tensor, other_tensor).numpy()[0][0]

                if person not in distance_matrix:
                    distance_matrix[person] = {}
                distance_matrix[person][connection] = distance

            real_distances.append(distance)

        # Fake connections
        for connection in real_connections:
            distance = None

            fake_other = random.choice(reduced_users)
            #while fake_other in real_connections:
            #    fake_other = random.choice(list(embedding_dict.keys()))

            if person in distance_matrix:
                if fake_other in distance_matrix[person]:
                    distance = distance_matrix[person][fake_other]

            elif fake_other in distance_matrix:
                if person in distance_matrix[fake_other]:
                    distance = distance_matrix[fake_other][person]

            if distance == None:
                other_embedding = embedding_dict[fake_other]
                
                person_tensor = Tensor(np.array(person_embedding))
                other_tensor = Tensor(np.array(other_embedding))
                
                distance = util.cos_sim(person_tensor, other_tensor).numpy()[0][0]

                if person not in distance_matrix:
                    distance_matrix[person] = {}
                    
                distance_matrix[person][connection] = distance
                
            fake_distances.append(distance)

    # Plot the distributions
    fig, _ = plt.subplots(figsize=(5, 3), dpi=150)

    real_distances = np.array(real_distances)
    fake_distances = np.array(fake_distances)

    counts, bins = np.histogram(real_distances, bins=100, weights=np.ones_like(real_distances) / len(real_distances))
    clean_counts = []
    clean_bins = []
    
    bad_real_count = 0
    
    for i in range(len(bins)):
        if i == len(counts):
            clean_bins.append(bins[i])
            break
            
        if (counts[i]*len(real_distances)) > 10:
            clean_counts.append(counts[i])
            clean_bins.append(bins[i])
        else:
            bad_real_count += 1

    clean_bins = np.array(clean_bins)
    bin_centers = 0.5*(clean_bins[1:] + clean_bins[:-1])
    
    plt.plot(bin_centers, clean_counts, color='lime', label='Real Connections', alpha=0.8)
    #plt.hist(clean_bins[:-1], clean_bins, weights=clean_counts, color='blue', label='Real Connections', alpha=0.4)
    #plt.stairs(clean_counts, clean_bins, color='blue', label='Real Connections', alpha=0.4)
    #plt.hist(real_distances, bins=100, weights=np.ones_like(real_distances) / len(real_distances),
    #plt.hist(real_distances, bins=100, density=True,
    #         color='blue', label='Real Connections', alpha=0.4)

    counts, bins = np.histogram(fake_distances, bins=100, weights=np.ones_like(fake_distances) / len(fake_distances))
    clean_counts = []
    clean_bins = []
    
    bad_fake_count = 0
    
    for i in range(len(bins)):
        if i == len(counts):
            clean_bins.append(bins[i])
            break
            
        if (counts[i]*len(fake_distances)) > 10:
            clean_counts.append(counts[i])
            clean_bins.append(bins[i])
        else:
            bad_fake_count += 1

    clean_bins = np.array(clean_bins)
    bin_centers = 0.5*(clean_bins[1:] + clean_bins[:-1])
    
    plt.plot(bin_centers, clean_counts, color='red', label='Fake Connections', alpha=0.8)
    #plt.hist(clean_bins[:-1], clean_bins, weights=clean_counts, color='red', label='Fake Connections', alpha=0.4)
    #plt.stairs(clean_counts, clean_bins, color='red', label='Fake Connections', alpha=0.4) 

    #plt.hist(fake_distances, bins=100, weights=np.ones_like(fake_distances) / len(fake_distances),
    #plt.hist(fake_distances, bins=100, density=True,
    #         color='red', label='Fake Connections', alpha=0.4)

    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency of Distance")
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(1.20, 0.8), ncol=1)

    plt.savefig(savename, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.clf()
        
    print("Excluded", str(bad_real_count), "real connections,", str(bad_fake_count), "fake connections", flush=True)

########################################################################################################################
def linear_variable_prediction(embedding_dict, variable_dict, years, dtype="single", baseline=None):

    return_dict = {}
    baseline_return_dict = {}

    embeddings_by_year = {}
    baseline_by_year = {}
    labels_by_year = {}
    test_counts_by_year = {}
    
    full_baseline_list = []
    full_embedding_list = []
    full_label_list = []

    test_counts_overall = 0

    for year in years:
    
        test_counts_by_year[year] = 0

        if year not in embeddings_by_year:
            embeddings_by_year[year] = []
            labels_by_year[year] = []
            baseline_by_year[year] = []
        
        if dtype == "single":
        
            model = LinearRegression()
            
            for person in variable_dict[year]:
            # Get the players with a valid variable for this year
                if person not in embedding_dict:
                    continue
                    
                if baseline is not None and person not in baseline:
                    continue

                embedding = embedding_dict[person]
                value = variable_dict[year][person]
            
                # If called with a baseline dict, append the values to the end of the embedding
                if baseline is not None:
                    baseline_values = baseline[person]
                    embedding += baseline_values
                    
                    baseline_by_year[year].append(baseline[person])
                    full_baseline_list.append(baseline[person])
            
                embeddings_by_year[year].append(embedding)
                labels_by_year[year].append(value)
                
                full_embedding_list.append(embedding)
                full_label_list.append(value)
                
        if dtype == 'pair':

            model = LogisticRegression()

            for pair in variable_dict[year]:
                person = pair[0]
                partner = pair[1]
                
                if person not in embedding_dict or partner not in embedding_dict:
                    continue
                    
                if baseline is not None:
                    if person not in baseline or partner not in baseline:
                        continue
                
                embedding_1 = embedding_dict[person]
                embedding_2 = embedding_dict[partner]
                
                if baseline is not None:
                    baseline_values_1 = baseline[person]
                    baseline_values_2 = baseline[partner]
                    assert len(baseline_values_1) == len(baseline_values_2)
                    
                    embedding_1 += baseline_values_1
                    embedding_2 += baseline_values_2
                    assert len(embedding_1) == len(embedding_2)
                    
                    baseline_by_year[year].append(baseline_values_1 + baseline_values_2)
                    full_baseline_list.append(baseline_values_1 + baseline_values_2)
                
                embedding = np.concatenate((embedding_1, embedding_2), axis=0)
                
                label = variable_dict[year][pair]
                
                embeddings_by_year[year].append(embedding)
                labels_by_year[year].append(label)
                
                full_embedding_list.append(embedding)
                full_label_list.append(label)
    
    if len(full_embedding_list) < 5 or len(full_label_list) < 5:
        print("Not enough samples! Aborting", flush=True)
        return None, None, None
        
    scores = cross_val_score(model, full_embedding_list, full_label_list, cv=5)
    overall_r2 = scores.mean()
    return_dict['OVERALL'] = overall_r2
    
    if baseline is not None:
        if dtype == 'single':
            baseline_model = LinearRegression()
        elif dtype == 'pair':
            baseline_model = LogisticRegression()
            
        scores = cross_val_score(baseline_model, full_baseline_list, full_label_list, cv=5)
        baseline_overall = scores.mean()
        baseline_return_dict['OVERALL'] = baseline_overall
    
    # Now do cross validation for each year
    scores_by_year = {}
    baseline_scores_by_year = {}
    for year in years:
        scores_by_year[year] = []
        baseline_scores_by_year[year] = []
    
    for i in range(5):
    
        full_embedding_list = []
        full_label_list = []
        full_baseline_list = []
        
        test_embeddings_by_year = {}
        test_labels_by_year = {}
        test_baseline_by_year = {}
        
        for year in years:
            embeddings = embeddings_by_year[year]
            labels = labels_by_year[year]
            baselines = baseline_by_year[year]
            
            combined = list(zip(embeddings, labels))
            random.shuffle(combined)
            embeddings, labels = zip(*combined)
            
            embeddings = list(embeddings)
            labels = list(labels)
            
            split_point = int(len(embeddings) * 0.8)
            
            train_embeddings = embeddings[:split_point]
            test_embeddings = embeddings[split_point:]
            
            train_labels = labels[:split_point]
            test_labels = labels[split_point:]
            
            train_baseline = baselines[:split_point]
            test_baseline = baselines[split_point:]
            
            full_embedding_list += train_embeddings
            full_label_list += train_labels
            full_baseline_list += train_baseline
            
            test_embeddings_by_year[year] = test_embeddings
            test_labels_by_year[year] = test_labels
            test_baseline_by_year[year] = test_baseline
            
            test_counts_by_year[year] = len(test_labels)
            
            # Get the test counts for the last fold
            if i==4:
                test_counts_overall += len(test_labels)
            
        if dtype == 'single':
            model = LinearRegression()
            model.fit(full_embedding_list, full_label_list)
        elif dtype == 'pair':
            model = LogisticRegression()
            model.fit(full_embedding_list, full_label_list)
        
        if baseline is not None:
        
            if dtype == 'single':
                baseline_model = LinearRegression()
                baseline_model.fit(full_baseline_list, full_label_list)
            if dtype == 'pair':
                baseline_model = LogisticRegression()
                baseline_model.fit(full_baseline_list, full_label_list)
        
        for year in years:
            score = model.score(test_embeddings_by_year[year], test_labels_by_year[year])
            scores_by_year[year].append(score)
            
            if baseline is not None:
                score = baseline_model.score(test_baseline_by_year[year], test_labels_by_year[year])
                baseline_scores_by_year[year].append(score)

    for year in years:
        r2 = np.mean(scores_by_year[year])
        return_dict[year] = r2
        
        if baseline is not None:
            r2 = np.mean(baseline_scores_by_year[year])
            baseline_return_dict[year] = r2
        
    test_counts_by_year['OVERALL'] = test_counts_overall
    
    # Return more if baseline was provided
    if baseline is not None:
        return return_dict, test_counts_by_year, baseline_return_dict
    else:
        return return_dict, test_counts_by_year

########################################################################################################################
def weighted_footrule(list1, list2):
    # Compute a distance between two lists of equal length.
    # The returned PHI is between 0 and 1, where the lists are identical if
    # PHI = 0 and are completely disjoint if PHI = 1.
    # Author: Samuel D. Relton, June 2015.
    # Reference: Amy N. Langville and Carl D. Meyer, Who's #1? The Science of Rating and
    # Ranking, Princeton University Press, 2012. Page 209.

    k = len(list1)
    phi = 0
    denom = sum(1 / (1 + i) for i in range(k))
    x = (k - 4 * (k // 2) + 2 * (k + 1) * sum(1 / (1 + i) for i in range(k // 2))) / denom
    unique_items = list(set(list1) | set(list2))

    for item in unique_items:
        if item in list1 and item in list2:
            pos1 = list1.index(item) + 1
            pos2 = list2.index(item) + 1
            phi += abs(pos1 - pos2) / min(pos1, pos2)
        elif item in list1:
            pos1 = list1.index(item) + 1
            phi += abs(pos1 - x) / min(pos1, x)
        else:
            pos2 = list2.index(item) + 1
            phi += abs(pos2 - x) / min(pos2, x)

    phi /= (-2 * k + 2 * x * denom)
    return phi


def embedding_rank_comparison(embedding_dict_1, embedding_dict_2, top_k=50, methods=['intersection', 'spearman']):
    results = {}

    # Get the top k nearest neighbors for the first embedding set
    nn_dict_1 = get_nearest_neighbor_e2e(corpus_embs_dict=embedding_dict_1,
                                         query_embs_dict=embedding_dict_1,
                                         check_pids=None,
                                         top_k=top_k,
                                         ignore_self=True,
                                         return_index=False)

    nn_data_1 = nn_dict_1['result']

    # Get the top k nearest neighbors for the second embedding set
    nn_dict_2 = get_nearest_neighbor_e2e(corpus_embs_dict=embedding_dict_2,
                                         query_embs_dict=embedding_dict_2,
                                         check_pids=None,
                                         top_k=top_k,
                                         ignore_self=True,
                                         return_index=False)

    nn_data_2 = nn_dict_2['result']

    # Get the intersected user set
    user_set_1 = set(nn_data_1.keys())
    user_set_2 = set(nn_data_2.keys())
    user_set = user_set_1.intersection(user_set_2)

    if 'intersection' in methods:

        similarities = []

        for person in user_set:
            person_results_1 = nn_data_1[person]
            person_results_2 = nn_data_2[person]

            top_k_1 = set()
            for result in person_results_1:
                other = result['pid']
                top_k_1.add(other)

            top_k_2 = set()
            for result in person_results_2:
                other = result['pid']
                top_k_2.add(other)

            intersect = top_k_1.intersection(top_k_2)

            proportion_similar = len(intersect) / len(top_k_1)
            similarities.append(proportion_similar)

        results['intersection'] = np.mean(similarities)

    if 'spearman' in methods:

        similarities = []

        for person in user_set:
            person_results_1 = nn_data_1[person]
            person_results_2 = nn_data_2[person]

            top_k_1 = []
            for result in person_results_1:
                other = result['pid']
                top_k_1.append(other)

            top_k_2 = []
            for result in person_results_2:
                other = result['pid']
                top_k_2.append(other)

            score = weighted_footrule(top_k_1, top_k_2)

            similarities.append(score)

        results['spearman'] = np.mean(similarities)

    return results

########################################################################################################################

def get_distance(person, partner, distance_matrix, embedding_dict, person_embedding):
    distance = None
    if person in distance_matrix:
        if partner in distance_matrix[person]:
            distance = distance_matrix[person][partner]
            
    if partner in distance_matrix:
        if person in distance_matrix[partner]:
            distance = distance_matrix[partner][person]
            
    if distance is None:
        partner_embedding = embedding_dict[partner]
        distance = util.cos_sim(person_embedding, partner_embedding).numpy()[0][0]
        if person not in distance_matrix:
            distance_matrix[person] = {}
        distance_matrix[person][partner] = distance
            
    return distance
    
    #--------------------------------------------------------------------------------------------------------#

def get_marriage_rank_by_year(embedding_dict, distance_matrix, dtype="marriage"):

    if dtype == 'marriage':
        with open("data/processed/marriages_by_year.pkl", "rb") as pkl_file:
            marriage_data = dict(pickle.load(pkl_file))
            
    elif dtype == 'partnership':
        with open("data/processed/partnerships_by_year.pkl", "rb") as pkl_file:
            marriage_data = dict(pickle.load(pkl_file))

    full_user_set = set(embedding_dict.keys())
    #reduced_user_set = set(random.sample(list(embedding_dict.keys()), 1000000))
        
    with open("data/processed/id_to_gender_map.pkl", "rb") as pkl_file:
        gender_map = dict(pickle.load(pkl_file))
        
    with open("data/processed/full_male_list.pkl", "rb") as pkl_file:
        full_male_list = list(pickle.load(pkl_file))
        
    # Reduce the male list to only those we have embeddings for
    reduced_male_set = set(full_male_list).intersection(full_user_set)
    reduced_male_list = list(reduced_male_set)
    assert len(reduced_male_list) > 0
        
    with open("data/processed/full_female_list.pkl", "rb") as pkl_file:
        full_female_list = list(pickle.load(pkl_file))
        
    # Reduce the female list to only those we have embeddings for
    reduced_female_set = set(full_female_list).intersection(full_user_set)
    reduced_female_list = list(reduced_female_set)
    assert len(reduced_female_list) > 0
    
    yearly_rank_averages = {}
    yearly_test_counts = {}
    overall_ranks = []
    
    # Object for randomly sampling
    rng = np.random.default_rng()
    
    years = len(marriage_data)
    max_marriages_per_year = np.max([len(marriage_data[year]) for year in marriage_data])
    
    male_sample = rng.choice(reduced_male_list, size=(years, max_marriages_per_year, 101), replace=True)
    female_sample = rng.choice(reduced_female_list, size=(years, max_marriages_per_year, 101), replace=True)
    
    for year_idx, year in enumerate(marriage_data):
        
        partner_ranks = []
        
        yearly_data = marriage_data[year]
        yearly_test_counts[int(year)] = 0
        
        for person_idx, person in enumerate(yearly_data):
        
            if person not in full_user_set:
                continue
                
            # For each person take their real partner and 100 unique randos of the same gender
            partner = yearly_data[person]
            
            if partner not in full_user_set:
                continue
                
            person_embedding = embedding_dict[person]
            distance = get_distance(person, partner, distance_matrix, embedding_dict, person_embedding)
            
            partner_distance = distance
            partner_rank = 1
            
            gender = gender_map[partner]
            
            if gender == 1:
                sample = list(male_sample[year_idx][person_idx])
            if gender == 2:
                sample = list(female_sample[year_idx][person_idx])
                
            if person in sample:
                sample.remove(person)
            else:
                sample.pop()
                
            for other in sample:
                distance = get_distance(person, other, distance_matrix, embedding_dict, person_embedding)
                if distance > partner_distance:
                    partner_rank += 1
            
            yearly_test_counts[int(year)] += 1
            partner_ranks.append(partner_rank)
            overall_ranks.append(partner_rank)
            
        #print(partner_ranks, flush=True)
        yearly_rank_averages[int(year)] = np.mean(partner_ranks)
     
    yearly_rank_averages['OVERALL'] = np.mean(overall_ranks)
    yearly_test_counts['OVERALL'] = np.sum(list(yearly_test_counts.values()))
    
    return yearly_rank_averages, yearly_test_counts
    
########################################################################################################################################################################################

def yearly_probability_prediction(embedding_dict, variable_dict, years, baseline=None, baseline_descriptor=None):

    return_dict = {}

    embeddings_by_year = {}
    labels_by_year = {}
    
    full_embedding_list = []
    full_label_list = []

    model = LinearRegression()

    max_year = max(years)
    min_year = min(years)
    
    max_difference = np.abs(max_year - min_year)

    for year in years:

        if year not in embeddings_by_year:
        
            embeddings_by_year[year] = []
            labels_by_year[year] = []
        
            for person in variable_dict[year]:
            # Get the players with a valid variable for this year
                if person not in embedding_dict:
                    continue

                embedding = embedding_dict[person]
                value = variable_dict[year][person]
                
                yearly_difference = np.abs(year - min_year)
                normal_year = yearly_difference / max_difference
                embedding.append(normal_year)
            
                embeddings_by_year[year].append(embedding)
                labels_by_year[year].append(value)
                
                full_embedding_list.append(embedding)
                full_label_list.append(value)
    
    combined = list(zip(full_embedding_list, full_label_list))
    random.shuffle(combined)
    full_embedding_list, full_label_list = zip(*combined)
    full_embedding_list = list(full_embedding_list)
    full_label_list = list(full_label_list)
    
    split_point = int(len(full_label_list) * 0.8)
    
    training_embeddings = full_embedding_list[:split_point]
    test_embeddings = full_embedding_list[split_point:]
    
    training_labels = full_label_list[:split_point]
    test_labels = full_label_list[split_point:]
    
    model.fit(training_embeddings, training_labels)
    
    predictions = model.predict(test_embeddings)
    
    dead_probabilities = []
    alive_probabilities = []
    
    for i in range(len(test_labels)):
        if test_labels[i] == 1:
            alive_probabilities.append(predictions[i])
        elif test_labels[i] == 0:
            dead_probabilities.append(predictions[i])
            
    difference = np.mean(dead_probabilities) - np.mean(alive_probabilities)
    return_dict['OVERALL'] = difference
    
    # Now do separate tests for each year
    full_embedding_list = []
    full_label_list = []
    
    test_embeddings_per_year = {}
    test_labels_per_year = {}
    
    for year in years:
        #print(year, len(embeddings_by_year[year]), flush=True)
        embeddings = embeddings_by_year[year]
        labels = labels_by_year[year]
        
        combined = list(zip(embeddings, labels))
        random.shuffle(combined)
        embeddings, labels = zip(*combined)
        
        embeddings = list(embeddings)
        labels = list(labels)
        
        split_point = int(len(embeddings) * 0.8)
        
        train_embeddings = embeddings[:split_point]
        test_embeddings = embeddings[split_point:]
        
        train_labels = labels[:split_point]
        test_labels = labels[split_point:]
        
        full_embedding_list += train_embeddings
        full_label_list += train_labels
        
        test_embeddings_per_year[year] = test_embeddings
        test_labels_per_year[year] = test_labels
        #print(year, len(test_embeddings_per_year[year]), flush=True)
        
    model = LinearRegression()
    model.fit(full_embedding_list, full_label_list)
    
    for year in years:
        test_embeddings = test_embeddings_per_year[year]
        test_labels = test_labels_per_year[year]
    
        predictions = model.predict(test_embeddings)
    
        dead_probabilities = []
        alive_probabilities = []
    
        for i in range(len(test_labels)):
            if test_labels[i] == 1:
                dead_probabilities.append(predictions[i])
            elif test_labels[i] == 0:
                alive_probabilities.append(predictions[i])
            else:
                assert test_labels[i] in {0, 1}
            
        #print(year, len(dead_probabilities), len(alive_probabilities), flush=True)
        #print(alive_probabilities, flush=True)
        
        if len(dead_probabilities) == 0:
            difference = 'NO'
        else:
            difference = np.mean(dead_probabilities) - np.mean(alive_probabilities)
        return_dict[year] = difference

    return return_dict

########################################################################################################################

def print_output_table(pdf, years, results, highlight=True, reverse=False):

    x_offset = 10
    y_offset = 10

    pdf.set_font('Arial', '', 8)

    plot_height = 100
    header_height = 20

    # A4 dimensions in mm
    max_width = 210
    max_height = 297

    # 1. Reformat data into rows
    data = []
    max_indices = []
    min_indices = []

    years.sort()

    for i, year in enumerate(years):

        max_value = 0.0
        max_index = None
        
        min_value = np.inf
        min_index = None
        
        row = [year]

        for j, emb_type in enumerate(results):

            value = results[emb_type][year]
            

            if not isinstance(value, str):

                row.append(str(np.round(value, decimals=3)))

                if value > max_value:
                    max_value = value
                    max_index = j + 1
                    
                if value < min_value:
                    min_value = value
                    min_index = j + 1
                    
            else:
                row.append(value)

        data.append(row)
        max_indices.append(max_index)
        min_indices.append(min_index)

    overall_row = ['OVERALL']
    for emb_type in results:
        overall_row.append(np.round(results[emb_type]['OVERALL'], decimals=3))

    data.append(overall_row)
    max_indices.append(np.argmax(overall_row[1:]) + 1)
    min_indices.append(np.argmin(overall_row[1:]) + 1)

    # 2. Print the table
    line_height = pdf.font_size * 2.5
    epw = max_width - 20
    col_width = epw / (len(results) + 1)

    header_row = [" "] + list(results.keys())

    # Print header row
    pdf.set_font('', style='BI')
    for datum in header_row:
        pdf.cell(w=col_width, h=line_height, txt=datum, border=1, align='C')
    pdf.ln(line_height)

    # Print other rows
    for i, row in enumerate(data):
        #print(row, flush=True)

        for j, datum in enumerate(row):
            if j == 0:
                pdf.set_font('', style='BI')
            elif highlight and not reverse and j == max_indices[i]:
                pdf.set_font('', style='B')
            elif highlight and reverse and j == min_indices[i]:
                pdf.set_font('', style='B')
            else:
                pdf.set_font('', style='')
            pdf.cell(w=col_width, h=line_height, txt=str(datum), border=1, align='C')

        pdf.ln(line_height)
        
    return pdf
   
#########################################################################################################################################################
   
def linear_baseline_prediction(embedding_dict, variable_dict, years, baseline, dtype="single"):

    return_dict = {}
    baseline_return_dict = {}

    embeddings_by_year = {}
    baseline_by_year = {}
    labels_by_year = {}
    test_counts_by_year = {}
    
    full_baseline_list = []
    full_embedding_list = []
    full_label_list = []

    test_counts_overall = 0

    model = LinearRegression()

    for year in years:
    
        test_counts_by_year[year] = 0

        if year not in embeddings_by_year:
            embeddings_by_year[year] = []
            labels_by_year[year] = []
            baseline_by_year[year] = []
        
        if dtype == "single":
            for person in variable_dict[year]:
            # Get the players with a valid variable for this year
                if person not in embedding_dict:
                    continue
                    
                if person not in baseline:
                    continue

                embedding = embedding_dict[person]
                value = variable_dict[year][person]
            
                baseline_values = baseline[person]
                expanded_embedding = embedding + baseline_values
                    
                baseline_by_year[year].append(baseline[person])
                full_baseline_list.append(baseline[person])
            
                embeddings_by_year[year].append(expanded_embedding)
                labels_by_year[year].append(value)
                
                full_embedding_list.append(expanded_embedding)
                full_label_list.append(value)
                
        if dtype == 'pair':

            for pair in variable_dict[year]:
                person = pair[0]
                partner = pair[1]
                
                if person not in embedding_dict or partner not in embedding_dict:
                    continue
                    
                if person not in baseline or partner not in baseline:
                    continue
                
                embedding_1 = embedding_dict[person]
                embedding_2 = embedding_dict[partner]
                
                baseline_values_1 = baseline[person]
                baseline_values_2 = baseline[partner]
                assert len(baseline_values_1) == len(baseline_values_2)
                
                extended_embedding_1 = embedding_1 + baseline_values_1
                extended_embedding_2 = embedding_2 + baseline_values_2
                assert len(extended_embedding_1) == len(extended_embedding_2)
                
                baseline_by_year[year].append(baseline_values_1 + baseline_values_2)
                full_baseline_list.append(baseline_values_1 + baseline_values_2)
                
                combined_embedding = np.concatenate((extended_embedding_1, extended_embedding_2))
                
                label = variable_dict[year][pair]
                
                embeddings_by_year[year].append(combined_embedding)
                labels_by_year[year].append(label)
                
                full_embedding_list.append(combined_embedding)
                full_label_list.append(label)
    
    if len(full_embedding_list) < 5 or len(full_label_list) < 5:
        print("Not enough samples! Aborting", flush=True)
        return None, None, None
        
    scores = cross_val_score(model, full_embedding_list, full_label_list, cv=5)
    overall_r2 = scores.mean()
    return_dict['OVERALL'] = overall_r2
    
    baseline_model = LinearRegression()
    scores = cross_val_score(baseline_model, full_baseline_list, full_label_list, cv=5)
    baseline_overall = scores.mean()
    baseline_return_dict['OVERALL'] = baseline_overall
    
    # Now do cross validation for each year
    scores_by_year = {}
    baseline_scores_by_year = {}
    for year in years:
        scores_by_year[year] = []
        baseline_scores_by_year[year] = []
    
    for i in range(5):
    
        full_embedding_list = []
        full_label_list = []
        full_baseline_list = []
        
        test_embeddings_by_year = {}
        test_labels_by_year = {}
        test_baseline_by_year = {}
        
        for year in years:
            embeddings = embeddings_by_year[year]
            labels = labels_by_year[year]
            baselines = baseline_by_year[year]
            
            combined = list(zip(embeddings, labels))
            random.shuffle(combined)
            embeddings, labels = zip(*combined)
            
            embeddings = list(embeddings)
            labels = list(labels)
            
            split_point = int(len(embeddings) * 0.8)
            
            train_embeddings = embeddings[:split_point]
            test_embeddings = embeddings[split_point:]
            
            train_labels = labels[:split_point]
            test_labels = labels[split_point:]
            
            train_baseline = baselines[:split_point]
            test_baseline = baselines[split_point:]
            
            full_embedding_list += train_embeddings
            full_label_list += train_labels
            full_baseline_list += train_baseline
            
            test_embeddings_by_year[year] = test_embeddings
            test_labels_by_year[year] = test_labels
            test_baseline_by_year[year] = test_baseline
            
            test_counts_by_year[year] = len(test_labels)
            
            # Get the test counts for the last fold
            if i==4:
                test_counts_overall += len(test_labels)
            
        model = LinearRegression()
        model.fit(full_embedding_list, full_label_list)
        
        if baseline is not None:
            baseline_model = LinearRegression()
            baseline_model.fit(full_baseline_list, full_label_list)
        
        for year in years:
            score = model.score(test_embeddings_by_year[year], test_labels_by_year[year])
            scores_by_year[year].append(score)
            
            if baseline is not None:
                score = baseline_model.score(test_baseline_by_year[year], test_labels_by_year[year])
                baseline_scores_by_year[year].append(score)

    for year in years:
        r2 = np.mean(scores_by_year[year])
        return_dict[year] = r2
        
        if baseline is not None:
            r2 = np.mean(baseline_scores_by_year[year])
            baseline_return_dict[year] = r2
        
    test_counts_by_year['OVERALL'] = test_counts_overall
    
    return return_dict, test_counts_by_year, baseline_return_dict

########################################################################################################################