import numpy as np
import pickle
import csv

# Assume we have 5 edge dictionaries indexed by RINPERSOON
# family_edges = {}
# household_edges = {}
# colleague_edges = {}
# neighbor_edges = {}
# education_edges = {}

root = '/gpfs/ostor/ossc9424/homedir/Dakota_network/intermediates/'

with open(root + "family_2010_adjacency_dict.pkl", 'rb') as pkl_file:
    family_edges = pickle.load(pkl_file)

with open(root + "household_2010_adjacency_dict.pkl", 'rb') as pkl_file:
    household_edges = pickle.load(pkl_file)

with open(root + "neighbor_2010_adjacency_dict.pkl", 'rb') as pkl_file:
    neighbor_edges = pickle.load(pkl_file)

with open(root + "colleague_2010_adjacency_dict.pkl", 'rb') as pkl_file:
    colleague_edges = pickle.load(pkl_file)

with open(root + "classmate_2010_adjacency_dict.pkl", 'rb') as pkl_file:
    education_edges = pickle.load(pkl_file)

num_family_connections = []
num_household_connections = []
num_colleague_connections = []
num_neighbor_connections = []
num_education_connections = []

edge_types = ['family', 'household', 'colleague', 'neighbor', 'education']

family_degree = []
household_degree = []
neighbor_degree = []
colleague_degree = []
education_degree = []

family_household_intersections = []
family_neighbor_intersections = []
family_colleague_intersections = []
family_education_intersections = []

household_neighbor_intersections = []
household_colleague_intersections = []
household_education_intersections = []

neighbor_colleague_intersections = []
neighbor_education_intersections = []

colleague_education_intersections = []

# Use family as the canonical layer for person inclusion
for person in family_edges:

    # Get family connections
    if person in family_edges:
        family_connections = set(family_edges[person])
    else:
        family_connections = set()

    # Get household connections if applicable
    if person in household_edges:
        household_connections = set(household_edges[person])
    else:
        household_connections = set()

    # Get neighbor connections if applicable
    if person in neighbor_edges:
        neighbor_connections = set(neighbor_edges[person])
    else:
        neighbor_connections = set()

    # Get colleague connections if applicable
    if person in colleague_edges:
        colleage_connections = set(colleague_edges[person])
    else:
        colleage_connections = set()

    # Get education connections if applicable
    if person in education_edges:
        education_connections = set(education_edges[person])
    else:
        education_connections = set()

    # Append the degree of each layer for this person
    family_degree.append(len(family_connections))
    household_degree.append(len(household_connections))
    neighbor_degree.append(len(neighbor_connections))
    colleague_degree.append(len(colleage_connections))
    education_degree.append(len(education_connections))

    family_household_intersection = len(family_connections.intersection(household_connections))
    family_neighbor_intersection = len(family_connections.intersection(neighbor_connections))
    family_colleague_intersection = len(family_connections.intersection(colleage_connections))
    family_education_intersection = len(family_connections.intersection(education_connections))
    # Append
    family_household_intersections.append(family_household_intersection)
    family_neighbor_intersections.append(family_neighbor_intersection)
    family_colleague_intersections.append(family_colleague_intersection)
    family_education_intersections.append(family_education_intersection)

    household_neighbor_intersection = len(household_connections.intersection(neighbor_connections))
    household_colleague_intersection = len(household_connections.intersection(colleage_connections))
    household_education_intersection = len(household_connections.intersection(education_connections))
    # Append
    household_neighbor_intersections.append(household_neighbor_intersection)
    household_colleague_intersections.append(household_colleague_intersection)
    household_education_intersections.append(household_education_intersection)

    neighbor_colleague_intersection = len(neighbor_connections.intersection(colleage_connections))
    neighbor_education_intersection = len(neighbor_connections.intersection(education_connections))
    # Append
    neighbor_colleague_intersections.append(neighbor_colleague_intersection)
    neighbor_education_intersections.append(neighbor_education_intersection)

    colleague_education_intersection = len(colleage_connections.intersection(education_connections))
    # Append
    colleague_education_intersections.append(colleague_education_intersection)

# Average across collections, get mean and std

print("Computing summary statistics...", flush=True)

save_data = {}

# Family stuff
print("Family degree N:", len(family_degree), flush=True)
save_data['family'] = {}
tenth, _, _, _, ninetieth = np.nanpercentile(family_degree, [10, 25, 50, 75, 90])
save_data['family']['degree_mean'] = np.mean(family_degree)
save_data['family']['degree_median'] = np.median(family_degree)
save_data['family']['degree_std'] = np.std(family_degree)
save_data['family']['degree_10th'] = tenth
save_data['family']['degree_90th'] = ninetieth
save_data['family']['degree_nonzero'] = np.count_nonzero(family_degree) / len(family_degree)
save_data['family']['N'] = len(family_degree)

# Household stuff
print("Household degree N:", len(household_degree), flush=True)
save_data['household'] = {}
tenth, _, _, _, ninetieth = np.nanpercentile(household_degree, [10, 25, 50, 75, 90])
save_data['household']['degree_mean'] = np.mean(household_degree)
save_data['household']['degree_median'] = np.median(household_degree)
save_data['household']['degree_std'] = np.std(household_degree)
save_data['household']['degree_10th'] = tenth
save_data['household']['degree_90th'] = ninetieth
save_data['household']['degree_nonzero'] = np.count_nonzero(household_degree) / len(household_degree)
save_data['household']['N'] = len(household_degree)

# Neighbor stuff
print("Neighbor degree N:", len(neighbor_degree), flush=True)
save_data['neighbor'] = {}
tenth, _, _, _, ninetieth = np.nanpercentile(neighbor_degree, [10, 25, 50, 75, 90])
save_data['neighbor']['degree_mean'] = np.mean(neighbor_degree)
save_data['neighbor']['degree_median'] = np.median(neighbor_degree)
save_data['neighbor']['degree_std'] = np.std(neighbor_degree)
save_data['neighbor']['degree_10th'] = tenth
save_data['neighbor']['degree_90th'] = ninetieth
save_data['neighbor']['degree_nonzero'] = np.count_nonzero(neighbor_degree) / len(neighbor_degree)
save_data['neighbor']['N'] = len(neighbor_degree)

# Colleague stuff
print("Colleague degree N:", len(colleague_degree), flush=True)
save_data['colleague'] = {}
tenth, _, _, _, ninetieth = np.nanpercentile(colleague_degree, [10, 25, 50, 75, 90])
save_data['colleague']['degree_mean'] = np.mean(colleague_degree)
save_data['colleague']['degree_median'] = np.median(colleague_degree)
save_data['colleague']['degree_std'] = np.std(colleague_degree)
save_data['colleague']['degree_10th'] = tenth
save_data['colleague']['degree_90th'] = ninetieth
save_data['colleague']['degree_nonzero'] = np.count_nonzero(colleague_degree) / len(colleague_degree)
save_data['colleague']['N'] = len(colleague_degree)

# Education stuff
print("Education degree N:", len(education_degree), flush=True)
save_data['education'] = {}
tenth, _, _, _, ninetieth = np.nanpercentile(education_degree, [10, 25, 50, 75, 90])
save_data['education']['degree_mean'] = np.mean(education_degree)
save_data['education']['degree_median'] = np.median(education_degree)
save_data['education']['degree_std'] = np.std(education_degree)
save_data['education']['degree_10th'] = tenth
save_data['education']['degree_90th'] = ninetieth
save_data['education']['degree_nonzero'] = np.count_nonzero(education_degree) / len(education_degree)
save_data['education']['N'] = len(education_degree)

# Intersection stuff
save_data['intersection'] = {}
tenth, _, _, _, ninetieth = np.nanpercentile(family_household_intersections, [10, 25, 50, 75, 90])
save_data['intersection']['family_household_mean'] = np.mean(family_household_intersections)
save_data['intersection']['family_household_median'] = np.median(family_household_intersections)
save_data['intersection']['family_household_std'] = np.std(family_household_intersections)
save_data['intersection']['family_household_10th'] = tenth
save_data['intersection']['family_household_90th'] = ninetieth
save_data['intersection']['family_household_nonzero'] = np.count_nonzero(family_household_intersections) / len(family_household_intersections)
save_data['intersection']['N'] = len(family_household_intersections)

tenth, _, _, _, ninetieth = np.nanpercentile(family_neighbor_intersections, [10, 25, 50, 75, 90])
save_data['intersection']['family_neighbor_mean'] = np.mean(family_neighbor_intersections)
save_data['intersection']['family_neighbor_median'] = np.median(family_neighbor_intersections)
save_data['intersection']['family_neighbor_std'] = np.std(family_neighbor_intersections)
save_data['intersection']['family_neighbor_10th'] = tenth
save_data['intersection']['family_neighbor_90th'] = ninetieth
save_data['intersection']['family_neighbor_nonzero'] = np.count_nonzero(family_neighbor_intersections) / len(family_neighbor_intersections)

tenth, _, _, _, ninetieth = np.nanpercentile(family_colleague_intersections, [10, 25, 50, 75, 90])
save_data['intersection']['family_colleague_mean'] = np.mean(family_colleague_intersections)
save_data['intersection']['family_colleague_median'] = np.median(family_colleague_intersections)
save_data['intersection']['family_colleague_std'] = np.std(family_colleague_intersections)
save_data['intersection']['family_colleague_10th'] = tenth
save_data['intersection']['family_colleague_90th'] = ninetieth
save_data['intersection']['family_colleague_nonzero'] = np.count_nonzero(family_colleague_intersections) / len(family_colleague_intersections)

tenth, _, _, _, ninetieth = np.nanpercentile(family_education_intersections, [10, 25, 50, 75, 90])
save_data['intersection']['family_education_mean'] = np.mean(family_education_intersections)
save_data['intersection']['family_education_median'] = np.median(family_education_intersections)
save_data['intersection']['family_education_std'] = np.std(family_education_intersections)
save_data['intersection']['family_education_10th'] = tenth
save_data['intersection']['family_education_90th'] = ninetieth
save_data['intersection']['family_education_nonzero'] = np.count_nonzero(family_education_intersections) / len(family_education_intersections)

tenth, _, _, _, ninetieth = np.nanpercentile(household_neighbor_intersections, [10, 25, 50, 75, 90])
save_data['intersection']['household_neighbor_mean'] = np.mean(household_neighbor_intersections)
save_data['intersection']['household_neighbor_median'] = np.median(household_neighbor_intersections)
save_data['intersection']['household_neighbor_std'] = np.std(household_neighbor_intersections)
save_data['intersection']['household_neighbor_10th'] = tenth
save_data['intersection']['household_neighbor_90th'] = ninetieth
save_data['intersection']['household_neighbor_nonzero'] = np.count_nonzero(household_neighbor_intersections) / len(household_neighbor_intersections)

tenth, _, _, _, ninetieth = np.nanpercentile(household_colleague_intersections, [10, 25, 50, 75, 90])
save_data['intersection']['household_colleague_mean'] = np.mean(household_colleague_intersections)
save_data['intersection']['household_colleague_median'] = np.median(household_colleague_intersections)
save_data['intersection']['household_colleague_std'] = np.std(household_colleague_intersections)
save_data['intersection']['household_colleague_10th'] = tenth
save_data['intersection']['household_colleague_90th'] = ninetieth
save_data['intersection']['household_colleague_nonzero'] = np.count_nonzero(household_colleague_intersections) / len(household_colleague_intersections)

tenth, _, _, _, ninetieth = np.nanpercentile(household_education_intersections, [10, 25, 50, 75, 90])
save_data['intersection']['household_education_mean'] = np.mean(household_education_intersections)
save_data['intersection']['household_education_median'] = np.median(household_education_intersections)
save_data['intersection']['household_education_std'] = np.std(household_education_intersections)
save_data['intersection']['household_education_10th'] = tenth
save_data['intersection']['household_education_90th'] = ninetieth
save_data['intersection']['household_education_nonzero'] = np.count_nonzero(household_education_intersections) / len(household_education_intersections)

tenth, _, _, _, ninetieth = np.nanpercentile(neighbor_colleague_intersections, [10, 25, 50, 75, 90])
save_data['intersection']['neighbor_colleague_mean'] = np.mean(neighbor_colleague_intersections)
save_data['intersection']['neighbor_colleague_median'] = np.median(neighbor_colleague_intersections)
save_data['intersection']['neighbor_colleague_std'] = np.std(neighbor_colleague_intersections)
save_data['intersection']['neighbor_colleague_10th'] = tenth
save_data['intersection']['neighbor_colleague_90th'] = ninetieth
save_data['intersection']['neighbor_colleague_nonzero'] = np.count_nonzero(neighbor_colleague_intersections) / len(neighbor_colleague_intersections)

tenth, _, _, _, ninetieth = np.nanpercentile(neighbor_education_intersections, [10, 25, 50, 75, 90])
save_data['intersection']['neighbor_education_mean'] = np.mean(neighbor_education_intersections)
save_data['intersection']['neighbor_education_median'] = np.median(neighbor_education_intersections)
save_data['intersection']['neighbor_education_std'] = np.std(neighbor_education_intersections)
save_data['intersection']['neighbor_education_10th'] = tenth
save_data['intersection']['neighbor_education_90th'] = ninetieth
save_data['intersection']['neighbor_education_nonzero'] = np.count_nonzero(neighbor_education_intersections) / len(neighbor_education_intersections)

tenth, _, _, _, ninetieth = np.nanpercentile(colleague_education_intersections, [10, 25, 50, 75, 90])
save_data['intersection']['colleague_education_mean'] = np.mean(colleague_education_intersections)
save_data['intersection']['colleague_education_median'] = np.median(colleague_education_intersections)
save_data['intersection']['colleague_education_std'] = np.std(colleague_education_intersections)
save_data['intersection']['colleague_education_10th'] = tenth
save_data['intersection']['colleague_education_90th'] = ninetieth
save_data['intersection']['colleague_education_nonzero'] = np.count_nonzero(colleague_education_intersections) / len(colleague_education_intersections)

# Save everything as csv table
columns = ['LAYER_TYPE', 'MEAN', 'MEDIAN', 'STD', '10TH', '90TH', 'NONZERO', 'N']
intersection_types = ['family_household_', 'family_neighbor_', 'family_colleague_', 'family_education_', 'household_neighbor_',
                      'household_colleague_', 'household_education_', 'neighbor_colleague_', 'neighbor_education_', 'colleague_education_']

with open('/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/network_summary_statistics.csv', 'w', newline='', encoding='utf-8') as csvfile:
    # Open the writer and write the header row
    writer = csv.writer(csvfile, delimiter=',', dialect='excel')
    writer.writerow(columns)

    save_rows = []

    for layer_type in save_data:

        relevant = save_data[layer_type]

        if layer_type == 'intersection':
            for intersection in intersection_types:
                row = []
                label = layer_type + "_" + intersection[:-1]
                row.append(label)

                row.append(relevant[intersection + 'mean'])
                row.append(relevant[intersection + 'median'])
                row.append(relevant[intersection + 'std'])
                row.append(relevant[intersection + '10th'])
                row.append(relevant[intersection + '90th'])
                row.append(relevant[intersection + 'nonzero'])
                row.append(relevant['N'])

                save_rows.append(row)

        else:
            row = []
            # First add label
            row.append(layer_type.upper())
            row.append(relevant['degree_mean'])
            row.append(relevant['degree_median'])
            row.append(relevant['degree_std'])
            row.append(relevant['degree_10th'])
            row.append(relevant['degree_90th'])
            row.append(relevant['degree_nonzero'])
            row.append(relevant['N'])

            save_rows.append(row)

    writer.writerows(save_rows)

# with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/network_summary_statistics.pkl", 'wb') as pkl_file:
#     pickle.dump(save_data, pkl_file)