import random
import csv
import sys
import argparse
import pickle
import time


feature_codes = [100, 101, 102,
                 200, 201,
                 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322,
                 400, 401, 402,
                 500, 501, 502, 503, 504, 505, 506
]

###########################################################################################

def process_network_file(in_path, lower, upper):

    relation_counts = {}

    start_row = 1

    with open(in_path, newline="\n") as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        i = 0
        for j, row in enumerate(reader):
            if i < start_row:
                i += 1
                continue

            source = int(row[1])
            relation = int(row[5])

            if source not in relation_counts:
                relation_counts[source] = {}
                for k in range(lower, upper+1):
                    relation_counts[source][k] = 0

            if relation < (lower + 1) or relation > upper:
                relation_counts[source][lower] += 1
            else:
                relation_counts[source][relation] += 1

    return relation_counts

######################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NodeFeatures")
    parser.add_argument(
        "--year",
        type=int,
        default=2016
    )
    args = parser.parse_args()
    year = str(args.year)
    
    full_start = time.time()

    root = "/gpfs/ostor/ossc9424/data/"
    
    #----------------------------------------------------------------------------------------------------------#
    
    section_start = time.time()
    # Neighbors
    neighbor_path = root + "BURENNETWERK" + year + "TABV1.csv"

    neighbor_counts = process_network_file(neighbor_path, 100, 102)
    
    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Calculated neighbor connection counts in", str(delta), "minutes", flush=True)
    
    #----------------------------------------------------------------------------------------------------------#

    section_start = time.time()
    # Colleagues
    colleague_path = root + "COLLEGANETWERK" + year + "TABV1.csv"

    colleague_counts = process_network_file(colleague_path, 200, 201)

    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Calculated colleague connection counts in", str(delta), "minutes", flush=True)

    # ----------------------------------------------------------------------------------------------------------#

    section_start = time.time()

    family_path = root + "cbs_data/Bevolking/FAMILIENETWERK" + year + "TABV1.csv"

    family_counts = process_network_file(family_path, 300, 322)

    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Calculated family connection counts in", str(delta), "minutes", flush=True)

    # ----------------------------------------------------------------------------------------------------------#
    
    section_start = time.time()
    # Householders
    household_path = root + "HUISGENOTENNETWERK" + year + "TABV1.csv"

    household_counts = process_network_file(household_path, 400, 402)
        
    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Calculated household connection counts in", str(delta), "minutes", flush=True)
    
    #----------------------------------------------------------------------------------------------------------#

    section_start = time.time()
    # Classmates
    classmate_path = root + "KLASGENOTENNETWERK" + year + "TABV1.csv"

    classmate_counts = process_network_file(classmate_path, 500, 506)

    section_end = time.time()
    delta = (section_end - section_start) / 60.
    print("Calculated classmate connection counts in", str(delta), "minutes", flush=True)

    # ----------------------------------------------------------------------------------------------------------#

    section_start = time.time()
    # Combine all the counts and write to a csv

    all_persons = set(family_counts.keys()).union(set(neighbor_counts.keys())).union(set(colleague_counts.keys())).union(set(household_counts.keys())).union(set(classmate_counts.keys()))

    # Write as a stream to avoid high memory loads
    with open(root + "graph/node_features/node_features_" + str(year) + ".csv", 'w', newline="\n") as out_csvfile:
        writer = csv.writer(out_csvfile, delimiter=' ')

    for person in all_persons:

        # Define the person row starting with RINPERSOON
        person_row = [person]

        ###############################################################################

        if person in neighbor_counts:
            person_neighbor_counts = neighbor_counts[person]
            person_row.append(person_neighbor_counts[100])
            person_row.append(person_neighbor_counts[101])
            person_row.append(person_neighbor_counts[102])

        else:
            person_row += [-1, -1, -1]

        ###############################################################################

        if person in colleague_counts:
            person_colleague_counts = colleague_counts[person]
            person_row.append(person_colleague_counts[200])
            person_row.append(person_colleague_counts[201])

        else:
            person_row += [-1, -1]

        ###############################################################################

        if person in family_counts:

            person_family_counts = family_counts[person]
            for i in range(300, 323):
                person_row.append(person_family_counts[i])

        else:
            for i in range(300, 323):
                person_row.append(-1)

        ################################################################################3

        if person in household_counts:
            person_household_counts = household_counts[person]
            person_row.append(person_household_counts[400])
            person_row.append(person_household_counts[401])
            person_row.append(person_household_counts[402])

        else:
            person_row += [-1, -1, -1]

        ###################################################################################

        if person in classmate_counts:
            person_classmate_counts = classmate_counts[person]
            for i in range(500, 507):
                person_row.append(person_classmate_counts[i])

        else:
            for i in range(500, 507):
                person_row.append(-1)

        writer.writerow(person_row)

    full_end = time.time()
    delta = (full_end - full_start) / 60.
    print("Finished calculating node features for year", year, flush=True)
    print("The process took", str(delta/60.), "hours", flush=True)
