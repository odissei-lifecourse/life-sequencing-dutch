### CBS Data (Step 0)

sav format

These are raw CBS files that we get access to. Every sav file has a data component and a metadata component.

The data files we currently use are:

1. INPATAB (related to income tax return)
    1. yearly
    2. more than 100 columns, many of which are correlated with each other or unnecessary for many people.
2. SPOLISBUS (related to jobs, timeline data)
    1. x did job y from date a to b.
    2. additional columns about the job.
3. household (timeline data)
    1. x lived at household y where the head of household was z from date a to b.
    2. additional columns about the household.
4. background (birth date, sex, municipality of birth)
5. Education data
    1. Different levels of educational degrees in hierarchical structure a person obtains at different points of their lives.
6. Network data
    1. family
    2. household: people who lived in the same household. An unregistered couple may not be neighbors in family layer, but will be neighbors in the household layer.
    3. colleague: people who work for the same employer. An interesting phenomenon arises where the CEO of a company and the lowest-ranked employee will be connected as well as thousands of employees of a national chain will be connected too.
    4. school: people who went to the same school will be connected.
    5. neighbors: neighbors in real life will be connected.

### Records with metadata (Step 1)

#### Data-driven-structure

a.parquet

a_meta.parquet

From every raw CBS file, two files are generated. E.g., for a file named “a.sav”, we generate a.parquet containing all the tabular data, and a_meta.parquet containing the metadata.

a.parquet has the following characteristics:

1. A row represents one event at one time. For example, a job from 2007 to 2011 is not one record. A row must be tied to one point in time, to one (or two) persons. So a row can represent the start of a new job in 2007 and another row can represent the end of a job in 2011.
2. It has the column RINPERSOON
3. It has the column daysSinceFirst
    1. We have our own “genesis time” : 30 December 1971. This column has the time for the corresponding event, which is expressed as the # of days since genesis time.
4. It has the column age
    1. The age of the person identified by RINPERSOON at the time of the event in years.
5. It can have a second column named RINPERSOON2 identifying the 2nd person related to the event.
    1. For example, in a marriage event, there are two persons.
6. It does not have any other id columns, for example: household id, or other RINPERSOON ids.
7. It contains all other columns from the original data representing real information, instead of ids.
8. If a file contains no other columns than RINPERSOON, daysSinceFirst, age, then an additional column is added with a constant value. For example, in case of death records, we only have the RINPERSOON and time. We add an additional column named “Death” with the constant value “\[D\]” to it. This is necessary for the next steps of the model. Without that identifier column, the model cannot know what kind of event happened in this record since we do not use the file names for anything. We only use the column names and the values in the next steps.

For creating daysSinceFirst and age, one has to find out the column signifying time of event for a file, map the RINPERSOON to the background file to get the birth year of the person, and then compute those two values.

a_meta.parquet describes the metadata of the columns in a.parquet and contains “x” rows where the number of columns in a.parquet is “x”

a_meta.parquet has the following columns:

1. Name: name of a column in a.parquet
2. Type: Either “Numeric” or “String” . All categorical variables should have “String” as type.
3. ValueLabels: a dictionary where key is a special value in that column and value is the natural language explanation of that special value.

For example, let’s say INPATAB is a file containing the INPBELI column in the previous step. Then INPATAB_meta.parquet may have a row like this:

| Name | Type | ValueLabels |
| --- | --- | --- |
| INPBELI | Numeric | {9999999999: "belongs to a household with missing income data",<br><br>8888888888: "ineligible for income tax return"} |
| --- | --- | --- |

The ValueLabels are mostly directly copy-pasted from the metadata part of the sav files from the previous step. We get rid of the special characters from the natural language description.

#### HandPicked-structure

For some files, our in-house sociologists Dr. Lucas sage and Dr. Ana Macanovic did some manual transformations to make the data better. Lucas created CSVs and Ana created parquets. The CSVs follow the exact format of “a.parquet” discussed before.

Files that went through handpicking are:

1. Education Data
    1. Merging multiple types of educational degrees of the same level into the same category. For example: \[[ana.macanovic@eui.eu](mailto:ana.macanovic@eui.eu) please fill this up\]
2. timeline data:
    1. household and job data contain one record per household or per job with a start time and end time. They are broken into two separate events with all the same column values, except for one new column we add \[beg_or_end\] which have values “beg” or “end”
3. Network data:
    1. We currently have the number of direct neighbors in each of the 5 layers of network we have at the beginning of the year for each person.
4. ?

#### background-file-structure

It must contain the following columns only:

1. RINPERSOON
2. birth_year
3. birth_month
4. gender
5. origin

### Keep Column Subsets (Step 2)

For the data-driven approach, we initially planned to use all the columns in the data. This creates an issue where 5 years of income tax return can consume the whole context window (512 tokens) of a person since each income tax record has > 100 columns. We should create a more sophisticated method at this step to keep only the column values that are meaningful for a person. For now, we decided on utilizing a subset of columns handpicked by Ana. This means every record of the same type has the same set of columns/attributes. The ideal scenario should be where each record decides which column is necessary for them which we leave for future work.

During the column subsetting, it is ensured that no file has more than 15 columns. The number 15 comes from the idea that in English, a sentence usually has at most 15-20 words and in our life-sequences,

### Categorical transformation(Step 3)

take the data from step2 as input and produces step3 data in parquet format

pop2vec/llm/src/new_code/preprocess.py

pop2vec/llm/slurm_scripts/preprocess.sh

We transform all columns to categorical columns with at most 100 categories + a few special categories. This limits the size of the vocabulary.

1. For Numeric columns, we convert the column into percentiles
2. For categorical columns, we keep the top 100 categories and replace the rest with “Others”.
3. For all columns, we replace the special values with their natural language explanation. If a column has more than 100 special values, we keep the top 100 special value.
4. For numeric columns, we replace the 0 with a special value called SPECIAL_NUMERIC_ZERO. Otherwise the 0 value can imbalance the distribution of percentiles for specific columns.
5. SPECIAL_STR_ZERO for the String columns.
6. In the end, a column can have at most ~200 distinct values.

### List of Sequences (Step 4)

take the data from step3 as input and produces step4 data in parquet format. one file is created: people.parquet

pop2vec/llm/src/new_code/create_life_seq_parquets.py

pop2vec/llm/slurm_scripts/create_parquet_seq.sh

Next we combine all the parquet files from the previous step, and create a combined parquet file. Each row in this parquet represents one life-sequence. It expects exactly 1 background file and the other event files. It contains the following columns:

1. RINPERSOON
2. background: a dictionary with the following keys:
    1. birth_year
    2. birth_month
    3. gender
    4. origin
3. sentence: a list of events. Each event is a list of strings. Each string is a token representing an attribute of the event. For the value of 92 for the column INPBELI, we will create a token named “INPBELI_92”
4. abspos: a list of absolute time of an event derived from DaysSinceFirst. abspos\[i\] denotes the number of days passed since genesis time when the event sentence\[i\] took place.
5. age: age of the person in years when the event sentence\[i\] took place.
6. segment: if the events sentence\[i-1\] and sentence\[i\] took place on the same day, then segment\[i\] = 1, otherwise segment\[i\] = 2

This part of the code does not utilize parallel processing and takes around 30 minutes for the entire population. This can be made parallel and can be done much faster.

We store the parquet using multiple (66) row groups. Each row group has the same size.

Having too many row groups (smaller group size) can mess up the parquet file structure due to which pandas won’t be able to read the parquet file.

### Training data(Step 5)

take the data from step4 as input and produces step5 data in hdf5 format. multiple files are created:

1. the vocab file in csv format
2. one or more .h5 files (based on the number of sequences we are processing)
3. multiple .h5 files can be merged into h5 file using the code pop2vec/llm/src/new_code/merge_hdf5.py \[NOTE THAT THIS IS NOT DONE IN THE SLURM SCRIPT RIGHT NOW, BUT CAN BE ADDED TO THE SLURM SCRIPT\]

pop2vec/llm/src/new_code/pipeline.py

pop2vec/llm/slurm_scripts/pipeline.sh

We create hdf5 files in chunks in parallel in this step that the model can directly use for batching and training.

1. We first create a vocabulary using all the files from step 3 if it is not created yet.
2. We load the vocabulary if it already exists.
3. We read the parquet from step 4 and process it in chunks parallely.
4. During processing, We tokenize the sentences, apply MLM to the data as well as do the transformations for CLS training.
5. We save each chunk in hdf5 formats

### Pre-Training

Take one hdf5 file from step5 as input and trains+creates model checkpoints.

pop2vec/llm/src/new_code/pretrain.py

pop2vec/llm/slurm_scripts/pretrain_small.sh

### Temporary current status

On OSSC, we have data as output from step 2 as the following folders

1. Data-driven-good
2. HandPicked

We also have as output from step 5 for **HandPicked** under tanzir/pop2vec/llm/projects/dutch_real/gen_data/llm_4_million_v1/merged.h5

**training 1 → HandPicked (Try on Friday)**

We can right now start training because the output of step 5 (the final step in data processing pipeline) is ready at:

tanzir/pop2vec/llm/projects/dutch_real/gen_data/llm_4_million_v1/merged.h5

**training 2 → Data-driven-good + HandPicked (Try later)**

Data-driven-good and HandPicked outputs from step2 are stored here:

homedir/data/llm/raw/data-driven-good (no background files \[[Flavio Hafner](mailto:hafner.flavio@gmail.com) Verify programmatically\])

homedir/data/llm/raw/handPicked (no background files \[[Flavio Hafner](mailto:hafner.flavio@gmail.com) Verify programmatically\])

Data-driven-good and HandPicked both have data for 27 million people processed by Ana. So we need to do subsetting.

Step4 will get rid of people who are not present in the given background file. So if we use a small background file with x people, we keep only those x people starting from step 4 and onwards. Step 5 then further removes more people who do not have rich enough data (usually around 50%).

homedir/data/llm/background_files contains the background files for different number of people.

homedir/data/llm/background_files/background_4million.csv is the one we should use for all experiments

**Directions to follow:**

1. These 1 file and 2 folders should be moved to a single new directory.
    1. homedir/data/llm/background_files/background_4million.csv
    2. homedir/data/llm/raw/data-driven-good
    3. homedir/data/llm/raw/handPicked
2. Then step 3 - 5 should be applied on this new directory, so we get one final merged.h5 file which we can then train on. This means specifying this new directory as the input to step 3.
