import pandas as pd
import time
import datatable as dt

# 210 seconds
def compress(df):
    st = time.time()
    df.to_csv('data/output.csv.gz', compression='gzip', index=False)
    end = time.time()
    print(f"compression time: {end-st} seconds")

# 149 seconds
def plain(df):
    st = time.time()
    df.to_csv('data/output.csv', index=False)
    end = time.time()
    print(f"plain time: {end-st} seconds")

# 
def buffer(df):
    st = time.time()
    with open('data/buffered.csv', 'w', buffering=4194304) as f:  # buffer size of 4MB
        df.to_csv(f, index=False)
    end = time.time()
    print(f"buffer time: {end-st} seconds")

def datatable(df):
    st = time.time()
    frame = dt.Frame(df)
    frame.to_csv('data/output_dt.csv')
    end = time.time()
    print(f"dt time: {end-st} seconds")


st = time.time()
df = pd.read_csv('data/final_train_data.csv')
end = time.time()
print(f"loading time: {end-st} seconds")
#compress(df)
#plain(df)
#datatable(df)
buffer(df)

exit(0)