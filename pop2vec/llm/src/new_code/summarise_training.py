"Create validation/training curves and summarise training for different models"

import pandas as pd 
import json
import matplotlib.pyplot as plt 


rootdir = "/gpfs/ostor/ossc9424/homedir/"
datapath = rootdir + "Tanzir/LifeToVec_Nov/projects/dutch_real/models" 
out_path = rootdir +  "output/" 

models = {
    "2017_small/lightning_logs/version_11/": {
        "batch_size": 256,
        "name": "small"
    },
    "2017_medium/lightning_logs/version_1": {
        "batch_size": 128,
        "name": "medium"
    },
    "2017_medium2x/lightning_logs/version_1": {
        "batch_size": 64,
        "name": "medium2x"
    },
    "2017_large/lightning_logs/version_1/": {
        "batch_size": 8,
        "name": "large"
    }
}


def read_logfile(filename, modelname):
    "Read csv file from logger. Deal with different data types across models."
    colnames = ["epoch", "step", "val/loss", "train/loss_step"]

    if modelname != "medium2x":
        df = pd.read_csv(
            filename + "metrics.csv",
            na_values=[],
            keep_default_na=False,
            usecols=colnames
        )
    else:
        df = pd.read_csv(
            filename + "metrics.csv",
            na_values=[],
            keep_default_na=False,
            usecols=colnames,
            low_memory=False,
            dtype={
                "step": str
            }
        )
        for v in ["step", "epoch"]:
            df[v] = pd.to_numeric(df[v])

    return df 


def make_plot(df, yvar, xvar, stage, meta):
    """Plot loss across training steps
    
    Args:
    df (pd.DataFrame): input data to plot from
    var (str): variable on y-axis
    xvar (str): variable on x-axis
    stage (str): indicates training/validation step 
    meta (dict): dictionary with model metadata
    """
    val_mask = ~df[yvar].isna()
    df_plot = df.loc[val_mask, [xvar, yvar]]

    plt.figure(figsize=(8,6))
    plt.plot(df_plot[xvar], df_plot[yvar], marker="o")
    plt.title(f"{stage} loss")
    plt.xlabel(xvar)
    plt.ylabel("loss")
    figname = out_path + meta.get("name") + "_" + stage + ".pdf"
    plt.savefig(figname)


def summarise(filename, meta):
    "Process csv logs and make validation/training curve"
    out_dict = {}
    modelname = meta.get("name")

    df = read_logfile(filename, modelname)

    df["n_samples"] = df["step"] * meta.get("batch_size")
    max_sample = df["n_samples"].max() 
    max_step = df["step"].max() 
    out_dict["max_sample"] = max_sample
    out_dict["max_step"] = max_step 

    fig_map = {
        "val": "val/loss",
        "train": "train/loss_step"
    }
    for stage, yvar in fig_map.items():
        df[yvar] = pd.to_numeric(df[yvar], errors="coerce")
        out_dict[f"min_loss_{stage}"] = df[yvar].min()

        xvar = "n_samples"
        nan_mask = df[yvar] == "nan"

        min_sample_nan = df.loc[nan_mask, xvar].min() 
        out_dict[f"first_nan_{stage}"] = min_sample_nan

        make_plot(df, yvar, xvar, stage, meta)

    return out_dict



if __name__ == "__main__":
    results = {}
    for model, metadata in models.items():
        modelpath = datapath + model 
        print(f"current model: {model}")
        res = summarise(modelpath, metadata) 
        results[model] = res 

    with open(f"{out_path}results.json", "w") as f: 
        json.dump(results, f, indent=2)
