# train_baselines.py  –  ultra-compact baseline runner
import sys, json, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             roc_auc_score, f1_score, accuracy_score)
# from mord import LogisticIT

X=sp.load_npz(f"{sys.argv[1]}/X.npz")
ids=pd.read_parquet(f"{sys.argv[1]}/ids.parquet").RINPERSOON.tolist()
row={k:i for i,k in enumerate(ids)}
cols=[]
out=[]
for cfg in pd.read_csv(sys.argv[2]).config:
  j=json.load(open(cfg))
  tr=pd.read_parquet(j["train_path"])
  va=pd.read_parquet(j["val_path"])
  for tgt,(typ,_) in j["target_column"].items():
    def Y(df):return df[[tgt,"RINPERSOON"]].dropna()
    trY=Y(tr); vaY=Y(va)
    Xt=X[[row[i] for i in trY.RINPERSOON]]
    Xv=X[[row[i] for i in vaY.RINPERSOON]]
    yt=trY[tgt].values; yv=vaY[tgt].values
    if typ=="numeric":
      m=Ridge(alpha=1)
      m.fit(Xt,yt);p=m.predict(Xv)
      out.append([j["task_file"],tgt,typ,r2_score(yv,p),
                  mean_squared_error(yv,p,squared=False),
                  mean_absolute_error(yv,p),np.nan,np.nan,np.nan])
    elif typ=="binary":
      m=LogisticRegression(penalty="l1",solver="saga",class_weight="balanced")
      m.fit(Xt,yt);p=m.predict_proba(Xv)[:,1];q=(p>=.5)
      out.append([j["task_file"],tgt,typ,np.nan,np.nan,np.nan,
                  accuracy_score(yv,q),roc_auc_score(yv,p),f1_score(yv,q)])
    elif typ=="categorical":
      m=LogisticRegression(multi_class="multinomial",solver="saga")
      m.fit(Xt,yt);q=m.predict(Xv)
      out.append([j["task_file"],tgt,typ,np.nan,np.nan,np.nan,
                  accuracy_score(yv,q),np.nan,np.nan])
    # elif typ=="ordinal":
    #   m=LogisticIT(alpha=1)
    #   m.fit(Xt.toarray(),yt);q=m.predict(Xv.toarray())
    #   out.append([j["task_file"],tgt,typ,np.nan,np.nan,
    #               mean_absolute_error(yv,q),accuracy_score(yv,q),np.nan,np.nan])

pd.DataFrame(out,columns=["task","target","type","R2","RMSE","MAE",
                          "ACC","AUC","F1"]).to_csv(sys.argv[3],index=False)
print("done")
