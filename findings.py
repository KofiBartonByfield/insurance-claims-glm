# Reproduces the numbers quoted in the README Findings. Run: python findings.py (after pip install -r requirements.txt)
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer
from src.data_loader import load_data
from src.features import *
from src.models import *
from src.evaluate import evaluate_model
df=prepare_features(load_data())
tr,te=train_test_split(df,test_size=0.2,random_state=42)
y,e=te.ClaimNb.values,te.Exposure.values
print("claims/policy-year", df.ClaimNb.sum()/df.Exposure.sum(), "zero-claim share", (df.ClaimNb==0).mean())
Xg=encode_for_glm(tr); Xt=encode_for_glm(te).reindex(columns=Xg.columns,fill_value=0)
glm=fit_poisson_glm(Xg,tr.ClaimNb.values,tr.Exposure.values)
xgb=fit_xgboost_poisson(encode_for_xgboost(tr),tr.ClaimNb.values,tr.Exposure.values)
pg=glm.predict(Xt)*e; px=xgb.predict(encode_for_xgboost(te))*e
print("sum pred/actual glm",pg.sum()/y.sum(),"xgb",px.sum()/y.sum())
# coefficients
c=pd.Series(glm.coef_,Xg.columns).sort_values()
print("GLM coefs (exp):"); print(np.exp(c).round(3).to_string())
# xgb importance
print(pd.Series(xgb.get_booster().get_score(importance_type="total_gain")).sort_values(ascending=False).pipe(lambda s:(s/s.sum()).round(3)).to_string())
# decile lift by xgb rate
te=te.assign(pg=pg,px=px)
te["dec"]=pd.qcut(te.px/te.Exposure,10,labels=False,duplicates="drop")
g=te.groupby("dec").agg(exp=("Exposure","sum"),act=("ClaimNb","sum"),px=("px","sum"),pg=("pg","sum"))
g["act_f"]=g.act/g.exp;g["xgb_f"]=g.px/g.exp;g["glm_f"]=g.pg/g.exp
print(g[["act_f","xgb_f","glm_f"]].round(3).to_string()); print("top/bottom actual", g.act_f.iloc[-1]/g.act_f.iloc[0])
# empirical drivage & bonusmalus
d=df.assign(b=pd.cut(df.DrivAge,[17,24,29,39,49,59,69,90]))
print((d.groupby("b").ClaimNb.sum()/d.groupby("b").Exposure.sum()).round(3).to_string())
d=df.assign(b=pd.cut(df.BonusMalus,[49,50,60,80,100,150]))
print((d.groupby("b").ClaimNb.sum()/d.groupby("b").Exposure.sum()).round(3).to_string())
print("share BM==50",(df.BonusMalus==50).mean())
# GLM with splines on numerics
def enc(d,st=None):
    num=["VehPower","VehAge","DrivAge","BonusMalus","Density"]
    base=pd.get_dummies(d[CATEGORICAL_FEATURES],drop_first=True).astype(float)
    return base,d[num]
bt,nt=enc(tr);be,ne=enc(te)
sp=SplineTransformer(n_knots=6,degree=3).fit(nt)
def full(b,n,cols=None):
    X=np.hstack([b.values,sp.transform(n)]);return X
glm2=fit_poisson_glm(full(bt,nt),tr.ClaimNb.values,tr.Exposure.values)
be=be.reindex(columns=bt.columns,fill_value=0)
p2=glm2.predict(full(be,ne))*e
for n,p in [("GLM",pg),("GLM+splines",p2),("XGB",px)]: print(n,evaluate_model(n,y,p,e))
