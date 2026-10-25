import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from fastcox import score_test, score_test_offset
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
k=pd.read_csv(U+'export_to_file.tsv',sep='\t')
k.columns=[c.strip('"').strip() for c in k.columns]
for c in k.columns: k[c]=k[c].astype(str).str.strip('"').str.strip()
cl=pd.DataFrame({'gsm':k.samplenames,
 'efs_t':pd.to_numeric(k.eventfreesurvival,errors='coerce'),
 'efs_e':k.event.str.lower().map({'yes':1.,'no':0.}),
 'os_t':pd.to_numeric(k.overallsurvival,errors='coerce'),
 'os_e':k.osevent.str.lower().map({'yes':1.,'no':0.}),
 'age18':k.age.str.lower().map({'more_18_months':1.,'less_18_months':0.}),
 'mycn':k.mycn.str.lower().map({'amp':1.,'non_amp':0.}),
 'stage':k.stage.str.lower().str.replace('st','',regex=False)})
cl['stage4']=cl.stage.eq('4').astype(float).where(cl.stage.isin(['1','2','3','4','4s']))
print('n',len(cl),'EFS events',cl.efs_e.sum(),'OS events',cl.os_e.sum(),'MYCN amp',cl.mycn.sum(),'stage4',cl.stage4.sum(),'age>=18mo',cl.age18.sum())
print('missing:',cl[['efs_t','efs_e','os_t','os_e','age18','mycn','stage4']].isna().sum().to_dict())
cl.to_csv('KOCAK_clinical_harmonized.csv',index=False)
e=pd.read_csv(U+'GSE45547_series_matrix_txt_1.gz',sep='\t',comment='!',index_col=0,low_memory=False)
e.columns=[c.strip('"') for c in e.columns]
e=e[[g for g in cl.gsm if g in e.columns]]
print('expr raw',e.shape,'range',round(np.nanmin(e.values),1),round(np.nanmax(e.values),1))
if np.nanmax(e.values)>100: e=np.log2(e.clip(lower=1)); print('applied log2')
mp=pd.read_csv('gpl16876_probe2symbol.csv',index_col=0,dtype={'id':str}).symbol
idx=pd.Index(e.index.astype(str)).map(mp); e=e[idx.notna()]; e.index=idx[idx.notna()]
K=e.groupby(level=0).max(); K=K[K.mean(1)>K.mean(1).quantile(.25)]
Z=K.sub(K.mean(1),axis=0).div(K.std(1).replace(0,np.nan),axis=0).dropna()
d=cl.set_index('gsm').loc[Z.columns].copy()
PROLIF=['CCNB1','CCNB2','CDK1','CDC20','BUB1','AURKA','TOP2A','CENPF','BIRC5','RRM2']
ps=[g for g in PROLIF if g in Z.index]; print('Kocak genes',Z.shape,'prolif genes',len(ps))
d['prolif']=Z.loc[ps].mean(); d['prolif']=(d.prolif-d.prolif.mean())/d.prolif.std()
print('rho(prolif,MYCN)=%.2f'%spearmanr(d.prolif,d.mycn.fillna(0)).correlation)
def uni(dur,ev,tag):
    x=d[[dur,ev]].dropna().index
    z,_,_=score_test(Z[x].values,d.loc[x,dur].values,d.loc[x,ev].values)
    o=pd.DataFrame({'gene':Z.index,'z':z,'p':2*norm.sf(abs(z))}); o['fdr']=multipletests(o.p.fillna(1),method='fdr_bh')[1]
    o.sort_values('p').to_csv(tag,index=False); print(tag,'n',len(x),'events',int(d.loc[x,ev].sum()),'FDR<0.05:',(o.fdr<0.05).sum()); return o
def adj(dur,ev,covs,tag):
    x=d.dropna(subset=covs+[dur,ev]); Zx=Z[x.index]
    base=CoxPHFitter(penalizer=1e-4).fit(x[covs+[dur,ev]],dur,ev)
    w=np.exp(base.predict_log_partial_hazard(x).values)
    z,_,_=score_test_offset(Zx.values,x[dur].values,x[ev].values,w)
    o=pd.DataFrame({'gene':Zx.index,'z_adj':z,'p_adj':2*norm.sf(abs(z))})
    o['fdr_adj']=multipletests(o.p_adj.fillna(1),method='fdr_bh')[1]
    o['n']=len(x); o['events']=int(x[ev].sum()); o['c_base']=base.concordance_index_
    o.sort_values('p_adj').to_csv(tag,index=False)
    print(tag,covs,'n',len(x),'events',int(x[ev].sum()),'FDR<0.05:',(o.fdr_adj<0.05).sum()); return o
uni('efs_t','efs_e','KOCAK_EFS_univariable.csv')
A=adj('efs_t','efs_e',['age18','stage4','mycn','prolif'],'KOCAK_EFS_adjusted_full.csv')
adj('efs_t','efs_e',['age18','stage4','mycn'],'KOCAK_EFS_adjusted_noprolif.csv')
adj('os_t','os_e',['age18','stage4','mycn','prolif'],'KOCAK_OS_adjusted_full.csv')
print(A.head(12).round(4).to_string(index=False))
for g in ['TRIP13','KIF2C','CCNB2','CDIN1','ALG2','ARK2C','LPIN2','UBE2Q1']:
    if g in set(A.gene): print(A[A.gene==g].round(4).to_string(index=False,header=False))
