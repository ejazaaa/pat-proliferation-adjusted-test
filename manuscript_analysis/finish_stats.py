import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from fastcox import score_test, score_test_offset
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
r2s=pd.read_csv('refseq2symbol.csv',index_col=0).symbol
def collapse(e,mp):
    idx=pd.Index(e.index.astype(str)).map(mp); e=e[idx.notna()]; e.index=idx[idx.notna()]
    return e.groupby(level=0).max()
cl=pd.read_csv('SEQC_clinical_harmonized.csv')
e=pd.read_csv(U+'GSE62564_SEQC_NB_RNA-Seq_log2RPM_txt.gz',sep='\t',index_col=0)
e=e[cl.title.tolist()]; e.columns=cl.gsm.tolist(); e.index=[str(i).split('.')[0] for i in e.index]
S=collapse(e,r2s); S=S[S.mean(1)>S.mean(1).quantile(.25)]
Z=S.sub(S.mean(1),axis=0).div(S.std(1).replace(0,np.nan),axis=0).dropna()
d=cl.set_index('gsm').loc[Z.columns].copy()
PROLIF=['CCNB1','CCNB2','CDK1','CDC20','BUB1','AURKA','TOP2A','CENPF','BIRC5','RRM2']
d['prolif']=Z.loc[PROLIF].mean(); d['prolif']=(d.prolif-d.prolif.mean())/d.prolif.std()
common=pd.read_csv('META3_SEQC_NRC_KOCAK.csv').gene.tolist()
Zc=Z.loc[[g for g in common if g in Z.index]]
def adjcount(covs,Zx):
    x=d.dropna(subset=covs+['efs_t','efs_e']); Zq=Zx[x.index]
    base=CoxPHFitter(penalizer=1e-4).fit(x[covs+['efs_t','efs_e']],'efs_t','efs_e')
    w=np.exp(base.predict_log_partial_hazard(x).values)
    z,_,_=score_test_offset(Zq.values,x['efs_t'].values,x['efs_e'].values,w)
    p=2*norm.sf(abs(z)); fdr=multipletests(np.nan_to_num(p,nan=1),method='fdr_bh')[1]
    return (fdr<0.05).sum(), len(Zq)
print('SEQC common-gene set, adj age+HR (no prolif):',adjcount(['age18','highrisk'],Zc))
print('SEQC common-gene set, full adj:',adjcount(['age18','highrisk','prolif'],Zc))
# score test vs full Cox agreement, 200 random genes
rng=np.random.default_rng(0); samp=rng.choice(Zc.index,200,replace=False)
x=d.dropna(subset=['age18','highrisk','prolif','efs_t','efs_e'])
base=CoxPHFitter(penalizer=1e-4).fit(x[['age18','highrisk','prolif','efs_t','efs_e']],'efs_t','efs_e')
w=np.exp(base.predict_log_partial_hazard(x).values)
zs,_,_=score_test_offset(Zc.loc[samp,x.index].values,x['efs_t'].values,x['efs_e'].values,w)
zf=[]
for g in samp:
    q=x.copy(); q['x']=Zc.loc[g,q.index].values
    f=CoxPHFitter(penalizer=1e-4).fit(q[['x','age18','highrisk','prolif','efs_t','efs_e']],'efs_t','efs_e')
    zf.append(f.summary.z['x'])
zf=np.array(zf)
print('score vs full-fit z: Pearson r=%.4f, median |diff|=%.3f, max |diff|=%.3f'%(np.corrcoef(zs,zf)[0,1],np.median(abs(zs-zf)),abs(zs-zf).max()))
np.save('zs.npy',zs); np.save('zf.npy',zf)
# KM data for TRIP13 median split
km=d[['efs_t','efs_e']].copy(); km['TRIP13']=Z.loc['TRIP13',km.index].values
km['grp']=np.where(km.TRIP13>km.TRIP13.median(),'high','low'); km.to_csv('KM_TRIP13_SEQC.csv')
