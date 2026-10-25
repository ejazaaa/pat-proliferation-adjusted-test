import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from scipy.stats import ttest_ind, spearmanr
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
PROLIF=['CCNB1','CCNB2','CDK1','CDC20','BUB1','AURKA','TOP2A','CENPF','BIRC5','RRM2']
def collapse(e,mp):
    idx=pd.Index(e.index.astype(str)).map(mp); e=e[idx.notna()]; e.index=idx[idx.notna()]
    return e.groupby(level=0).max()
z=lambda v:(v-v.mean())/v.std()
r2s=pd.read_csv('refseq2symbol.csv',index_col=0).symbol
cl=pd.read_csv('SEQC_clinical_harmonized.csv')
e=pd.read_csv(U+'GSE62564_SEQC_NB_RNA-Seq_log2RPM_txt.gz',sep='\t',index_col=0)
e=e[cl.title.tolist()]; e.columns=cl.gsm.tolist(); e.index=[str(i).split('.')[0] for i in e.index]
S=collapse(e,r2s); S=S[S.mean(1)>S.mean(1).quantile(.25)]
d=cl.set_index('gsm').loc[S.columns].copy()
# Step 1: DEG high-risk vs lower-risk (the neuroblastoma analogue of the published workflow)
hi=d.index[d.highrisk.eq(1)]; lo=d.index[d.highrisk.eq(0)]
t,p=ttest_ind(S[hi],S[lo],axis=1,equal_var=False,nan_policy='omit')
deg=pd.DataFrame({'gene':S.index,'log2fc':S[hi].mean(1)-S[lo].mean(1),'p':p})
deg['fdr']=multipletests(deg.p.fillna(1),method='fdr_bh')[1]
up=deg[(deg.fdr<0.05)&(deg.log2fc>1)].copy(); dn=deg[(deg.fdr<0.05)&(deg.log2fc<-1)].copy()
print('DEGs: up',len(up),'down',len(dn))
degs=pd.concat([up,dn]).gene.tolist()
# Step 2: hub selection by co-expression connectivity (proxy for the PPI/cytoHubba step)
Zd=S.loc[degs].apply(z,axis=1)
C=np.corrcoef(Zd.values); np.fill_diagonal(C,0)
deg_k=pd.Series((np.abs(C)>0.5).sum(1),index=Zd.index).sort_values(ascending=False)
hubs=deg_k.head(12).index.tolist()
print('co-expression hubs:',hubs)
deg_k.to_csv('NB_internal_hub_degree.csv')
# Step 3: test hubs the way the workflow would, then adjusted
d['prolif_all']=z(S.loc[[g for g in PROLIF if g in S.index]].apply(z,axis=1).mean())
rows=[]
for g in hubs:
    pset=[x for x in PROLIF if x in S.index and x!=g]
    x=d.copy(); x['x']=z(S.loc[g,x.index]); x['prolif']=z(S.loc[pset].apply(z,axis=1).mean())
    x=x.dropna(subset=['x','prolif','efs_t','efs_e','age18','highrisk'])
    uni=CoxPHFitter(penalizer=1e-4).fit(x[['x','efs_t','efs_e']],'efs_t','efs_e')
    adj=CoxPHFitter(penalizer=1e-4).fit(x[['x','age18','highrisk','prolif','efs_t','efs_e']],'efs_t','efs_e')
    rows.append({'gene':g,'degree':int(deg_k[g]),'z_uni':uni.summary.z['x'],'p_uni':uni.summary.p['x'],
                 'z_adj':adj.summary.z['x'],'p_adj':adj.summary.p['x'],'rho_prolif':spearmanr(x.x,x.prolif).correlation})
r=pd.DataFrame(rows); r.to_csv('NB_internal_hubs_tested.csv',index=False)
print(r.round(3).to_string(index=False))
print('univariable |z|>1.96:',(r.z_uni.abs()>1.96).sum(),'/',len(r),'| adjusted:',(r.z_adj.abs()>1.96).sum())
