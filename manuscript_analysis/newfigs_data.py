import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from scipy.stats import spearmanr, mannwhitneyu
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
PROLIF=['CCNB1','CCNB2','CDK1','CDC20','BUB1','AURKA','TOP2A','CENPF','BIRC5','RRM2']
WT12=['KIF2C','CENPE','KIF20A','TYMS','TOP2A','KIF15','UBE2C','CCNB2','TPX2','EZH2','CDC20','CENPF']
def collapse(e,mp):
    idx=pd.Index(e.index.astype(str)).map(mp); e=e[idx.notna()]; e.index=idx[idx.notna()]
    return e.groupby(level=0).max()
z=lambda v:(v-v.mean())/v.std()
r2s=pd.read_csv('refseq2symbol.csv',index_col=0).symbol
cl=pd.read_csv('SEQC_clinical_harmonized.csv')
e=pd.read_csv(U+'GSE62564_SEQC_NB_RNA-Seq_log2RPM_txt.gz',sep='\t',index_col=0)
e=e[cl.title.tolist()]; e.columns=cl.gsm.tolist(); e.index=[str(i).split('.')[0] for i in e.index]
S=collapse(e,r2s); S=S[S.mean(1)>S.mean(1).quantile(.25)]
ZS=S.sub(S.mean(1),axis=0).div(S.std(1).replace(0,np.nan),axis=0).dropna(); dS=cl.set_index('gsm').loc[ZS.columns].copy()
dS['prolif']=z(ZS.loc[[g for g in PROLIF if g in ZS.index]].mean())
# correlation of each gene with proliferation, per cohort (SEQC) + for index genes all cohorts
m=pd.read_csv('META3_SEQC_NRC_KOCAK.csv')
common=[g for g in m.gene if g in ZS.index]
rho=pd.Series({g:spearmanr(ZS.loc[g],dS.prolif).correlation for g in common})
rho.to_csv('rho_prolif_SEQC.csv')
# discrimination: C-index of base model vs base+gene, SEQC, for index + top consensus
f=pd.read_csv('FULLFIT_consensus_and_index.csv')
tops=f[f.consensus_fullfit].nsmallest(10,'meta_p').gene.tolist()
base=CoxPHFitter(penalizer=1e-4).fit(dS[['age18','highrisk','prolif','efs_t','efs_e']],'efs_t','efs_e')
rows=[{'gene':'base model','c':base.concordance_index_,'delta':0.0,'grp':'base'}]
for g in ['TRIP13','KIF2C','CCNB2']+tops:
    if g not in ZS.index: continue
    x=dS.copy(); x['x']=z(ZS.loc[g,x.index])
    ff=CoxPHFitter(penalizer=1e-4).fit(x[['x','age18','highrisk','prolif','efs_t','efs_e']],'efs_t','efs_e')
    rows.append({'gene':g,'c':ff.concordance_index_,'delta':ff.concordance_index_-base.concordance_index_,
                 'grp':'index' if g in ['TRIP13','KIF2C','CCNB2'] else 'consensus'})
pd.DataFrame(rows).to_csv('cindex_gain.csv',index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
# class-level: adjusted z distribution for WT12 hub genes vs all vs consensus (use fullfit for WT12)
cohorts=[('SEQC',ZS,dS,['age18','highrisk'])]
wt=[]
for g in WT12:
    if g not in ZS.index: continue
    pset=[x for x in PROLIF if x in ZS.index and x!=g]
    x=dS.copy(); x['x']=z(ZS.loc[g,x.index]); x['prolif']=z(ZS.loc[pset].mean())
    ff=CoxPHFitter(penalizer=1e-4).fit(x[['x','age18','highrisk','prolif','efs_t','efs_e']],'efs_t','efs_e')
    uni=CoxPHFitter(penalizer=1e-4).fit(x[['x','efs_t','efs_e']],'efs_t','efs_e')
    wt.append({'gene':g,'z_uni':uni.summary.z['x'],'z_adj':ff.summary.z['x'],'p_adj':ff.summary.p['x'],
               'rho_prolif':spearmanr(x.x,x.prolif).correlation})
wt=pd.DataFrame(wt); wt.to_csv('WT12_in_SEQC.csv',index=False)
print(wt.round(3).to_string(index=False))
print('WT12 adjusted |z|>1.96:',(wt.z_adj.abs()>1.96).sum(),'of',len(wt))
# KM for top consensus gene
top=tops[0]
km=dS[['efs_t','efs_e','highrisk']].copy(); km['x']=ZS.loc[top,km.index].values
km['grp']=np.where(km.x>km.x.median(),'high','low'); km['gene']=top; km.to_csv('KM_topconsensus.csv')
print('KM gene:',top)
