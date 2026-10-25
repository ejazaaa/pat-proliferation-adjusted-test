import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from scipy.stats import spearmanr, mannwhitneyu
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
PROLIF=['CCNB1','CCNB2','CDK1','CDC20','BUB1','AURKA','TOP2A','CENPF','BIRC5','RRM2']
MIT=['KIF2C','CENPE','KIF20A','TYMS','TOP2A','KIF15','UBE2C','CCNB2','TPX2','EZH2','CDC20','CENPF','TRIP13',
     'CCNB1','CDK1','BUB1','AURKA','BIRC5','RRM2','MKI67','PLK1','AURKB','CCNA2','MCM2','PCNA','TTK','NUSAP1','PRC1','RACGAP1','BUB1B','CDK2','CHEK1','FOXM1','KIF11','KIF23','MAD2L1','NDC80','ZWINT','CDKN3','ASPM']
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
rows=[]
for g in MIT:
    if g not in S.index: continue
    pset=[x for x in PROLIF if x in S.index and x!=g]
    x=d.copy(); x['x']=z(S.loc[g,x.index]); x['prolif']=z(S.loc[pset].apply(z,axis=1).mean())
    x=x.dropna(subset=['x','prolif','efs_t','efs_e','age18','highrisk'])
    u=CoxPHFitter(penalizer=1e-4).fit(x[['x','efs_t','efs_e']],'efs_t','efs_e')
    a=CoxPHFitter(penalizer=1e-4).fit(x[['x','age18','highrisk','prolif','efs_t','efs_e']],'efs_t','efs_e')
    rows.append({'gene':g,'z_uni':u.summary.z['x'],'p_uni':u.summary.p['x'],'z_adj':a.summary.z['x'],
                 'p_adj':a.summary.p['x'],'rho_prolif':spearmanr(x.x,x.prolif).correlation})
r=pd.DataFrame(rows); r.to_csv('mitotic_class_tested.csv',index=False)
print('mitotic/cell-cycle genes tested:',len(r))
print('univariable |z|>1.96:',(r.z_uni.abs()>1.96).sum(),'| adjusted:',(r.z_adj.abs()>1.96).sum(),
      '| adjusted in the WORSE direction:',((r.z_adj>1.96)).sum())
print('median |rho| with proliferation: %.2f (range %.2f-%.2f)'%(r.rho_prolif.abs().median(),r.rho_prolif.abs().min(),r.rho_prolif.abs().max()))
print(r.round(3).to_string(index=False))
