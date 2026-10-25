import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
PROLIF=['CCNB1','CCNB2','CDK1','CDC20','BUB1','AURKA','TOP2A','CENPF','BIRC5','RRM2']
def collapse(e,mp):
    idx=pd.Index(e.index.astype(str)).map(mp); e=e[idx.notna()]; e.index=idx[idx.notna()]
    return e.groupby(level=0).max()
z=lambda v:(v-v.mean())/v.std()
# SEQC
r2s=pd.read_csv('refseq2symbol.csv',index_col=0).symbol
cl=pd.read_csv('SEQC_clinical_harmonized.csv')
e=pd.read_csv(U+'GSE62564_SEQC_NB_RNA-Seq_log2RPM_txt.gz',sep='\t',index_col=0)
e=e[cl.title.tolist()]; e.columns=cl.gsm.tolist(); e.index=[str(i).split('.')[0] for i in e.index]
S=collapse(e,r2s); S=S[S.mean(1)>S.mean(1).quantile(.25)]
ZS=S.sub(S.mean(1),axis=0).div(S.std(1).replace(0,np.nan),axis=0).dropna()
dS=cl.set_index('gsm').loc[ZS.columns].copy()
# Kocak
k=pd.read_csv('KOCAK_clinical_harmonized.csv')
ek=pd.read_csv(U+'GSE45547_series_matrix_txt_1.gz',sep='\t',comment='!',index_col=0,low_memory=False)
ek.columns=[c.strip('"') for c in ek.columns]; ek=ek[[g for g in k.gsm if g in ek.columns]]
ek=np.log2(ek.clip(lower=1))
mp=pd.read_csv('gpl16876_probe2symbol.csv',index_col=0,dtype={'id':str}).symbol
K=collapse(ek,mp); K=K[K.mean(1)>K.mean(1).quantile(.25)]
ZK=K.sub(K.mean(1),axis=0).div(K.std(1).replace(0,np.nan),axis=0).dropna()
dK=k.set_index('gsm').loc[ZK.columns].copy()
# NRC
n=pd.read_csv(U+'GSE85047_series_matrix_txt_1.gz',sep='\t',comment='!',index_col=0)
n.columns=[c.strip('"') for c in n.columns]
cn=pd.read_csv('NRC_clinical_harmonized.csv'); n=n[[g for g in cn.gsm if g in n.columns]]
p2s=pd.read_csv('gpl5175_probe2symbol.csv',index_col=0,dtype={'id':str}).symbol
N=collapse(n,p2s); N=N[N.mean(1)>N.mean(1).quantile(.25)]
ZN=N.sub(N.mean(1),axis=0).div(N.std(1).replace(0,np.nan),axis=0).dropna()
dN=cn.set_index('gsm').loc[ZN.columns].copy()
cohorts=[('SEQC',ZS,dS,['age18','highrisk']),('NRC',ZN,dN,['age18','stage4','mycn']),('Kocak',ZK,dK,['age18','stage4','mycn'])]
rows=[]
for g in PROLIF+['TRIP13','KIF2C']:
    r={'gene':g}
    for name,Z,d,cov in cohorts:
        if g not in Z.index: r[name]=np.nan; continue
        pset=[x for x in PROLIF if x in Z.index and x!=g]      # leave-one-out
        full=[x for x in PROLIF if x in Z.index]
        x=d.copy(); x['x']=z(Z.loc[g,x.index]); x['loo']=z(Z.loc[pset].mean()); x['all']=z(Z.loc[full].mean())
        x=x.dropna(subset=['x','loo','all','efs_t','efs_e']+cov)
        f1=CoxPHFitter(penalizer=1e-4).fit(x[['x']+cov+['loo','efs_t','efs_e']],'efs_t','efs_e')
        f2=CoxPHFitter(penalizer=1e-4).fit(x[['x']+cov+['all','efs_t','efs_e']],'efs_t','efs_e')
        r[name+'_z_loo']=f1.summary.z['x']; r[name+'_p_loo']=f1.summary.p['x']
        r[name+'_z_selfincl']=f2.summary.z['x']
    rows.append(r)
res=pd.DataFrame(rows)
res.to_csv('leave_one_out_prolif.csv',index=False)
print(res.round(3).to_string(index=False))
