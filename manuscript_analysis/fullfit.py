import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
PROLIF=['CCNB1','CCNB2','CDK1','CDC20','BUB1','AURKA','TOP2A','CENPF','BIRC5','RRM2']
def collapse(e,mp):
    idx=pd.Index(e.index.astype(str)).map(mp); e=e[idx.notna()]; e.index=idx[idx.notna()]
    return e.groupby(level=0).max()
z=lambda v:(v-v.mean())/v.std()
def prep(Z,d,common):
    Z=Z.loc[[g for g in common if g in Z.index]]
    return Z,d
r2s=pd.read_csv('refseq2symbol.csv',index_col=0).symbol
cl=pd.read_csv('SEQC_clinical_harmonized.csv')
e=pd.read_csv(U+'GSE62564_SEQC_NB_RNA-Seq_log2RPM_txt.gz',sep='\t',index_col=0)
e=e[cl.title.tolist()]; e.columns=cl.gsm.tolist(); e.index=[str(i).split('.')[0] for i in e.index]
S=collapse(e,r2s); S=S[S.mean(1)>S.mean(1).quantile(.25)]
ZS=S.sub(S.mean(1),axis=0).div(S.std(1).replace(0,np.nan),axis=0).dropna(); dS=cl.set_index('gsm').loc[ZS.columns].copy()
k=pd.read_csv('KOCAK_clinical_harmonized.csv')
ek=pd.read_csv(U+'GSE45547_series_matrix_txt_1.gz',sep='\t',comment='!',index_col=0,low_memory=False)
ek.columns=[c.strip('"') for c in ek.columns]; ek=np.log2(ek[[g for g in k.gsm if g in ek.columns]].clip(lower=1))
K=collapse(ek,pd.read_csv('gpl16876_probe2symbol.csv',index_col=0,dtype={'id':str}).symbol)
K=K[K.mean(1)>K.mean(1).quantile(.25)]
ZK=K.sub(K.mean(1),axis=0).div(K.std(1).replace(0,np.nan),axis=0).dropna(); dK=k.set_index('gsm').loc[ZK.columns].copy()
n=pd.read_csv(U+'GSE85047_series_matrix_txt_1.gz',sep='\t',comment='!',index_col=0)
n.columns=[c.strip('"') for c in n.columns]; cn=pd.read_csv('NRC_clinical_harmonized.csv')
N=collapse(n[[g for g in cn.gsm if g in n.columns]],pd.read_csv('gpl5175_probe2symbol.csv',index_col=0,dtype={'id':str}).symbol)
N=N[N.mean(1)>N.mean(1).quantile(.25)]
ZN=N.sub(N.mean(1),axis=0).div(N.std(1).replace(0,np.nan),axis=0).dropna(); dN=cn.set_index('gsm').loc[ZN.columns].copy()
m=pd.read_csv('META3_SEQC_NRC_KOCAK.csv'); common=m.gene.tolist()
cohorts=[('SEQC',ZS,dS,['age18','highrisk']),('NRC',ZN,dN,['age18','stage4','mycn']),('Kocak',ZK,dK,['age18','stage4','mycn'])]
genes=sorted(set(m[m.replicated_3of3].gene)|{'TRIP13','KIF2C','CCNB2'})
rows=[]
for g in genes:
    r={'gene':g}
    for name,Z,d,cov in cohorts:
        if g not in Z.index: continue
        pset=[x for x in PROLIF if x in Z.index and x!=g]
        x=d.copy(); x['x']=z(Z.loc[g,x.index]); x['prolif']=z(Z.loc[pset].mean())
        x=x.dropna(subset=['x','prolif','efs_t','efs_e']+cov)
        f=CoxPHFitter(penalizer=1e-4).fit(x[['x']+cov+['prolif','efs_t','efs_e']],'efs_t','efs_e')
        r[f'{name}_hr']=np.exp(f.params_['x']); r[f'{name}_z']=f.summary.z['x']; r[f'{name}_p']=f.summary.p['x']
    rows.append(r)
res=pd.DataFrame(rows)
W=np.sqrt([183,96,150])
Zs=res[['SEQC_z','NRC_z','Kocak_z']].values
res['meta_z']=np.nansum(Zs*W,1)/np.sqrt((W**2).sum())
res['meta_p']=2*norm.sf(abs(res.meta_z))
res['n_p05']=(res[['SEQC_p','NRC_p','Kocak_p']]<0.05).sum(1)
res['same_dir']=np.abs(np.sign(Zs).sum(1))==3
res['consensus_fullfit']=(res.n_p05==3)&res.same_dir
res.sort_values('meta_p').to_csv('FULLFIT_consensus_and_index.csv',index=False)
print('index genes:'); print(res[res.gene.isin(['TRIP13','KIF2C','CCNB2'])][['gene','SEQC_hr','SEQC_z','SEQC_p','NRC_z','NRC_p','Kocak_z','Kocak_p','meta_p']].round(4).to_string(index=False))
print('\nof the 44 score-test consensus genes, still consensus on full fits:',res[res.gene.isin(m[m.replicated_3of3].gene)].consensus_fullfit.sum(),'of 44')
print(res[res.gene.isin(m[m.replicated_3of3].gene)].sort_values('meta_p').head(12)[['gene','SEQC_z','SEQC_p','NRC_z','NRC_p','Kocak_z','Kocak_p','meta_z','consensus_fullfit']].round(3).to_string(index=False))
