import numpy as np, pandas as pd, warnings, sys, gzip
sys.path.insert(0,'/home/claude/nb')
from fastcox import score_test, score_test_offset
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
r2s=pd.read_csv('refseq2symbol.csv',index_col=0).symbol
p2s=pd.read_csv('gpl5175_probe2symbol.csv',index_col=0,dtype={'id':str}).symbol

def collapse(e,mp):
    idx=pd.Index(e.index.astype(str)).map(mp)
    e=e[idx.notna()]; e.index=idx[idx.notna()]
    return e.groupby(level=0).max()   # max-expression probe per gene

# ---- SEQC (gene level) ----
cl=pd.read_csv('SEQC_clinical_harmonized.csv')
e=pd.read_csv(U+'GSE62564_SEQC_NB_RNA-Seq_log2RPM_txt.gz',sep='\t',index_col=0)
e=e[cl.title.tolist()]; e.columns=cl.gsm.tolist()
e.index=[str(i).split('.')[0] for i in e.index]
S=collapse(e,r2s); S=S[S.mean(1)>S.mean(1).quantile(.25)]
Zs=S.sub(S.mean(1),axis=0).div(S.std(1).replace(0,np.nan),axis=0).dropna()
ds=cl.set_index('gsm').loc[Zs.columns].copy()
print('SEQC genes',Zs.shape)
# ---- NRC (gene level) ----
n=pd.read_csv(U+'GSE85047_series_matrix_txt_1.gz',sep='\t',comment='!',index_col=0)
n.columns=[c.strip('"') for c in n.columns]
cn=pd.read_csv('NRC_clinical_harmonized.csv')
n=n[[g for g in cn.gsm if g in n.columns]]
N=collapse(n,p2s); N=N[N.mean(1)>N.mean(1).quantile(.25)]
Zn=N.sub(N.mean(1),axis=0).div(N.std(1).replace(0,np.nan),axis=0).dropna()
dn=cn.set_index('gsm').loc[Zn.columns].copy()
print('NRC genes',Zn.shape)
common=sorted(set(Zs.index)&set(Zn.index)); print('genes in both cohorts',len(common))
Zs=Zs.loc[common]; Zn=Zn.loc[common]
PROLIF=['CCNB1','CCNB2','CDK1','CDC20','BUB1','PLK1','AURKA','TOP2A','MKI67','CENPF','BIRC5','RRM2','TYMS','UBE2C']
pset=[g for g in PROLIF if g in common]; print('proliferation genes available',pset)
def prep(Z,d):
    d=d.copy(); d['prolif']=Z.loc[pset].mean().reindex(d.index)
    d['prolif']=(d.prolif-d.prolif.mean())/d.prolif.std(); return d
ds=prep(Zs,ds); dn=prep(Zn,dn)
print('SEQC rho(prolif,highrisk)=%.2f'%spearmanr(ds.prolif,ds.highrisk).correlation,
      '| NRC rho(prolif,MYCN)=%.2f'%spearmanr(dn.prolif,dn.mycn.fillna(0)).correlation)
def uni(Z,d,dur,ev):
    k=d[[dur,ev]].dropna().index
    z,_,_=score_test(Z[k].values,d.loc[k,dur].values,d.loc[k,ev].values)
    o=pd.DataFrame({'gene':Z.index,'z':z,'p':2*norm.sf(abs(z))}); o['fdr']=multipletests(o.p.fillna(1),method='fdr_bh')[1]; return o
def adj(Z,d,dur,ev,covs):
    x=d.dropna(subset=covs+[dur,ev]); Zx=Z[x.index]
    base=CoxPHFitter(penalizer=1e-4).fit(x[covs+[dur,ev]],dur,ev)
    w=np.exp(base.predict_log_partial_hazard(x).values)
    z,_,_=score_test_offset(Zx.values,x[dur].values,x[ev].values,w)
    o=pd.DataFrame({'gene':Zx.index,'z_adj':z,'p_adj':2*norm.sf(abs(z))})
    o['fdr_adj']=multipletests(o.p_adj.fillna(1),method='fdr_bh')[1]
    o['n']=len(x); o['events']=int(x[ev].sum()); o['c_base']=base.concordance_index_
    return o
S_uni=uni(Zs,ds,'efs_t','efs_e'); S_adj=adj(Zs,ds,'efs_t','efs_e',['age18','highrisk','prolif'])
N_uni=uni(Zn,dn,'efs_t','efs_e'); N_adj=adj(Zn,dn,'efs_t','efs_e',['age18','stage4','mycn','prolif'])
N_adj_np=adj(Zn,dn,'efs_t','efs_e',['age18','stage4','mycn'])
print('SEQC uni FDR<0.05 %d | adj %d'%((S_uni.fdr<0.05).sum(),(S_adj.fdr_adj<0.05).sum()))
print('NRC  uni FDR<0.05 %d | adj %d (no prolif %d)'%((N_uni.fdr<0.05).sum(),(N_adj.fdr_adj<0.05).sum(),(N_adj_np.fdr_adj<0.05).sum()))
m=(S_adj.rename(columns={'z_adj':'SEQC_z_adj','p_adj':'SEQC_p_adj','fdr_adj':'SEQC_fdr_adj'})[['gene','SEQC_z_adj','SEQC_p_adj','SEQC_fdr_adj']]
   .merge(N_adj.rename(columns={'z_adj':'NRC_z_adj','p_adj':'NRC_p_adj','fdr_adj':'NRC_fdr_adj'})[['gene','NRC_z_adj','NRC_p_adj','NRC_fdr_adj']],on='gene')
   .merge(S_uni.rename(columns={'z':'SEQC_z_uni','fdr':'SEQC_fdr_uni'})[['gene','SEQC_z_uni','SEQC_fdr_uni']],on='gene')
   .merge(N_uni.rename(columns={'z':'NRC_z_uni','fdr':'NRC_fdr_uni'})[['gene','NRC_z_uni','NRC_fdr_uni']],on='gene'))
m['same_direction_adj']=np.sign(m.SEQC_z_adj)==np.sign(m.NRC_z_adj)
m['replicated']=(m.SEQC_fdr_adj<0.05)&(m.NRC_p_adj<0.05)&m.same_direction_adj
m['combined_z']=(m.SEQC_z_adj+m.NRC_z_adj)/np.sqrt(2)
m['combined_p']=2*norm.sf(abs(m.combined_z)); m['combined_fdr']=multipletests(m.combined_p,method='fdr_bh')[1]
m.sort_values('combined_p').to_csv('REPLICATION_SEQC_NRC_genelevel.csv',index=False)
print('SEQC-significant genes:',(m.SEQC_fdr_adj<0.05).sum(),'| replicated in NRC:',m.replicated.sum())
print('direction concordance among SEQC-significant: %.1f%%'%(100*m.loc[m.SEQC_fdr_adj<0.05,'same_direction_adj'].mean()))
print('meta-analysis FDR<0.05:',(m.combined_fdr<0.05).sum())
print(m.nsmallest(15,'combined_p')[['gene','SEQC_z_adj','SEQC_fdr_adj','NRC_z_adj','NRC_p_adj','combined_z','combined_fdr','replicated']].round(4).to_string(index=False))
for g in ['TRIP13','KIF2C','CCNB2','CDIN1','PSTK','TUBD1','UBE2Q1','REEP4','ALG2','RAB11FIP4','TRMT6','MYCN']:
    if g in set(m.gene): print(m[m.gene==g][['gene','SEQC_z_uni','SEQC_fdr_uni','SEQC_z_adj','SEQC_fdr_adj','NRC_z_adj','NRC_p_adj','combined_fdr']].round(4).to_string(index=False))
