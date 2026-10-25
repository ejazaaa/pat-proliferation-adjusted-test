import numpy as np, pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
rep=pd.read_csv('REPLICATION_SEQC_NRC_genelevel.csv')
k=pd.read_csv('KOCAK_EFS_adjusted_full.csv').rename(columns={'z_adj':'KOCAK_z_adj','p_adj':'KOCAK_p_adj','fdr_adj':'KOCAK_fdr_adj'})[['gene','KOCAK_z_adj','KOCAK_p_adj','KOCAK_fdr_adj']]
ku=pd.read_csv('KOCAK_EFS_univariable.csv').rename(columns={'z':'KOCAK_z_uni','fdr':'KOCAK_fdr_uni'})[['gene','KOCAK_z_uni','KOCAK_fdr_uni']]
m=rep.merge(k,on='gene').merge(ku,on='gene')
print('genes in all three cohorts:',len(m))
# sample sizes for weighting (events)
w=np.array([183,96,150],float); w=np.sqrt(w)/np.sqrt((w**2).sum()/w.sum()) # weights ~ sqrt(events)
W=np.sqrt(np.array([183,96,150],float))
Zs=m[['SEQC_z_adj','NRC_z_adj','KOCAK_z_adj']].values
m['meta3_z']=(Zs*W).sum(1)/np.sqrt((W**2).sum())
m['meta3_p']=2*norm.sf(abs(m.meta3_z))
m['meta3_fdr']=multipletests(m.meta3_p,method='fdr_bh')[1]
m['n_cohorts_p05']=(m[['SEQC_p_adj','NRC_p_adj','KOCAK_p_adj']]<0.05).sum(1)
sign=np.sign(Zs); m['all_same_direction']=(np.abs(sign.sum(1))==3)
m['replicated_3of3']=(m.n_cohorts_p05==3)&m.all_same_direction
m['replicated_2of3']=(m.n_cohorts_p05>=2)&(np.abs(sign.sum(1))>=1)
m.sort_values('meta3_p').to_csv('META3_SEQC_NRC_KOCAK.csv',index=False)
print('adjusted FDR<0.05 per cohort: SEQC %d, NRC %d, KOCAK %d'%((m.SEQC_fdr_adj<0.05).sum(),(m.NRC_fdr_adj<0.05).sum(),(m.KOCAK_fdr_adj<0.05).sum()))
print('nominal p<0.05 in all three + same direction:',m.replicated_3of3.sum())
print('three-cohort meta FDR<0.05:',(m.meta3_fdr<0.05).sum())
print(m.nsmallest(20,'meta3_p')[['gene','SEQC_z_adj','NRC_z_adj','KOCAK_z_adj','meta3_z','meta3_fdr','n_cohorts_p05','all_same_direction']].round(3).to_string(index=False))
print('\nconsensus genes (p<0.05 all three, same direction):')
print(m[m.replicated_3of3][['gene','SEQC_z_adj','SEQC_p_adj','NRC_z_adj','NRC_p_adj','KOCAK_z_adj','KOCAK_p_adj','meta3_fdr']].round(4).to_string(index=False))
print('\ncheck genes:')
ck=['TRIP13','KIF2C','CCNB2','CDIN1','PSTK','ALG2','ARK2C','LPIN2','UBE2Q1','REEP4','MYCN']
print(m[m.gene.isin(ck)][['gene','SEQC_z_uni','KOCAK_z_uni','SEQC_z_adj','NRC_z_adj','KOCAK_z_adj','meta3_z','meta3_fdr','n_cohorts_p05']].round(3).to_string(index=False))
