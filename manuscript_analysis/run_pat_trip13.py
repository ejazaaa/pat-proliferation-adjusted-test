import numpy as np, pandas as pd, sys
sys.path.insert(0,'/home/claude/nb')
from pat_core import pat, verdict
U='/mnt/user-data/uploads/'
def collapse(e,mp):
    idx=pd.Index(e.index.astype(str)).map(mp); e=e[idx.notna()]; e.index=idx[idx.notna()]
    return e.groupby(level=0).max()
r2s=pd.read_csv('refseq2symbol.csv',index_col=0).symbol
# SEQC
cl=pd.read_csv('SEQC_clinical_harmonized.csv')
e=pd.read_csv(U+'GSE62564_SEQC_NB_RNA-Seq_log2RPM_txt.gz',sep='\t',index_col=0)
e=e[cl.title.tolist()]; e.columns=cl.gsm.tolist(); e.index=[str(i).split('.')[0] for i in e.index]
SEQC=(collapse(e,r2s), cl.set_index('gsm'), ['age18','highrisk'])
# NRC
n=pd.read_csv(U+'GSE85047_series_matrix_txt_1.gz',sep='\t',comment='!',index_col=0); n.columns=[c.strip('"') for c in n.columns]
cn=pd.read_csv('NRC_clinical_harmonized.csv'); n=n[[g for g in cn.gsm if g in n.columns]]
NRC=(collapse(n,pd.read_csv('gpl5175_probe2symbol.csv',index_col=0,dtype={'id':str}).symbol), cn.set_index('gsm'), ['age18','stage4','mycn'])
# Kocak
k=pd.read_csv('KOCAK_clinical_harmonized.csv')
ek=pd.read_csv(U+'GSE45547_series_matrix_txt_1.gz',sep='\t',comment='!',index_col=0,low_memory=False)
ek.columns=[c.strip('"') for c in ek.columns]; ek=np.log2(ek[[g for g in k.gsm if g in ek.columns]].clip(lower=1))
KOCAK=(collapse(ek,pd.read_csv('gpl16876_probe2symbol.csv',index_col=0,dtype={'id':str}).symbol), k.set_index('gsm'), ['age18','stage4','mycn'])
cohorts={'SEQC':SEQC,'NRC':NRC,'Kocak':KOCAK}
for nm,(X,_,_) in cohorts.items():
    print(nm,'MKI67 present:', 'MKI67' in X.index, '| CCNB2:', 'CCNB2' in X.index)
rows=[]
for gene in ['TRIP13','CCNB2','MKI67','EPN2','FERMT2']:
    for nm,(X,C,cov) in cohorts.items():
        for ep in ['efs','os']:
            for mode in ['CCNB2','MKI67','score']:
                r=pat(X,C.loc[[s for s in C.index if s in X.columns]],gene,endpoint=ep,prolif_mode=mode,covariates=cov)
                r['cohort']=nm; rows.append(r)
res=pd.DataFrame(rows); res.to_csv('PAT_results_all.csv',index=False)
t=res[(res.gene=='TRIP13')&(res.status=='ok')]
cols=['cohort','endpoint','prolif_covariate','n','events','hr_univariable','p_univariable','hr_adjusted','ci_low','ci_high','p_adjusted','p_lrt','delta_c','delta_aic','rho_prolif']
print(t[cols].round(4).to_string(index=False))
print()
for g in ['TRIP13','EPN2','FERMT2','CCNB2']:
    rr=res[(res.gene==g)&(res.prolif_covariate=='CCNB2')].to_dict('records')
    v,why=verdict(rr); print('%-8s %-26s %s'%(g,v,why))
print()
print(res[res.status!='ok'][['gene','cohort','endpoint','prolif_covariate','status']].drop_duplicates().to_string(index=False))
