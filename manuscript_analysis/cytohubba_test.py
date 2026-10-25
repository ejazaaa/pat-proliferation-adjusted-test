import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
lists={}
for m in ['Degree','MCC','Closeness','EPC','MNC']:
    d=pd.read_csv(U+m+'.csv',skiprows=1)
    lists['cytoHubba '+m]=d.Name.tolist()
mc=pd.read_excel(U+'MCODE_cluster1_csv.xlsx',header=None)
mc.columns=['gene','gene2','score','status','cluster']
mcg=mc.loc[mc.cluster.astype(str).str.contains('Cluster 1'),'gene'].tolist()  # exclude Unclustered rows
lists['MCODE cluster 1']=mcg
for k,v in lists.items(): print(k,len(v),v)
PROLIF=['CCNB1','CCNB2','CDK1','CDC20','BUB1','AURKA','TOP2A','CENPF','BIRC5','RRM2']
r2s=pd.read_csv('refseq2symbol.csv',index_col=0).symbol
cl=pd.read_csv('SEQC_clinical_harmonized.csv')
e=pd.read_csv(U+'GSE62564_SEQC_NB_RNA-Seq_log2RPM_txt.gz',sep='\t',index_col=0)
e=e[cl.title.tolist()]; e.columns=cl.gsm.tolist(); e.index=[str(i).split('.')[0] for i in e.index]
idx=pd.Index(e.index).map(r2s); e=e[idx.notna()]; e.index=idx[idx.notna()]
S=e.groupby(level=0).max(); S=S[S.mean(1)>S.mean(1).quantile(.25)]
d=cl.set_index('gsm').loc[S.columns].copy()
z=lambda v:(v-v.mean())/v.std()
rows=[]
for method,genes in lists.items():
    for g in genes:
        if g not in S.index: rows.append({'method':method,'gene':g,'note':'not measured'}); continue
        pset=[x for x in PROLIF if x in S.index and x!=g]
        x=d.copy(); x['x']=z(S.loc[g,x.index]); x['prolif']=z(S.loc[pset].apply(z,axis=1).mean())
        x=x.dropna(subset=['x','prolif','efs_t','efs_e','age18','highrisk'])
        u=CoxPHFitter(penalizer=1e-4).fit(x[['x','efs_t','efs_e']],'efs_t','efs_e')
        a=CoxPHFitter(penalizer=1e-4).fit(x[['x','age18','highrisk','prolif','efs_t','efs_e']],'efs_t','efs_e')
        rows.append({'method':method,'gene':g,'z_uni':u.summary.z['x'],'p_uni':u.summary.p['x'],
                     'z_adj':a.summary.z['x'],'p_adj':a.summary.p['x'],'rho_prolif':spearmanr(x.x,x.prolif).correlation})
r=pd.DataFrame(rows); r.to_csv('cytoHubba_MCODE_tested.csv',index=False)
print(r.round(3).to_string(index=False))
print()
for m in lists:
    s=r[(r.method==m)&r.z_uni.notna()]
    worse=((s.z_adj>1.96)).sum(); better=((s.z_adj<-1.96)).sum()
    print('%-20s n=%2d | uni sig %2d | adj sig %d (worse %d, reversed %d) | median |rho| %.2f'%(
        m,len(s),(s.z_uni.abs()>1.96).sum(),(s.z_adj.abs()>1.96).sum(),worse,better,s.rho_prolif.abs().median()))
u=r.dropna(subset=['z_uni']).drop_duplicates('gene')
print('\nUNION of all cytoHubba/MCODE nominations: %d unique genes; uni sig %d; adj sig %d (worse %d)'%(
    len(u),(u.z_uni.abs()>1.96).sum(),(u.z_adj.abs()>1.96).sum(),(u.z_adj>1.96).sum()))
