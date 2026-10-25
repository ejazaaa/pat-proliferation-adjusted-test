import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
# verified: Zhong2018 4-gene matches Chicco S1.2 exactly; Xia2021 5-gene matches publisher abstract;
# Ma2022 5-gene matches publisher abstract; Tang2025 confirmed (RFC3 present). Others VERIFY BEFORE PUBLICATION.
SETS={
'Zhong 2018 candidates (10)':"NCAN,AHCY,STK33,UBE2S,EXO1,ERCC6L,SLCO4A1,CKS2,MAD2L1,MND1",
'Zhong 2018 final (4)':"ERCC6L,AHCY,STK33,NCAN",
'Wang 2020 PPI hubs (10)':"CD69,CD3D,CD2,CCL5,CCR2,CD48,LCK,IL7R,IRF8,HLA-DRB1",
'Wang 2020 prognostic (14)':"ABCA6,SELENOP,SLAMF8,GPR171,ABCA9,ARHGAP15,IL7R,HLA-DPB1,GZMA,GPR183,CCL19,ITK,FGL2,CD1C",
'Wang 2020 independent (5)':"ARHGAP15,ABCA9,CCL19,SLAMF8,CD1C",
'Xia 2021 PPI hubs (10)':"MTHFD2,AHCY,EXO1,CA10,CNTN1,NCAN,NRCAM,PHGDH,PTN,GAP43",
'Xia 2021 final (5)':"CPLX3,GDPD5,SPAG6,NXPH1,AHI1",
'Song 2022 OS panel (10)':"OPRK1,RIPK2,NLRP3,KIF1B,HIF1A,MEP1A,HRH1,BEST1,LY6E,GABBR1",
'Song 2022 EFS panel (11)':"GNAI3,PTGER2,OPRK1,RIPK2,NLRP3,KIF1B,HIF1A,ACVR2A,ATP2B1,LPAR1,GABBR1",
'Ma 2022 candidates (12)':"CRABP1,AMH,CHGA,PTGER1,SHC3,GAL,MAPT,PLXNC1,SCG2,LIFR,TRIM55,OPTN",
'Ma 2022 final (5)':"CHGA,PTGER1,SHC3,PLXNC1,TRIM55",
'Yang 2022 final (10)':"ABCA2,B3GALT4,NEU4,ST3GAL1,ST3GAL3,ST6GALNAC4,ST6GALNAC5,ST6GALNAC6,ST8SIA2,ST8SIA3",
'Wang 2024 final (5)':"GAL,GFRA3,MARCKS,PSMD13,ZNHIT3",
'Tang 2025 hub/prognostic (11)':"AURKA,BLM,BRCA1,BRCA2,CCNA2,CHEK1,E2F1,MAD2L1,PLK1,RAD51,RFC3"}
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
for sig,g in SETS.items():
    for gene in [x.strip() for x in g.split(',')]:
        if gene not in S.index: rows.append({'set':sig,'gene':gene,'measured':False}); continue
        pset=[x for x in PROLIF if x in S.index and x!=gene]
        x=d.copy(); x['x']=z(S.loc[gene,x.index]); x['prolif']=z(S.loc[pset].apply(z,axis=1).mean())
        x=x.dropna(subset=['x','prolif','efs_t','efs_e','age18','highrisk'])
        u=CoxPHFitter(penalizer=1e-4).fit(x[['x','efs_t','efs_e']],'efs_t','efs_e')
        a=CoxPHFitter(penalizer=1e-4).fit(x[['x','age18','highrisk','prolif','efs_t','efs_e']],'efs_t','efs_e')
        rows.append({'set':sig,'gene':gene,'measured':True,'z_uni':u.summary.z['x'],'p_uni':u.summary.p['x'],
                     'z_adj':a.summary.z['x'],'p_adj':a.summary.p['x'],'rho_prolif':spearmanr(x.x,x.prolif).correlation})
r=pd.DataFrame(rows); r.to_csv('chatgpt_signatures_tested.csv',index=False)
t=r[r.measured==True]
print('%-32s %5s %5s %5s %6s %8s'%('set','meas','uni','adj','worse','med|rho|'))
for s in SETS:
    q=t[t.set==s]
    if len(q)==0: continue
    print('%-32s %5d %5d %5d %6d %8.2f'%(s,len(q),(q.z_uni.abs()>1.96).sum(),(q.z_adj.abs()>1.96).sum(),(q.z_adj>1.96).sum(),q.rho_prolif.abs().median()))
u=t.drop_duplicates('gene')
print('\nUNION %d unique genes: uni sig %d (%.0f%%), adj sig %d (%.0f%%), adj toward worse %d (%.0f%%)'%(
  len(u),(u.z_uni.abs()>1.96).sum(),100*(u.z_uni.abs()>1.96).mean(),
  (u.z_adj.abs()>1.96).sum(),100*(u.z_adj.abs()>1.96).mean(),(u.z_adj>1.96).sum(),100*(u.z_adj>1.96).mean()))
# combine with Chicco 10 signatures
ch=pd.read_csv('published_signatures_tested.csv'); ch=ch[ch.measured==True]
allg=pd.concat([ch[['gene','z_uni','z_adj','rho_prolif']],t[['gene','z_uni','z_adj','rho_prolif']]]).drop_duplicates('gene')
print('\nGRAND TOTAL across 10 Chicco signatures + 8 further studies: %d unique genes'%len(allg))
print('  univariable |z|>1.96: %d (%.0f%%)'%((allg.z_uni.abs()>1.96).sum(),100*(allg.z_uni.abs()>1.96).mean()))
print('  adjusted    |z|>1.96: %d (%.0f%%)'%((allg.z_adj.abs()>1.96).sum(),100*(allg.z_adj.abs()>1.96).mean()))
print('  adjusted toward worse outcome: %d (%.0f%%)'%((allg.z_adj>1.96).sum(),100*(allg.z_adj>1.96).mean()))
allg.to_csv('ALL_published_genes_tested.csv',index=False)
