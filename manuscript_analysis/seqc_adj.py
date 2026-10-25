import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from fastcox import score_test, score_test_offset
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
clin=pd.read_csv('SEQC_clinical_harmonized.csv')
e=pd.read_csv(U+'GSE62564_SEQC_NB_RNA-Seq_log2RPM_txt.gz',sep='\t',index_col=0)
e=e[clin.title.tolist()]; e.columns=clin.gsm.tolist()
e=e[e.mean(1)>e.mean(1).quantile(.25)]
Z=e.sub(e.mean(1),axis=0).div(e.std(1).replace(0,np.nan),axis=0).dropna()
d=clin.set_index('gsm').loc[Z.columns].copy()
uni=pd.read_csv('SEQC_EFS_univariable_all.csv')
pc=pd.Series(np.load('SEQC_PC1.npy'),index=Z.columns) if False else None
cand=uni.loc[uni.fdr<0.05,'refseq'].tolist()
sub=Z.loc[cand].values
u_,s_,vt=np.linalg.svd(sub-sub.mean(1,keepdims=True),full_matrices=False)
pc=pd.Series(vt[0],index=Z.columns)
if spearmanr(pc,d.highrisk).correlation<0: pc=-pc
d['pc']=(pc-pc.mean())/pc.std()
print('candidates',len(cand),'rho(PC1,HR)=%.2f'%spearmanr(d.pc,d.highrisk).correlation)

def adj_screen(dur,ev,covs,tag,rows=None):
    x=d.dropna(subset=covs+[dur,ev]); Zs=Z.loc[rows if rows is not None else Z.index, x.index]
    base=CoxPHFitter(penalizer=1e-4).fit(x[covs+[dur,ev]],dur,ev)
    w=np.exp(base.predict_log_partial_hazard(x).values)
    z,_,_=score_test_offset(Zs.values,x[dur].values,x[ev].values,w)
    o=pd.DataFrame({'refseq':Zs.index,'z_adj':z,'p_adj':2*norm.sf(abs(z))})
    o['fdr_adj']=multipletests(o.p_adj.fillna(1),method='fdr_bh')[1]
    o['rho_PC1']=[spearmanr(Zs.loc[g],x.pc).correlation for g in Zs.index]
    o['c_base']=base.concordance_index_
    o=o.merge(uni[['refseq','z','p','fdr']].rename(columns={'z':'z_uni','p':'p_uni','fdr':'fdr_uni'}),on='refseq',how='left')
    o['direction']=np.where(o.z_adj>0,'high = worse','high = better')
    o.sort_values('p_adj').to_csv(tag,index=False)
    print(tag,'covs',covs,'n',len(x),'events',int(x[ev].sum()),'FDR<0.05:',(o.fdr_adj<0.05).sum())
    return o
a1=adj_screen('efs_t','efs_e',['age18','highrisk'],'SEQC_EFS_adjusted_age_HR.csv')
a2=adj_screen('efs_t','efs_e',['age18','highrisk','pc'],'SEQC_EFS_adjusted_age_HR_PC1.csv')
a3=adj_screen('os_t','os_e',['age18','highrisk','pc'],'SEQC_OS_adjusted_age_HR_PC1.csv')
top=a2.nsmallest(300,'p_adj').refseq.tolist()
out=[]
for g in top:
    x=d.copy(); x['x']=Z.loc[g,x.index].values
    f=CoxPHFitter(penalizer=1e-4).fit(x[['x','age18','highrisk','pc','efs_t','efs_e']],'efs_t','efs_e')
    out.append(dict(refseq=g,hr_adj=np.exp(f.params_['x']),p_full=f.summary.p['x'],c_full=f.concordance_index_))
pd.DataFrame(out).merge(a2,on='refseq').sort_values('p_full').to_csv('SEQC_EFS_top300_fullfit.csv',index=False)
print('top 10 by full fit:')
print(pd.DataFrame(out).merge(a2,on='refseq').sort_values('p_full').head(10)[['refseq','hr_adj','p_full','rho_PC1','c_base','c_full','fdr_uni']].round(4).to_string(index=False))
