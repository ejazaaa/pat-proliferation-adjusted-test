import gzip, numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from fastcox import score_test
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
d=clin.set_index('gsm').loc[Z.columns]
print('genes',Z.shape[0],'samples',Z.shape[1])

def screen(Z,d,dur,ev,tag):
    z,_,_=score_test(Z.values,d[dur].values,d[ev].values)
    o=pd.DataFrame({'refseq':Z.index,'z':z,'p':2*norm.sf(abs(z))})
    o['fdr']=multipletests(o.p.fillna(1),method='fdr_bh')[1]
    o['direction']=np.where(o.z>0,'high = worse','high = better')
    o.sort_values('p').to_csv(tag,index=False); return o

uni=screen(Z,d,'efs_t','efs_e','SEQC_EFS_univariable_all.csv')
print('EFS univariable FDR<0.05:',(uni.fdr<0.05).sum())
uni_os=screen(Z,d,'os_t','os_e','SEQC_OS_univariable_all.csv')
print('OS univariable FDR<0.05:',(uni_os.fdr<0.05).sum())
hr=d.highrisk.eq(1); Zh=Z.loc[:,hr.values]
uni_hr=screen(Zh,d[hr.values],'efs_t','efs_e','SEQC_EFS_univariable_highriskonly.csv')
print('EFS within high-risk FDR<0.05:',(uni_hr.fdr<0.05).sum())

# proliferation factor = PC1 of EFS-significant genes
cand=uni.loc[uni.fdr<0.05,'refseq'].tolist()
sub=Z.loc[cand].values
u_,s_,vt=np.linalg.svd(sub-sub.mean(1,keepdims=True),full_matrices=False)
pc=pd.Series(vt[0],index=Z.columns)
if spearmanr(pc,d.highrisk).correlation<0: pc=-pc
print('candidates',len(cand),'PC1 var %.1f%%'%(100*s_[0]**2/(s_**2).sum()),'rho(PC1,highrisk)=%.2f'%spearmanr(pc,d.highrisk).correlation)
np.save('SEQC_PC1.npy',pc.values)

def adjusted(cand,d,Z,dur,ev,covs,tag):
    out=[]; base={}
    x0=d.copy(); x0['pc']= (pc.reindex(x0.index)-pc.mean())/pc.std()
    for g in cand:
        x=x0.copy(); x['x']=Z.loc[g,x.index].values
        x=x.dropna(subset=['x',dur,ev]+covs)
        r={'refseq':g,'n':len(x),'events':int(x[ev].sum()),'rho_PC1':spearmanr(x.x,x.pc).correlation}
        try:
            c=CoxPHFitter(penalizer=1e-4).fit(x[['x']+covs+[dur,ev]],dur,ev)
            k=len(x)
            if k not in base: base[k]=CoxPHFitter(penalizer=1e-4).fit(x[covs+[dur,ev]],dur,ev)
            r.update(hr_adj=np.exp(c.params_['x']),p_adj=c.summary.p['x'],
                     c_base=base[k].concordance_index_,c_full=c.concordance_index_)
        except Exception as ex: r['error']=str(ex)[:60]
        out.append(r)
    o=pd.DataFrame(out); o['fdr_adj']=multipletests(o.p_adj.fillna(1),method='fdr_bh')[1]
    o=o.merge(uni[['refseq','z','p','fdr']].rename(columns={'z':'z_uni','p':'p_uni','fdr':'fdr_uni'}),on='refseq',how='left')
    o.sort_values('p_adj').to_csv(tag,index=False); return o

adj=adjusted(cand,d,Z,'efs_t','efs_e',['age18','highrisk','pc'],'SEQC_EFS_adjusted_age_HR_PC1.csv')
print('adjusted (age+HR+PC1) FDR<0.05:',(adj.fdr_adj<0.05).sum())
print(adj.sort_values('p_adj').head(15)[['refseq','n','events','rho_PC1','hr_adj','p_adj','fdr_adj','c_base','c_full']].round(4).to_string(index=False))
