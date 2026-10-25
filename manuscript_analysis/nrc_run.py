import gzip, numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from fastcox import score_test, score_test_offset
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
F='/mnt/user-data/uploads/GSE85047_series_matrix_txt_1.gz'
rows=[]
for l in gzip.open(F,'rt',errors='replace'):
    p=[x.strip('"') for x in l.rstrip('\n').split('\t')]
    if p[0]=='!Sample_characteristics_ch1': rows.append(p[1:])
    elif p[0]=='!Sample_geo_accession': gsm=p[1:]
    elif p[0]=='!series_matrix_table_begin': break
c=pd.DataFrame({'gsm':gsm})
for r in rows:
    for i,cell in enumerate(r):
        if ':' in cell:
            k,v=cell.split(':',1); c.loc[i,k.strip()]=v.strip()
num=lambda k: pd.to_numeric(c[k],errors='coerce')
cl=pd.DataFrame({'gsm':c.gsm,'age_days':num('age_at_diagnosis(days)'),
 'efs_t':num('profression_free_survival_time(days)'),
 'efs_e':c['event_progression_free'].str.strip().str.lower().map({'yes':1.,'no':0.}),
 'os_t':num('overall_survival_time(days)'),
 'os_e':c['event_overall'].str.strip().str.lower().map({'yes':1.,'no':0.}),
 'stage':c['inss'].astype(str).str.strip(),'mycn_raw':c['mycn_amplification'].astype(str).str.strip()})
cl['age18']=(cl.age_days>=547).astype(float)
st=cl.stage.str.lower().str.replace('st','',regex=False)
cl['stage4']=st.eq('4').astype(float).where(st.isin(['1','2','3','4','4s']))
m=cl.mycn_raw.str.lower()
cl['mycn']=np.select([m.str.startswith('non')|m.eq('0')|m.eq('no'),m.str.startswith('amp')|m.eq('1')|m.eq('yes')],[0.,1.],np.nan)
print('MYCN values:',cl.mycn_raw.value_counts().to_dict())
print('stage:',cl.stage.value_counts().to_dict())
print('n',len(cl),'EFS events',cl.efs_e.sum(),'OS events',cl.os_e.sum(),'MYCN amp',cl.mycn.sum(),'stage4',cl.stage4.sum())
cl.to_csv('NRC_clinical_harmonized.csv',index=False)
e=pd.read_csv(F,sep='\t',skiprows=lambda i: False, comment='!',index_col=0)
e=e.dropna(how='all'); e.columns=[x.strip('"') for x in e.columns]
e=e[[g for g in cl.gsm if g in e.columns]]
print('expr',e.shape,'range',round(np.nanmin(e.values),2),round(np.nanmax(e.values),2))
e=e[e.mean(1)>e.mean(1).quantile(.25)]
Z=e.sub(e.mean(1),axis=0).div(e.std(1).replace(0,np.nan),axis=0).dropna()
d=cl.set_index('gsm').loc[Z.columns].copy()
print('after filter',Z.shape)
def screen(Zs,ds,dur,ev,tag):
    k=ds[[dur,ev]].dropna().index; Zs=Zs[k]; ds=ds.loc[k]
    z,_,_=score_test(Zs.values,ds[dur].values,ds[ev].values)
    o=pd.DataFrame({'probe':Zs.index,'z':z,'p':2*norm.sf(abs(z))})
    o['fdr']=multipletests(o.p.fillna(1),method='fdr_bh')[1]
    o['direction']=np.where(o.z>0,'high = worse','high = better')
    o.sort_values('p').to_csv(tag,index=False); print(tag,'n',len(ds),'events',int(ds[ev].sum()),'FDR<0.05:',(o.fdr<0.05).sum()); return o
uni=screen(Z,d,'efs_t','efs_e','NRC_EFS_univariable_all.csv')
# proliferation axis = PC1 of EFS-significant probes
cand=uni.loc[uni.fdr<0.05,'probe'].tolist()
sub=Z.loc[cand].values
_,s_,vt=np.linalg.svd(sub-sub.mean(1,keepdims=True),full_matrices=False)
pc=pd.Series(vt[0],index=Z.columns)
if spearmanr(pc,d.mycn.fillna(0)).correlation<0: pc=-pc
d['pc']=(pc-pc.mean())/pc.std()
print('candidates',len(cand),'PC1 var %.1f%%'%(100*s_[0]**2/(s_**2).sum()),'rho(PC1,MYCN)=%.2f'%spearmanr(d.pc,d.mycn.fillna(0)).correlation)
def adj(dur,ev,covs,tag):
    x=d.dropna(subset=covs+[dur,ev]); Zs=Z[x.index]
    base=CoxPHFitter(penalizer=1e-4).fit(x[covs+[dur,ev]],dur,ev)
    w=np.exp(base.predict_log_partial_hazard(x).values)
    z,_,_=score_test_offset(Zs.values,x[dur].values,x[ev].values,w)
    o=pd.DataFrame({'probe':Zs.index,'z_adj':z,'p_adj':2*norm.sf(abs(z))})
    o['fdr_adj']=multipletests(o.p_adj.fillna(1),method='fdr_bh')[1]
    o['rho_PC1']=[spearmanr(Zs.loc[g],x.pc).correlation for g in Zs.index]
    o['c_base']=base.concordance_index_
    o['direction']=np.where(o.z_adj>0,'high = worse','high = better')
    o.sort_values('p_adj').to_csv(tag,index=False)
    print(tag,'covs',covs,'n',len(x),'events',int(x[ev].sum()),'FDR<0.05:',(o.fdr_adj<0.05).sum()); return o
a_full=adj('efs_t','efs_e',['age18','stage4','mycn','pc'],'NRC_EFS_adjusted_age_stage_MYCN_PC1.csv')
a_nopc=adj('efs_t','efs_e',['age18','stage4','mycn'],'NRC_EFS_adjusted_age_stage_MYCN.csv')
os_=adj('os_t','os_e',['age18','stage4','mycn','pc'],'NRC_OS_adjusted_age_stage_MYCN_PC1.csv')
nonamp=d.mycn.eq(0)
print('MYCN non-amplified n',int(nonamp.sum()))
screen(Z[d.index[nonamp]],d[nonamp],'efs_t','efs_e','NRC_EFS_univariable_MYCNnonamp.csv')
print(a_full.head(12).round(4).to_string(index=False))
