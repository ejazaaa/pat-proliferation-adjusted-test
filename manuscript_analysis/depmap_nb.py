import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
U='/mnt/user-data/uploads/'
m=pd.read_csv(U+'Model.csv')
nbids=set(m.loc[m.OncotreePrimaryDisease.eq('Neuroblastoma'),'ModelID'])
name=dict(zip(m.ModelID,m.CellLineName))
def load(f):
    d=pd.read_csv(U+f,index_col=0,dtype={0:str}).astype('float32')
    d.columns=[c.split(' (')[0] for c in d.columns]; return d
eff=load('CRISPRGeneEffect.csv'); prob=load('CRISPRGeneDependency.csv')
ce=set(pd.read_csv(U+'CRISPRInferredCommonEssentials.csv').iloc[:,0].str.split(' (',regex=False).str[0])
ex=pd.read_csv(U+'OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv',index_col=0)
ex=ex[ex.IsDefaultEntryForModel.eq('Yes')].set_index('ModelID').iloc[:,4:].astype('float32')
ex.columns=[c.split(' (')[0] for c in ex.columns]
ex=ex.loc[:,~ex.columns.duplicated()]
nb=[i for i in eff.index if i in nbids]; ot=[i for i in eff.index if i not in nbids]
print('CRISPR models: NB',len(nb),'other',len(ot),'genes',eff.shape[1])
E_nb,E_ot=eff.loc[nb],eff.loc[ot]
P_nb,P_ot=prob.reindex(nb),prob.reindex(ot)
rows=[]
nbex=[i for i in nb if i in ex.index]
for g in eff.columns:
    a=E_nb[g].dropna(); b=E_ot[g].dropna()
    if len(a)<10: continue
    p=mannwhitneyu(a,b,alternative='less').pvalue
    r=np.nan
    if g in ex.columns and len(nbex)>=10:
        x=ex.loc[nbex,g]; y=eff.loc[nbex,g]; k=x.notna()&y.notna()
        if x[k].std()>0: r=spearmanr(x[k],y[k]).correlation
    rows.append(dict(gene=g,nb_n=len(a),nb_mean_chronos=a.mean(),nb_median_chronos=a.median(),
      nb_frac_chronos_lt_neg0_5=(a<-0.5).mean(),nb_frac_prob_gt_0_5=(P_nb[g]>0.5).mean() if g in prob.columns else np.nan,
      other_mean_chronos=b.mean(),other_frac_prob_gt_0_5=(P_ot[g]>0.5).mean() if g in prob.columns else np.nan,
      delta_nb_minus_other=a.mean()-b.mean(),mw_p_nb_more_dependent=p,common_essential=g in ce,
      nb_mean_expr_log2tpm=ex.loc[nbex,g].mean() if g in ex.columns else np.nan,
      nb_spearman_expr_vs_chronos=r))
res=pd.DataFrame(rows)
res['mw_fdr']=multipletests(res.mw_p_nb_more_dependent,method='fdr_bh')[1]
res['passes_funnel']=(res.nb_mean_chronos<=-0.6)&(res.nb_frac_prob_gt_0_5>=0.4)&(res.delta_nb_minus_other<=-0.25)&(res.mw_fdr<=0.05)&(~res.common_essential)
res.sort_values('delta_nb_minus_other').to_csv('NB_DepMap26Q1_genomewide_dependency.csv',index=False)
print('funnel survivors',res.passes_funnel.sum())
print(res[res.passes_funnel].sort_values('delta_nb_minus_other')[['gene','nb_mean_chronos','nb_frac_prob_gt_0_5','other_frac_prob_gt_0_5','delta_nb_minus_other','mw_fdr']].round(3).to_string(index=False))
print(res[res.gene.isin(['TRIP13','CDIN1','PSTK','TUBD1','CCNB2','KIF2C','MYCN','PHOX2B','HAND2','GATA3','ISL1'])].round(3).to_string(index=False))
# ADRN/MES
sig=pd.read_excel(U+'41588_2017_BFng3899_MOESM3_ESM.xlsx',header=None,skiprows=2).dropna()
sig.columns=['gene','state']
adrn=[g for g in sig.gene[sig.state=='ADRN'] if g in ex.columns]; mes=[g for g in sig.gene[sig.state=='MES'] if g in ex.columns]
print('signature genes found ADRN',len(adrn),'/',(sig.state=='ADRN').sum(),'MES',len(mes),'/',(sig.state=='MES').sum())
nbx=ex.loc[[i for i in ex.index if i in nbids]]
z=(nbx-nbx.mean())/nbx.std().replace(0,np.nan)
st=pd.DataFrame({'ModelID':nbx.index,'CellLineName':[name.get(i) for i in nbx.index],
 'ADRN_score':z[adrn].mean(1).values,'MES_score':z[mes].mean(1).values,
 'MYCN_log2TPM':nbx['MYCN'].values,'PHOX2B_log2TPM':nbx['PHOX2B'].values,'PRRX1_log2TPM':nbx['PRRX1'].values,'TRIP13_log2TPM':nbx['TRIP13'].values,
 'in_CRISPR':[i in nb for i in nbx.index]})
st['ADRN_minus_MES']=st.ADRN_score-st.MES_score
st['state_call']=np.where(st.ADRN_minus_MES>0.5,'ADRN',np.where(st.ADRN_minus_MES<-0.5,'MES','Mixed'))
st=st.sort_values('ADRN_minus_MES',ascending=False)
st.to_csv('NB_DepMap_lines_ADRN_MES_scores.csv',index=False)
print(st.round(2).to_string(index=False))
# state-selective dependencies
ad=[i for i in st.ModelID[st.state_call=='ADRN'] if i in nb]; me=[i for i in st.ModelID[st.state_call!='ADRN'] if i in nb]
print('CRISPR ADRN',len(ad),'MES/mixed',len(me))
rows=[]
for g in eff.columns:
    a=eff.loc[ad,g].dropna(); b=eff.loc[me,g].dropna()
    if len(a)<5 or len(b)<3: continue
    rows.append(dict(gene=g,adrn_mean=a.mean(),mes_mixed_mean=b.mean(),delta=a.mean()-b.mean(),p=mannwhitneyu(a,b).pvalue))
sd=pd.DataFrame(rows); sd['fdr']=multipletests(sd.p,method='fdr_bh')[1]
sd['common_essential']=sd.gene.isin(ce)
sd.sort_values('p').to_csv('NB_ADRN_vs_MES_dependency.csv',index=False)
print(sd.sort_values('p').head(15).round(4).to_string(index=False))
