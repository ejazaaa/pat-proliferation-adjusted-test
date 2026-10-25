import numpy as np, pandas as pd
ch=pd.read_csv('published_signatures_tested.csv')          # Chicco 10 signatures
cg=pd.read_csv('chatgpt_signatures_tested.csv')            # 8 further studies (verified)
hub=pd.read_csv('cytoHubba_MCODE_tested.csv')              # 5 cytoHubba methods + MCODE
mit=pd.read_csv('mitotic_class_tested.csv')                # 40 mitotic genes
ch=ch[ch.measured==True].rename(columns={'signature':'source'})
cg=cg[cg.measured==True].rename(columns={'set':'source'})
ch['study_group']='Chicco 2023 compendium (10 signatures)'
cg['study_group']='Further published studies (8)'
pub=pd.concat([ch,cg],ignore_index=True)
pub.to_csv('ALL_published_signature_genes.csv',index=False)
u=pub.drop_duplicates('gene')
def summ(d,label):
    n=len(d); uni=(d.z_uni.abs()>1.96).sum(); adj=(d.z_adj.abs()>1.96).sum(); worse=(d.z_adj>1.96).sum()
    print('%-46s n=%3d uni %3d (%2.0f%%) adj %3d (%2.0f%%) worse %3d (%2.0f%%) med|rho| %.2f'%(
        label,n,uni,100*uni/n,adj,100*adj/n,worse,100*worse/n,d.rho_prolif.abs().median()))
    return dict(set=label,n=n,uni=uni,adj=adj,worse=worse,med_rho=round(d.rho_prolif.abs().median(),2))
rows=[summ(u,'All published prognostic genes (18 studies)'),
      summ(ch.drop_duplicates('gene'),'Chicco 2023 compendium (10 signatures)'),
      summ(cg.drop_duplicates('gene'),'Further published studies (8)'),
      summ(hub[hub.z_uni.notna()].drop_duplicates('gene'),'Hub genes, cytoHubba + MCODE (this study)'),
      summ(mit,'Mitotic / cell-cycle gene class (40)')]
pd.DataFrame(rows).to_csv('TableX_summary_by_source.csv',index=False)
# per-method hub detail
print()
for m in hub.method.dropna().unique():
    s=hub[(hub.method==m)&hub.z_uni.notna()]
    print('%-22s n=%2d uni %2d adj %d (worse %d) med|rho| %.2f'%(m,len(s),(s.z_uni.abs()>1.96).sum(),(s.z_adj.abs()>1.96).sum(),(s.z_adj>1.96).sum(),s.rho_prolif.abs().median()))
# three most reproducible genes
print()
print(pub[pub.gene.isin(['AHCY','DPYSL3','NME1'])].drop_duplicates('gene')[['gene','z_uni','z_adj','p_adj','rho_prolif']].round(3).to_string(index=False))
