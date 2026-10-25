import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
net=pd.read_csv(U+'string_interactions_short_1405_nodes.tsv',sep='\t')
net.columns=[c.lstrip('#') for c in net.columns]
print('edges',len(net),'nodes',len(set(net.node1)|set(net.node2)))
import networkx as nx
G=nx.Graph()
for a,b,w in zip(net.node1,net.node2,net.combined_score): G.add_edge(a,b,weight=w)
deg=pd.Series(dict(G.degree())).sort_values(ascending=False)
clo=pd.Series(nx.closeness_centrality(G)).sort_values(ascending=False)
bet=pd.Series(nx.betweenness_centrality(G,k=min(500,G.number_of_nodes()),seed=1)).sort_values(ascending=False)
mcc={}
# MCC approximation: sum over maximal cliques containing v of (|C|-1)!
for c in nx.find_cliques(G):
    f=__import__("math").factorial(len(c)-1) if len(c)<=12 else float('inf')
    for v in c: mcc[v]=mcc.get(v,0)+ (f if f!=float('inf') else 1e12)
mcc=pd.Series(mcc).sort_values(ascending=False)
hubs={'Degree':deg.head(12).index.tolist(),'MCC':mcc.head(12).index.tolist(),'Closeness':clo.head(12).index.tolist(),'Betweenness':bet.head(12).index.tolist()}
for k,v in hubs.items(): print(k,v)
pd.DataFrame({'gene':deg.index,'degree':deg.values}).to_csv('STRING_degree.csv',index=False)
pd.DataFrame(dict([(k,pd.Series(v)) for k,v in hubs.items()])).to_csv('STRING_hub_lists.csv',index=False)
# test them
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
for method,genes in hubs.items():
    for g in genes:
        if g not in S.index: rows.append({'method':method,'gene':g,'note':'not measured'}); continue
        pset=[x for x in PROLIF if x in S.index and x!=g]
        x=d.copy(); x['x']=z(S.loc[g,x.index]); x['prolif']=z(S.loc[pset].apply(z,axis=1).mean())
        x=x.dropna(subset=['x','prolif','efs_t','efs_e','age18','highrisk'])
        u=CoxPHFitter(penalizer=1e-4).fit(x[['x','efs_t','efs_e']],'efs_t','efs_e')
        a=CoxPHFitter(penalizer=1e-4).fit(x[['x','age18','highrisk','prolif','efs_t','efs_e']],'efs_t','efs_e')
        rows.append({'method':method,'gene':g,'degree':int(deg.get(g,0)),'z_uni':u.summary.z['x'],'p_uni':u.summary.p['x'],
                     'z_adj':a.summary.z['x'],'p_adj':a.summary.p['x'],'rho_prolif':spearmanr(x.x,x.prolif).correlation})
r=pd.DataFrame(rows); r.to_csv('STRING_hubs_tested.csv',index=False)
print(r.round(3).to_string(index=False))
for m in hubs:
    s=r[(r.method==m)&r.z_uni.notna()]
    print('%-12s n=%2d uni sig %2d adj sig %d  median |rho| %.2f'%(m,len(s),(s.z_uni.abs()>1.96).sum(),(s.z_adj.abs()>1.96).sum(),s.rho_prolif.abs().median()))
