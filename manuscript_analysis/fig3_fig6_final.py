import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, networkx as nx
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.8,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':600})
B,R,G,GY='#2c5f8a','#c0504d','#4f8a3d','#cccccc'
U='/mnt/user-data/uploads/'
f=pd.read_csv('FULLFIT_consensus_and_index.csv').set_index('gene')
m=pd.read_csv('META3_SEQC_NRC_KOCAK.csv').set_index('gene')
# ================= FIGURE 3 : index genes + survival curves =================
fig=plt.figure(figsize=(7.1,5.6))
gs=fig.add_gridspec(2,2,hspace=.45,wspace=.3)
ax=fig.add_subplot(gs[0,0])
labels=['TRIP13 alone','+ age','+ high-risk','+ proliferation']
hr=np.array([2.38,2.04,1.72,1.40]); lo=np.array([1.99,1.71,1.41,1.03]); hi=np.array([2.83,2.43,2.10,1.90])
y=np.arange(4)[::-1]
ax.errorbar(hr,y,xerr=[hr-lo,hi-hr],fmt='o',color=B,ms=5,capsize=3,lw=1.2)
ax.axvline(1,color='k',ls='--',lw=.8); ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=7.5)
ax.set_xlabel('Hazard ratio per SD (SEQC, EFS)'); ax.set_xlim(.9,3.0)
ax.set_title('A  TRIP13, nested models',loc='left',fontweight='bold',fontsize=9)
ax2=fig.add_subplot(gs[0,1])
genes=['TRIP13','CCNB2']; x=np.arange(2); w=.16
series=[('SEQC univariable',[m.loc[g,'SEQC_z_uni'] for g in genes],B,.4),
        ('Kocak univariable',[m.loc[g,'KOCAK_z_uni'] for g in genes],G,.4),
        ('SEQC adjusted',[f.loc[g,'SEQC_z'] for g in genes],B,1.0),
        ('NRC adjusted',[f.loc[g,'NRC_z'] for g in genes],R,1.0),
        ('Kocak adjusted',[f.loc[g,'Kocak_z'] for g in genes],G,1.0)]
for i,(lab,v,col,a) in enumerate(series): ax2.bar(x+(i-2)*w,v,w*.9,color=col,alpha=a,label=lab,edgecolor=col,lw=.4)
for v in (1.96,-1.96): ax2.axhline(v,color='grey',ls=':',lw=.8)
ax2.axhline(0,color='k',lw=.8); ax2.text(1.42,2.2,'p = 0.05',fontsize=6,color='grey')
ax2.set_xticks(x); ax2.set_xticklabels(genes); ax2.set_ylabel('Cox z statistic'); ax2.set_ylim(-3,12); ax2.set_xlim(-.6,1.6)
ax2.legend(frameon=False,fontsize=6,ncol=2,loc='upper center')
ax2.set_title('B  Index genes before and after adjustment',loc='left',fontweight='bold',fontsize=9)
km=pd.read_csv('KM_TRIP13_SEQC.csv',index_col=0)
for j,(sub,ttl,note) in enumerate([
    (km,'C  TRIP13, all patients','adjusted p = 0.029'),
    (km[km.grp.notna()].assign(_hr=pd.read_csv('SEQC_clinical_harmonized.csv').set_index('gsm').highrisk.reindex(km.index)).query('_hr==1'),
     'D  TRIP13, high-risk patients only','adjusted p = 0.18')]):
    a=fig.add_subplot(gs[1,j]); k=KaplanMeierFitter()
    for grp,col in [('high',R),('low',B)]:
        s=sub[sub.grp==grp]; k.fit(s.efs_t/365.25,s.efs_e,label=f'{grp} (n={len(s)})'); k.plot_survival_function(ax=a,color=col,ci_show=True,lw=1.4)
    lr=logrank_test(sub[sub.grp=='high'].efs_t,sub[sub.grp=='low'].efs_t,sub[sub.grp=='high'].efs_e,sub[sub.grp=='low'].efs_e)
    a.text(.04,.10,'log-rank p = %.2g\n%s'%(lr.p_value,note),transform=a.transAxes,fontsize=6.8)
    a.set_xlabel('Years'); a.set_ylabel('Event-free survival'); a.set_ylim(0,1); a.legend(frameon=False,fontsize=6.5,loc='upper right')
    a.set_title(ttl,loc='left',fontweight='bold',fontsize=9)
fig.savefig('Fig3.png',bbox_inches='tight'); plt.close(fig)

# ================= FIGURE 6 : six panels incl. network =================
hub=pd.read_csv('cytoHubba_MCODE_tested.csv'); hub=hub[hub.z_uni.notna()]
mit=pd.read_csv('mitotic_class_tested.csv'); pub=pd.read_csv('ALL_published_signature_genes.csv').drop_duplicates('gene')
rho=pd.read_csv('rho_prolif_SEQC.csv',index_col=0).iloc[:,0]
hubs=set(hub.gene); zad={r.gene:r.z_adj for _,r in hub.drop_duplicates('gene').iterrows()}
net=pd.read_csv(U+'string_interactions_short_1405_nodes.tsv',sep='\t'); net.columns=[c.lstrip('#') for c in net.columns]
Gn=nx.Graph()
for a_,b_,w_ in zip(net.node1,net.node2,net.combined_score): Gn.add_edge(a_,b_,weight=float(w_))
mc=pd.read_excel(U+'MCODE_cluster1_csv.xlsx',header=None); mc.columns=['gene','g2','score','status','cluster']
mcg=mc.loc[mc.cluster.astype(str).str.contains('Cluster 1'),'gene'].tolist()
fig=plt.figure(figsize=(7.1,5.8))
gs=fig.add_gridspec(2,2,hspace=.5,wspace=.3)
# A
axA=fig.add_subplot(gs[0,0])
sets=[('Published\n(18 studies)',pub),('Hub genes\n(cytoHubba/MCODE)',hub.drop_duplicates('gene')),('Mitotic class',mit)]
xs=np.arange(3); w=.33
u=[100*(d.z_uni.abs()>1.96).mean() for _,d in sets]; a_=[100*(d.z_adj.abs()>1.96).mean() for _,d in sets]
wo=[100*(d.z_adj>1.96).mean() for _,d in sets]
axA.bar(xs-w/2,u,w,color=B,label='univariable'); axA.bar(xs+w/2,a_,w,color=R,label='adjusted')
axA.bar(xs+w/2,wo,w,color=R,alpha=.5,hatch='///',edgecolor='white',lw=0,label='adjusted, direction as claimed')
for i,(uu,aa) in enumerate(zip(u,a_)):
    axA.text(i-w/2,uu+2.5,'%.0f%%'%uu,ha='center',fontsize=6.5); axA.text(i+w/2,aa+2.5,'%.0f%%'%aa,ha='center',fontsize=6.5)
axA.set_xticks(xs); axA.set_xticklabels(['%s\nn=%d'%(s[0],len(s[1])) for s in sets],fontsize=6.5)
axA.set_ylim(0,125); axA.set_yticks([0,25,50,75,100]); axA.set_ylabel('Genes significant for EFS (%)')
axA.legend(frameon=False,fontsize=5.8,loc='upper center',handlelength=1.2)
axA.set_title('A  Three independent gene sources',loc='left',fontweight='bold',fontsize=9)
# B
axB=fig.add_subplot(gs[0,1])
meth=list(hub.method.unique()); lab=[x.replace('cytoHubba ','').replace('MCODE cluster 1','MCODE') for x in meth]
uu=[(hub[hub.method==x].z_uni.abs()>1.96).sum() for x in meth]; aa=[(hub[hub.method==x].z_adj.abs()>1.96).sum() for x in meth]
xs=np.arange(len(meth))
axB.bar(xs-.2,uu,.4,color=B,label='univariable'); axB.bar(xs+.2,aa,.4,color=R,label='adjusted')
axB.set_xticks(xs); axB.set_xticklabels(lab,fontsize=6.5,rotation=30,ha='right')
axB.set_ylabel('Genes with |z| > 1.96'); axB.set_ylim(0,14); axB.legend(frameon=False,fontsize=6.5)
axB.set_title('B  Every ranking method behaves alike',loc='left',fontweight='bold',fontsize=9)
# C
axC=fig.add_subplot(gs[1,0]); mm=mit.sort_values('z_uni').reset_index(drop=True)
for i,r in mm.iterrows(): axC.plot([r.z_uni,r.z_adj],[i,i],color='#dddddd',lw=.7,zorder=1)
axC.scatter(mm.z_uni,range(len(mm)),s=8,color=B,label='univariable',zorder=3)
axC.scatter(mm.z_adj,range(len(mm)),s=8,color=R,label='adjusted',zorder=3)
for v in (1.96,-1.96): axC.axvline(v,color='grey',ls=':',lw=.8)
axC.axvline(0,color='k',lw=.8); axC.set_yticks([]); axC.set_ylim(-1.5,len(mm)+.5); axC.set_xlim(-6,12)
axC.set_xlabel('Cox z statistic'); axC.set_ylabel('40 mitotic / cell-cycle genes')
axC.legend(frameon=False,fontsize=6.5,loc='upper left')
axC.set_title('C  The mitotic programme as a class',loc='left',fontweight='bold',fontsize=9)
# D
axD=fig.add_subplot(gs[1,1])
common=[g for g in m.index if g in rho.index]
axD.scatter(rho[common],m.loc[common,'SEQC_z_adj'].abs(),s=2,c=GY,lw=0,rasterized=True)
cons=pd.read_csv('FULLFIT_consensus_and_index.csv'); cons=cons[cons.consensus_fullfit]
axD.scatter(rho.reindex(cons.gene).values,cons.SEQC_z.abs(),s=12,color=G,lw=0,label='consensus (n=44)')
axD.scatter(mit.rho_prolif,mit.z_adj.abs(),s=12,color=R,lw=0,label='mitotic (n=40)')
axD.scatter(hub.drop_duplicates('gene').rho_prolif,hub.drop_duplicates('gene').z_adj.abs(),s=20,facecolor='none',edgecolor=B,lw=1,label='hub genes (n=26)')
axD.axhline(1.96,color='grey',ls=':',lw=.8); axD.set_ylim(0,5.5); axD.set_xlim(-.85,1.05)
axD.set_xlabel('Spearman rho with proliferation'); axD.set_ylabel('| adjusted z |')
axD.legend(frameon=False,fontsize=6,loc='upper right')
axD.set_title('D  Collinearity predicts failure',loc='left',fontweight='bold',fontsize=9)
fig.savefig('Fig6.png',bbox_inches='tight'); plt.close(fig)

# ================= FIGURE 7 : network panels =================
fig=plt.figure(figsize=(7.1,4.3))
gs=fig.add_gridspec(1,2,width_ratios=[1.45,1],wspace=.24)
# E network
axE=fig.add_subplot(gs[0,0])
deg=dict(Gn.degree()); keep=[n for n,d in sorted(deg.items(),key=lambda x:-x[1])[:400]]
S=Gn.subgraph(keep); pos=nx.spring_layout(S,seed=11,k=.11,iterations=120,weight=None)
nx.draw_networkx_edges(S,pos,alpha=.06,width=.25,edge_color='#888888',ax=axE)
nodes=list(S.nodes()); vals=np.array([rho.get(n,np.nan) for n in nodes],dtype=float)
sc=axE.scatter([pos[n][0] for n in nodes],[pos[n][1] for n in nodes],c=np.nan_to_num(vals),cmap='RdBu_r',vmin=-1,vmax=1,
  s=[55 if n in hubs else 9 for n in nodes],linewidths=[.6 if n in hubs else 0 for n in nodes],edgecolors='k',zorder=3)
hi_=[n for n in nodes if rho.get(n,0)>0.8]; lo_=[n for n in nodes if rho.get(n,0)<-0.2]
xs_=[pos[n][0] for n in nodes]; ys_=[pos[n][1] for n in nodes]
span=max(max(xs_)-min(xs_),max(ys_)-min(ys_))
for grp,txt in [(hi_,'mitotic hub lobe\n(rho > 0.9)'),(lo_,'immune / signalling\nlobe (rho < 0)')]:
    if not grp: continue
    gx=np.mean([pos[n][0] for n in grp]); gy=np.mean([pos[n][1] for n in grp])
    ox=np.sign(gx-np.mean(xs_)) or 1; oy=np.sign(gy-np.mean(ys_)) or 1
    axE.annotate(txt,xy=(gx,gy),xytext=(gx+ox*span*.30,gy+oy*span*.30),fontsize=6,
        ha='center',va='center',arrowprops=dict(arrowstyle='->',lw=.6,color='#444444'))
axE.set_axis_off(); axE.margins(.34)
cb=fig.colorbar(sc,ax=axE,fraction=.04,pad=.01); cb.set_label('rho with proliferation',fontsize=6); cb.ax.tick_params(labelsize=5.5)
axE.set_title('A  STRING network',loc='left',fontweight='bold',fontsize=9,pad=8)
# F MCODE
axF=fig.add_subplot(gs[0,1])
H=Gn.subgraph([g for g in mcg if g in Gn]); p2=nx.spring_layout(H,seed=3,k=1.0)
nx.draw_networkx_edges(H,p2,alpha=.35,width=.8,ax=axF)
s2=axF.scatter([p2[n][0] for n in H.nodes()],[p2[n][1] for n in H.nodes()],c=[zad.get(n,0) for n in H.nodes()],
  cmap='RdBu_r',vmin=-4,vmax=4,s=360,edgecolors='k',linewidths=.7,zorder=3)
for n in H.nodes():
    axF.annotate(f'{n}\n{zad.get(n,float("nan")):.1f}',p2[n],fontsize=5,ha='center',va='center',zorder=4,
                 color='white' if abs(zad.get(n,0))>2.4 else 'black')
axF.set_axis_off(); axF.margins(.18)
cb2=fig.colorbar(s2,ax=axF,fraction=.04,pad=.01); cb2.set_label('adjusted Cox z',fontsize=6); cb2.ax.tick_params(labelsize=5.5)
axF.set_title('B  MCODE top cluster',loc='left',fontweight='bold',fontsize=9,pad=8)
fig.savefig('Fig7net.png',bbox_inches='tight'); plt.close(fig)
from PIL import Image
print('Fig3',Image.open('Fig3.png').size,'Fig6',Image.open('Fig6.png').size,'Fig7net',Image.open('Fig7net.png').size)
