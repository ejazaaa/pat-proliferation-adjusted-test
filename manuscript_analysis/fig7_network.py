import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, networkx as nx
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.8,'savefig.dpi':600})
U='/mnt/user-data/uploads/'
net=pd.read_csv(U+'string_interactions_short_1405_nodes.tsv',sep='\t'); net.columns=[c.lstrip('#') for c in net.columns]
rho=pd.read_csv('rho_prolif_SEQC.csv',index_col=0).iloc[:,0]
hub=pd.read_csv('cytoHubba_MCODE_tested.csv'); hub=hub[hub.z_uni.notna()]
hubs=list(dict.fromkeys(hub.gene)); zad={r.gene:r.z_adj for _,r in hub.drop_duplicates('gene').iterrows()}
mc=pd.read_excel(U+'MCODE_cluster1_csv.xlsx',header=None); mc.columns=['gene','g2','score','status','cluster']
mcg=mc.loc[mc.cluster.astype(str).str.contains('Cluster 1'),'gene'].tolist()
hi=net[net.combined_score>=0.7]
G=nx.Graph()
for a,b,w in zip(hi.node1,hi.node2,hi.combined_score): G.add_edge(a,b,weight=float(w))
G=G.subgraph(max(nx.connected_components(G),key=len)).copy()
fig=plt.figure(figsize=(7.1,6.8))
gs=fig.add_gridspec(2,2,height_ratios=[1.35,1],hspace=.18,wspace=.2)
# ---- A: whole high-confidence network
axA=fig.add_subplot(gs[0,:])
pos=nx.spring_layout(G,seed=5,k=.45,iterations=400,weight=None)
nx.draw_networkx_edges(G,pos,alpha=.08,width=.28,edge_color='#9a9a9a',ax=axA)
nodes=list(G.nodes()); hubset=set(hubs)
meas=[n for n in nodes if n in rho.index and not np.isnan(rho.get(n,np.nan))]
unmeas=[n for n in nodes if n not in meas]
axA.scatter([pos[n][0] for n in unmeas],[pos[n][1] for n in unmeas],c='#e8e8e8',s=9,lw=0,zorder=2)
mm=[n for n in meas if n not in hubset]
axA.scatter([pos[n][0] for n in mm],[pos[n][1] for n in mm],c=[rho[n] for n in mm],cmap='RdBu_r',vmin=-1,vmax=1,s=13,lw=0,zorder=3)
hh=[n for n in nodes if n in hubset]
axA.scatter([pos[n][0] for n in hh],[pos[n][1] for n in hh],
   c=[rho.get(n,0) if n in meas else .0 for n in hh],cmap='RdBu_r',vmin=-1,vmax=1,s=70,linewidths=1.0,edgecolors='k',zorder=5)
xs=[pos[n][0] for n in nodes]; ys=[pos[n][1] for n in nodes]; span=max(max(xs)-min(xs),max(ys)-min(ys))
for sel,txt in [([n for n in nodes if rho.get(n,0)>0.8],'proliferation module (rho > 0.8)'),
                ([n for n in nodes if rho.get(n,0)<-0.2],'immune / signalling module\n(rho < 0)')]:
    if not sel: continue
    gx=np.mean([pos[n][0] for n in sel]); gy=np.mean([pos[n][1] for n in sel])
    ox=np.sign(gx-np.mean(xs)) or 1; oy=np.sign(gy-np.mean(ys)) or 1
    axA.annotate(txt,xy=(gx,gy),xytext=(gx+ox*span*.34,gy+oy*span*.26),fontsize=6.5,ha='center',va='center',
                 bbox=dict(boxstyle='round,pad=.25',fc='white',ec='#bbbbbb',lw=.4),
                 arrowprops=dict(arrowstyle='->',lw=.7,color='#444444'))
axA.set_axis_off(); axA.margins(.16)
sm=plt.cm.ScalarMappable(cmap='RdBu_r',norm=plt.Normalize(-1,1))
cb=fig.colorbar(sm,ax=axA,fraction=.022,pad=.01); cb.set_label('Spearman rho with proliferation',fontsize=6.5); cb.ax.tick_params(labelsize=6)
axA.legend(handles=[Line2D([],[],marker='o',ls='',mfc='#dddddd',mec='k',mew=1.0,ms=7,label='hub gene (cytoHubba / MCODE)'),
                    Line2D([],[],marker='o',ls='',mfc='#bbbbbb',mec='none',ms=3.8,label='other gene, expression measured'),
                    Line2D([],[],marker='o',ls='',mfc='#e8e8e8',mec='none',ms=3.2,label='not measured in SEQC')],
           frameon=False,fontsize=6,loc='upper left')
axA.set_title('A  STRING network, high-confidence edges (score \u2265 0.7): %d genes, %d interactions'%(G.number_of_nodes(),G.number_of_edges()),
              loc='left',fontweight='bold',fontsize=8.5)
# ---- B: hub subnetwork, labelled
axB=fig.add_subplot(gs[1,0])
Gall=nx.Graph()
for a,b,w in zip(net.node1,net.node2,net.combined_score): Gall.add_edge(a,b,weight=float(w))
H=Gall.subgraph([g for g in hubs if g in Gall]).copy()
pB=nx.spring_layout(H,seed=14,k=2.6,iterations=800)
nx.draw_networkx_edges(H,pB,alpha=.22,width=.5,ax=axB)
axB.scatter([pB[n][0] for n in H.nodes()],[pB[n][1] for n in H.nodes()],
   c=[np.nan_to_num(rho.get(n,0)) for n in H.nodes()],cmap='RdBu_r',vmin=-1,vmax=1,s=215,edgecolors='k',linewidths=.6,zorder=3)
for n in H.nodes(): axB.annotate(n,pB[n],fontsize=5.1,ha='center',va='center',zorder=4,
    color='white' if abs(rho.get(n,0))>.75 else 'black')
axB.set_axis_off(); axB.margins(.20)
axB.set_title('B  The 26 hub genes and their interactions',loc='left',fontweight='bold',fontsize=8.5)
# ---- C: MCODE cluster, coloured by adjusted z
axC=fig.add_subplot(gs[1,1])
M=Gall.subgraph([g for g in mcg if g in Gall]); pC=nx.spring_layout(M,seed=3,k=1.0)
nx.draw_networkx_edges(M,pC,alpha=.35,width=.8,ax=axC)
s2=axC.scatter([pC[n][0] for n in M.nodes()],[pC[n][1] for n in M.nodes()],c=[zad.get(n,0) for n in M.nodes()],
   cmap='RdBu_r',vmin=-4,vmax=4,s=340,edgecolors='k',linewidths=.7,zorder=3)
for n in M.nodes(): axC.annotate(f'{n}\n{zad.get(n,float("nan")):.1f}',pC[n],fontsize=4.9,ha='center',va='center',
   zorder=4,color='white' if abs(zad.get(n,0))>2.4 else 'black')
axC.set_axis_off(); axC.margins(.16)
cb2=fig.colorbar(s2,ax=axC,fraction=.04,pad=.02); cb2.set_label('adjusted Cox z',fontsize=6.5); cb2.ax.tick_params(labelsize=6)
axC.set_title('C  MCODE top cluster (score 7.0)',loc='left',fontweight='bold',fontsize=8.5)
fig.savefig('Fig7.png',bbox_inches='tight'); plt.close(fig)
n_hub_prolif=sum(1 for g in hubs if rho.get(g,0)>0.8)
print('hubs with rho>0.8:',n_hub_prolif,'of',len(hubs))
from PIL import Image; print(Image.open('Fig7.png').size)
