import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import Patch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.8,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':600})
B,R,G,GY='#2c5f8a','#c0504d','#4f8a3d','#cccccc'
f=pd.read_csv('FULLFIT_consensus_and_index.csv').set_index('gene')
m=pd.read_csv('META3_SEQC_NRC_KOCAK.csv').set_index('gene')
# ---------- Fig 3: index genes (TRIP13, CCNB2) ----------
fig,ax=plt.subplots(1,2,figsize=(7.1,3.0))
labels=['TRIP13 alone','+ age','+ high-risk','+ proliferation']
hr=np.array([2.38,2.04,1.72,1.40]); lo=np.array([1.99,1.71,1.41,1.03]); hi=np.array([2.83,2.43,2.10,1.90])
y=np.arange(4)[::-1]
ax[0].errorbar(hr,y,xerr=[hr-lo,hi-hr],fmt='o',color=B,ms=5,capsize=3,lw=1.2)
ax[0].axvline(1,color='k',ls='--',lw=.8); ax[0].set_yticks(y); ax[0].set_yticklabels(labels,fontsize=7.5)
ax[0].set_xlabel('Hazard ratio per SD (SEQC, EFS)'); ax[0].set_xlim(.9,3.0)
ax[0].set_title('A  TRIP13, nested models',loc='left',fontweight='bold',fontsize=9)
genes=['TRIP13','CCNB2']; x=np.arange(2); w=.16
series=[('SEQC univariable',[m.loc[g,'SEQC_z_uni'] for g in genes],B,.4),
        ('Kocak univariable',[m.loc[g,'KOCAK_z_uni'] for g in genes],G,.4),
        ('SEQC adjusted',[f.loc[g,'SEQC_z'] for g in genes],B,1.0),
        ('NRC adjusted',[f.loc[g,'NRC_z'] for g in genes],R,1.0),
        ('Kocak adjusted',[f.loc[g,'Kocak_z'] for g in genes],G,1.0)]
for i,(lab,v,col,a) in enumerate(series): ax[1].bar(x+(i-2)*w,v,w*.9,color=col,alpha=a,label=lab,edgecolor=col,lw=.4)
ax[1].axhline(1.96,color='grey',ls=':',lw=.8); ax[1].axhline(-1.96,color='grey',ls=':',lw=.8); ax[1].axhline(0,color='k',lw=.8)
ax[1].text(1.42,2.2,'p = 0.05',fontsize=6,color='grey')
ax[1].set_xticks(x); ax[1].set_xticklabels(genes); ax[1].set_ylabel('Cox z statistic'); ax[1].set_ylim(-3,12); ax[1].set_xlim(-.6,1.6)
ax[1].legend(frameon=False,fontsize=6,ncol=2,loc='upper center')
ax[1].set_title('B  Index genes before and after adjustment',loc='left',fontweight='bold',fontsize=9)
fig.tight_layout(); fig.savefig('Fig3.png',bbox_inches='tight'); plt.close(fig)

# ---------- Fig 6: three independent nomination routes ----------
hub=pd.read_csv('cytoHubba_MCODE_tested.csv'); hub=hub[hub.z_uni.notna()]
mit=pd.read_csv('mitotic_class_tested.csv')
pub=pd.read_csv('ALL_published_signature_genes.csv').drop_duplicates('gene')
fig=plt.figure(figsize=(7.1,6.4))
gs=fig.add_gridspec(2,2,height_ratios=[1,1.05],hspace=.42,wspace=.3)
# A: counts by source
axA=fig.add_subplot(gs[0,0])
sets=[('Published\n(18 studies)',pub),('Hub genes\n(cytoHubba/MCODE)',hub.drop_duplicates('gene')),('Mitotic class',mit)]
xs=np.arange(3); w=.33
u=[100*(d.z_uni.abs()>1.96).mean() for _,d in sets]
a=[100*(d.z_adj.abs()>1.96).mean() for _,d in sets]
wo=[100*(d.z_adj>1.96).mean() for _,d in sets]
axA.bar(xs-w/2,u,w,color=B,label='univariable')
axA.bar(xs+w/2,a,w,color=R,label='adjusted')
axA.bar(xs+w/2,wo,w,color=R,alpha=.5,hatch='///',edgecolor='white',lw=0,label='adjusted, direction as claimed')
for i,(uu,aa,(lab,d)) in enumerate(zip(u,a,sets)):
    axA.text(i-w/2,uu+2.5,'%.0f%%'%uu,ha='center',fontsize=6.5)
    axA.text(i+w/2,aa+2.5,'%.0f%%'%aa,ha='center',fontsize=6.5)
axA.set_xticks(xs)
axA.set_xticklabels(['%s\nn=%d'%(s[0],len(s[1])) for s in sets],fontsize=6.5)
axA.set_ylim(0,125); axA.set_yticks([0,25,50,75,100])
axA.set_ylabel('Genes significant for EFS (%)')
axA.legend(frameon=False,fontsize=5.8,loc='upper center',bbox_to_anchor=(.5,1.03),ncol=1,handlelength=1.2)
axA.set_title('A  Three independent gene sources',loc='left',fontweight='bold',fontsize=9)
# B: per cytoHubba method
axB=fig.add_subplot(gs[0,1])
meth=[x for x in hub.method.unique()]
lab=[x.replace('cytoHubba ','').replace('MCODE cluster 1','MCODE') for x in meth]
uu=[(hub[hub.method==x].z_uni.abs()>1.96).sum() for x in meth]
aa=[(hub[hub.method==x].z_adj.abs()>1.96).sum() for x in meth]
xs=np.arange(len(meth))
axB.bar(xs-.2,uu,.4,color=B,label='univariable'); axB.bar(xs+.2,aa,.4,color=R,label='adjusted')
axB.set_xticks(xs); axB.set_xticklabels(lab,fontsize=6.5,rotation=30,ha='right')
axB.set_ylabel('Genes with |z| > 1.96'); axB.set_ylim(0,14); axB.legend(frameon=False,fontsize=6.5)
axB.set_title('B  Every ranking method behaves alike',loc='left',fontweight='bold',fontsize=9)
# C: mitotic class paired
axC=fig.add_subplot(gs[1,0])
mm=mit.sort_values('z_uni').reset_index(drop=True)
for i,r in mm.iterrows(): axC.plot([r.z_uni,r.z_adj],[i,i],color='#dddddd',lw=.7,zorder=1)
axC.scatter(mm.z_uni,range(len(mm)),s=8,color=B,label='univariable',zorder=3)
axC.scatter(mm.z_adj,range(len(mm)),s=8,color=R,label='adjusted',zorder=3)
for v in (1.96,-1.96): axC.axvline(v,color='grey',ls=':',lw=.8)
axC.axvline(0,color='k',lw=.8); axC.set_yticks([]); axC.set_ylim(-1.5,len(mm)+.5); axC.set_xlim(-6,12)
axC.set_xlabel('Cox z statistic'); axC.set_ylabel('40 mitotic / cell-cycle genes')
axC.legend(frameon=False,fontsize=6.5,loc='upper left')
axC.set_title('C  The mitotic programme as a class',loc='left',fontweight='bold',fontsize=9)
# D: collinearity
axD=fig.add_subplot(gs[1,1])
allz=m; rho=pd.read_csv('rho_prolif_SEQC.csv',index_col=0).iloc[:,0]
common=[g for g in allz.index if g in rho.index]
axD.scatter(rho[common],allz.loc[common,'SEQC_z_adj'].abs(),s=2,c=GY,lw=0,rasterized=True)
cons=pd.read_csv('FULLFIT_consensus_and_index.csv'); cons=cons[cons.consensus_fullfit]
axD.scatter(rho.reindex(cons.gene).values,cons.SEQC_z.abs(),s=12,color=G,lw=0,label='consensus (n=44)')
axD.scatter(mit.rho_prolif,mit.z_adj.abs(),s=12,color=R,lw=0,label='mitotic (n=40)')
axD.scatter(hub.drop_duplicates('gene').rho_prolif,hub.drop_duplicates('gene').z_adj.abs(),s=20,facecolor='none',edgecolor=B,lw=1,label='hub genes (n=26)')
axD.axhline(1.96,color='grey',ls=':',lw=.8); axD.set_ylim(0,5.5); axD.set_xlim(-.85,1.05)
axD.set_xlabel('Spearman rho with proliferation'); axD.set_ylabel('| adjusted z |')
axD.legend(frameon=False,fontsize=6,loc='upper right')
axD.set_title('D  Collinearity predicts failure',loc='left',fontweight='bold',fontsize=9)
fig.savefig('Fig6.png',bbox_inches='tight'); plt.close(fig)

# ---------- Fig 7: C-index ----------
allc=pd.read_csv('cindex_gain.csv'); base=allc.query("grp=='base'").c.iloc[0]
c=allc[(allc.grp!='base')&(allc.gene!='KIF2C')].sort_values('delta')
fig,ax=plt.subplots(figsize=(4.6,3.4))
ax.barh(np.arange(len(c)),c.delta,color=[R if g=='index' else G for g in c.grp],height=.65)
ax.set_yticks(np.arange(len(c))); ax.set_yticklabels(c.gene,fontsize=7)
ax.axvline(0,color='k',lw=.8); ax.set_xlim(-0.0005,0.018)
ax.set_xlabel('Change in C-index when added to the standard model')
ax.text(.99,.06,f'base model C = {base:.3f}\n(age, risk group, proliferation)',transform=ax.transAxes,ha='right',fontsize=6.5)
ax.legend(handles=[Patch(color=R,label='index mitotic genes'),Patch(color=G,label='top consensus genes')],
          frameon=False,fontsize=6.5,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2)
fig.tight_layout(); fig.savefig('Fig7.png',bbox_inches='tight'); plt.close(fig)
from PIL import Image
for n in ['Fig3','Fig6','Fig7']: print(n, Image.open(n+'.png').size)
