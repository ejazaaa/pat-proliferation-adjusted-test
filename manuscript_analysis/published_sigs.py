import numpy as np, pandas as pd, warnings, sys
sys.path.insert(0,'/home/claude/nb')
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
U='/mnt/user-data/uploads/'
SIGS={
'Cangelosi 2020 (6)':"AK4,EGLN1,FAM162A,MTFP1,PDK1,PGK1",
'Zhong 2021 (5)':"CDK4,PIK3R1,THRA,MAP2K2,ULBP2",
'Jin 2020 (9)':"AMH,IGHV3-20,IGKV1-16,IRF7,KLRK1,MARCO,SECTM1,SOCS1,UNC93B1",
'Vermeulen 2009 (59)':"AHCY,AKR1C1,ARHGEF7,BIRC5,CADM1,CAMTA1,CAMTA2,CD44,CDCA5,CDKN3,CHD5,CLSTN1,CPSG3,DDC,DPYSL3,ECEL1,ELAVL4,EPB41L3,EPHA5,EPN2,FYN,GNB1,HIVEP2,INPP1,MAP2K4,MAP7,MAPT,MCM2,MRPL3,MTSS1,MYCN,NHLH2,NME1,NRCAM,NTRK1,ODC1,PAICS,PDE4DIP,PIK3R1,PLAGL1,PLAT,PMP22,PRAME,PRDM2,PRKACB,PRKCZ,PTN,PTPRF,PTPRH,PTPRN2,QPCT,SCG2,SLC25A5,SLC6A8,SNAPC1,TNFRSF25,TYMS,ULK2,WSB1",
'De Preter 2010 (42)':"AHCY,AKR1C1,ARHGEF7,BIRC5,CADM1,CAMTA2,CDCA5,CDKN3,CLSTN1,DDC,DPYSL3,ECEL1,EPB41L3,EPHA5,EPN2,FYN,GNB1,HIVEP2,INPP1,MAP7,MAPT,MCM2,MRPL3,MYCN,NCAN,NME1,NRCAM,NTRK1,ODC1,PAICS,PLAGL1,PMP22,PRKACB,PRKCZ,PTN,PTPRN2,SCG2,SLC25A5,SNAPC1,TYMS,ULK2,WSB1",
'Valentijn 2012 (157)':"ACSL4,AGPAT4,ANAPC1,ANO4,ARHGEF3,ARL4D,ASTN1,ATAD2,C10orf26,C19orf43,C1orf114,C1orf97,C20orf72,C4orf34,C5orf41,C7orf60,C9orf95,CAMK2N1,CCNI,CELF3,CLU,CNNM1,CNTN1,CRTAP,CSRNP3,CYP4V2,DCTPP1,DHCR7,DIMT1L,DIRAS3,DKC1,DNER,E2F5,E2F8,ECE2,EIF2C4,EIF2S2,EPB41L1,FAM134B,FAM167A,FAM174A,FAM190B,FAM36A,FAM82A2,FAM85A,FOXRED2,GABRA5,GAR1,GNAZ,GPR85,GPX7,GRSF1,GTPBP1,HJURP,HMP19,HPDL,IARS,IFT20,IMPDH2,IQCG,KIAA0226,KIAA0895,KIF23,KIF4A,KLHL14,LIG3,LOC442075,LRRC59,LYAR,LYSMD2,MAP2,MBNL2,MCCC2,MCM3,MCM5,METTL9,MFAP4,MFSD6,MKRN1,MRPL45,MSH6,MTAP,MYPOP,NAT15,NCAPH,NOC4L,NRBP2,NT5DC3,NUDT14,OIP5,OSCP1,PCCB,PDE5A,PFKM,PHF20,PHGDH,PKIA,PLEKHA3,PMM2,POLA2,POLD2,POLE2,PRDX6,PREP,PRMT1,PRMT7,PUS7,PYCR1,RAB6B,RAD51,RAD51AP1,RAD54L,RAGE,RAVER1,RAVER2,RHOBTB3,RHOC,RND2,RNF144A,RNF146,RRM2,RUNDC3A,SCN3A,SESTD1,SH3BGR,SH3BGRL2,SKA3,SLBP,SLC20A1,SRF,STMN2,SUSD5,SYNPO2,SYT13,SYT4,TBX2,TCTN3,TEX15,TIPIN,TMEM150C,TMEM170B,TP53INP1,TPPP3,TRAK2,TRIM69,UBE2H,UBL7,UBQLN2,WDR18,WDR76,WDR77,XPO5,ZBTB4,ZNF25,ZNF317,ZNF608,ZNRF3",
'Zhong 2018 (4)':"AHCY,ERCC6L,NCAN,STK33",
'Garcia 2012 (3)':"CHD5,NME1,PAFAH1B1",
'Frumm 2013 (59)':"ABCB1,ADD3,ARHGEF3,ASCL1,CACNA2D2,CALML4,CDKN1A,CHGA,CHST5,CPSF1,CRABP2,CRH,CTSB,CYP26A1,DDAH2,DDEF2,DLK1,DOCK4,DPYSL3,DUSP6,EGR1,FLJ10357,FLJ12584,FLNB,GATA3,HEBP2,HERC2,HOXD4,HS2ST1,IER3,IGFBP5,ISGF3G,JARID2,KCTD12,KCTD13,MAFF,MMP11,MTMR1,N4BP1,OIP106,PELO,PLAT,PLEKHA6,PLK2,PPIF,PRKCH,PTBP1,RAB20,RAP80,RBP1,RGS16,SLC29A1,ST6GAL1,STX6,SYNJ2,SYT11,TGM2,TPST1,VGF",
'Wang 2020 (5)':"HNRNPC,IGF2BP2,METT14,WTAP,YTHDF1"}
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
for sig,g in SIGS.items():
    for gene in [x.strip() for x in g.split(',')]:
        if gene not in S.index: rows.append({'signature':sig,'gene':gene,'measured':False}); continue
        pset=[x for x in PROLIF if x in S.index and x!=gene]
        x=d.copy(); x['x']=z(S.loc[gene,x.index]); x['prolif']=z(S.loc[pset].apply(z,axis=1).mean())
        x=x.dropna(subset=['x','prolif','efs_t','efs_e','age18','highrisk'])
        u=CoxPHFitter(penalizer=1e-4).fit(x[['x','efs_t','efs_e']],'efs_t','efs_e')
        a=CoxPHFitter(penalizer=1e-4).fit(x[['x','age18','highrisk','prolif','efs_t','efs_e']],'efs_t','efs_e')
        rows.append({'signature':sig,'gene':gene,'measured':True,'z_uni':u.summary.z['x'],'p_uni':u.summary.p['x'],
                     'z_adj':a.summary.z['x'],'p_adj':a.summary.p['x'],'rho_prolif':spearmanr(x.x,x.prolif).correlation})
r=pd.DataFrame(rows); r.to_csv('published_signatures_tested.csv',index=False)
tot=r[r.measured==True]
print('genes in signatures: %d total, %d measured in SEQC'%(len(r),len(tot)))
print('%-24s %5s %5s %6s %6s %6s'%('signature','meas','uni','adj','worse','med|rho|'))
for sig in SIGS:
    s=tot[tot.signature==sig]
    if len(s)==0: continue
    print('%-24s %5d %5d %6d %6d %6.2f'%(sig,len(s),(s.z_uni.abs()>1.96).sum(),(s.z_adj.abs()>1.96).sum(),(s.z_adj>1.96).sum(),s.rho_prolif.abs().median()))
u=tot.drop_duplicates('gene')
print('\nUNION %d unique genes: uni sig %d (%.0f%%), adj sig %d (%.0f%%), adj worse %d'%(
   len(u),(u.z_uni.abs()>1.96).sum(),100*(u.z_uni.abs()>1.96).mean(),(u.z_adj.abs()>1.96).sum(),100*(u.z_adj.abs()>1.96).mean(),(u.z_adj>1.96).sum()))
print('\nThe three most reproducible genes (Chicco et al.):')
print(tot[tot.gene.isin(['AHCY','DPYSL3','NME1'])].drop_duplicates('gene')[['gene','z_uni','p_uni','z_adj','p_adj','rho_prolif']].round(3).to_string(index=False))
