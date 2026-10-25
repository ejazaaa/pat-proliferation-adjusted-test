import pandas as pd, numpy as np, gzip
U='/mnt/user-data/uploads/'
# RefSeq -> symbol (UCSC refGene, hg19)
rg=pd.read_csv(U+'refGene_txt.gz',sep='\t',header=None,usecols=[1,12],names=['refseq','symbol'])
rg['refseq']=rg.refseq.str.split('.').str[0]
r2s=rg.drop_duplicates('refseq').set_index('refseq').symbol
print('refGene map',len(r2s))
# GPL5175 probeset -> symbol (first gene in gene_assignment)
g5=pd.read_csv(U+'GPL5175-3188.txt',sep='\t',comment='#',usecols=['ID','gene_assignment'],dtype={'ID':str},low_memory=False)
def firstsym(s):
    if not isinstance(s,str) or s.strip() in ('','---'): return np.nan
    parts=s.split('//')
    return parts[1].strip() if len(parts)>1 else np.nan
g5['symbol']=g5.gene_assignment.map(firstsym)
p2s=g5.dropna(subset=['symbol']).drop_duplicates('ID').set_index('ID').symbol
print('GPL5175 mapped',len(p2s),'of',len(g5))
# GPL16876 -> symbol
g16=pd.read_csv(U+'GPL16876-29646.txt',sep='\t',comment='#',usecols=['ID','GeneSymbol'],dtype={'ID':str},low_memory=False)
k2s=g16.dropna(subset=['GeneSymbol']).drop_duplicates('ID').set_index('ID').GeneSymbol
print('GPL16876 mapped',len(k2s))
# HGNC aliases -> approved symbol, to harmonise across platforms
h=pd.read_csv(U+'hgnc_complete_set.txt',sep='\t',low_memory=False)
alias={}
for _,row in h.iterrows():
    s=row['symbol']
    for col in ['alias_symbol','prev_symbol']:
        v=row.get(col)
        if isinstance(v,str):
            for a in v.split('|'):
                alias.setdefault(a.strip(),s)
approved=set(h.symbol)
def harmonise(x):
    if not isinstance(x,str): return np.nan
    return x if x in approved else alias.get(x,x)
for name,ser in [('refseq2symbol',r2s),('gpl5175_probe2symbol',p2s),('gpl16876_probe2symbol',k2s)]:
    out=ser.map(harmonise).rename('symbol'); out.index.name='id'
    out.to_csv(f'{name}.csv'); print(name,'written',out.nunique(),'unique symbols')
