import numpy as np
def score_test(X, time, event):
    o=np.argsort(-np.asarray(time,float))
    X=np.asarray(X,float)[:,o]; t=np.asarray(time,float)[o]; d=np.asarray(event,int)[o]
    c1=np.cumsum(X,1); c2=np.cumsum(X**2,1); n=np.arange(1,X.shape[1]+1)
    last=np.zeros(len(t),bool); last[:-1]=t[:-1]!=t[1:]; last[-1]=True
    idx=np.maximum.accumulate(np.where(last,np.arange(len(t)),0))
    mean=c1[:,idx]/n[idx]; var=c2[:,idx]/n[idx]-mean**2
    U=((X-mean)*d).sum(1); V=(var*d).sum(1)
    with np.errstate(invalid='ignore',divide='ignore'):
        z=U/np.sqrt(V)
    return z,U,V

def score_test_offset(X, time, event, w):
    o=np.argsort(-np.asarray(time,float))
    X=np.asarray(X,float)[:,o]; t=np.asarray(time,float)[o]; d=np.asarray(event,int)[o]; w=np.asarray(w,float)[o]
    sw=np.cumsum(w); c1=np.cumsum(X*w,1); c2=np.cumsum(X**2*w,1)
    last=np.zeros(len(t),bool); last[:-1]=t[:-1]!=t[1:]; last[-1]=True
    idx=np.maximum.accumulate(np.where(last,np.arange(len(t)),0))
    mean=c1[:,idx]/sw[idx]; var=c2[:,idx]/sw[idx]-mean**2
    U=((X-mean)*d).sum(1); V=(var*d).sum(1)
    with np.errstate(invalid='ignore',divide='ignore'):
        return U/np.sqrt(V), U, V
