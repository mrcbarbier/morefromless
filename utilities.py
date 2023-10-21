import numpy as np, pandas as pd,scipy.integrate as scint, scipy.ndimage as ndimage, itertools
import time,sys, matplotlib.pyplot as plt, scipy.fftpack as fft,scipy.signal as ssignal, os, ast
from copy import deepcopy
import cProfile
profiler=cProfile.Profile()

def ifelse(x,a,b):
    if x:
        return a
    return b

def offdiag(x):
    x=np.array(x)
    return x[np.eye(x.shape[0]) ==0]

def code_debugger(skip=0):
    import code
    import inspect
    stack=inspect.stack()
    dic = {}
    dic.update(stack[1+skip][0].f_globals)
    dic.update(stack[1+skip][0].f_locals)
    def trace():
        tmp=['  '+' '.join([str(x[y]) for y in [1, 2]]) + '\n      ' + str(not x[4] or x[4][0]).strip() for x in
                         stack if x[4]  ]
        tmp=tmp[ np.argmax(['code_debugger' in t for t in tmp ] ) :]
        print(('\n'.join(tmp[::-1])))
    dic['trace']=trace
    code.interact(local=dic)

def prolog(fname,col=2):
    stats=[[i.code ,i.totaltime,i.inlinetime,i.callcount,i.reccallcount] for i in profiler.getstats()]
    stats=sorted(stats,key=lambda e:e[col],reverse=1)
    with open(fname,'w') as prolog:

        st='code totaltime inlinetime callcount reccallcount'
        prolog.write(st+'\n')
        for i in stats:
            if not i:
                continue
            st=' '.join([str(z) for z in  i])
            prolog.write(st+'\n')


def reorder_FT(x,k=None):
    """Fourier transforms are typically in order [0..k(N-1),-k(N)...-k(1)].
    This reorders them to [-k(N)..k(N-1)] or back."""
    if k is None:
        lx, ly = x.shape[-2:]
        kx,ky=fft.fftfreq(lx), fft.fftfreq(ly)
    else:
        kx,ky=k
    x2=x[np.ix_(*[list(range(z)) for z in x.shape[:-2]]+[ np.argsort(kx),np.argsort(ky) ])]
    return x2

def checkpos_FT(x,idx=0):
    """Checks positivity of first element of x, by looking at conjugate symmetry of Fourier Transform."""
    t=x
    while len(t.shape)>2:
        t=t[idx]
    lx,ly=t.shape
    check=np.max(np.abs(t[lx / 2:,1: ] - np.conj(t[1:lx / 2 + 1,1:][::-1, ::-1])))
     #np.max(np.abs(t[lx / 2:, ly / 2:] - np.conj(t[1:lx / 2 + 1, 1:ly / 2 + 1][::-1, ::-1])))
    # code_debugger()
    return check

def setpos_FT(x):
    """Enforce positivity of x by symmetry of its Fourier transform."""
    lx,ly=x.shape[-2:]
    x2=x.copy()
    x2[:,lx / 2+1:,1:]=np.conj(x2[:,1:lx / 2 ,1:][:,::-1, ::-1])
    x2[:,lx / 2,ly / 2+1:]=np.conj(x2[:,lx / 2,1:ly / 2][:,::-1] )
    x2[:,lx/2,ly/2]=x2[:,lx/2,ly/2].real
    if checkpos_FT(x2)>0.0001:
        print('setpos_FT failed')
        code_debugger()
    return x2


def offdiag(mat):
    return mat[np.eye(mat.shape[0])!=1]

def matrix_symmetrize(mat,gamma):
    mn,sd=np.mean(offdiag(mat)),np.std(offdiag(mat))
    if sd>0:
        g=(mat-mn)/sd
    else:
        g=mat
    if gamma is None:
        gamma=0
    assert -1 <= gamma <= 1
    if gamma==0:
        asymm=0
    else:
        asymm=(1- np.sqrt(1-gamma**2) )/gamma
    g= (g+asymm * g.T) / np.sqrt(1+ asymm**2 )
    g=g*sd+mn
    g[np.eye(mat.shape[0])>0]=np.diag(mat)
    return g

def linfit(xs,ys,log=''):
    """Convenience function for linear regression and subsequent plot."""
    from scipy.stats import linregress
    x,y=xs,ys
    if 'x' in log:
        x=np.log10(x)
    if 'y' in log:
        y = np.log10(y)
    slope, intercept, r, p, stderr = linregress(x,y)
    pxs=np.sort(xs)
    pys=np.sort(x)
    pys=slope*pys+intercept
    if 'y' in log:
        pys=10**(pys)
    desc=r'${:.2f} + {:.2f} x$ ($R^2$={:.2f}, p={:.2f})'.format(intercept,slope,r**2,p)
    if 'x' in log and 'y' in log:
        desc=r"${{{: .2f}}} \; x^{{{: .2f}}}$ ($R^2$={:.2f}, p={:.2g})".format(10**intercept,slope,r**2,p)
    return pxs,pys,desc,slope,intercept,r,p,stderr

def powlaw(expo,xmin,xmax,shape=None):
    """Generate numbers from a distribution proportional to x**expo"""
    xx=expo+1
    if not xx:
        return np.exp(np.log(xmin) + np.log(xmax/xmin)*np.random.random(shape))
    return (xmin**xx + (xmax**xx-xmin**xx)*np.random.random(shape))**(1./xx)

def generate_noise(shape,method='fft', **dprm):
    """Generate noisy landscape (for now only method=fft and filter works convincingly"""
    samples = dprm.get('samples', 500)
    dimres = []
    if method=='fft':
        if len(shape)==2:
            shape=shape+(1,)
        res=np.zeros(shape)
        for z in range(shape[-1]):
            ft = np.random.random(shape[:-1])
            ft-=np.mean(ft)
            ft[0,0]=10
            color=dprm.get('color',1)
            lx, ly = ft.shape
            kx, ky = fft.fftfreq(lx), fft.fftfreq(ly)
            k=np.add.outer(kx ** 2, ky ** 2)**0.5
            ft=ft / (0.00000001+k** (color )) * np.exp(-k/np.max(k)*1./dprm.get('cutoff',1000))
            locres=fft.fft2(ft )
            locres=reorder_FT(locres.real+locres.imag)
            # plt.imshow(locres)
            # plt.show()
            res[:,:,z]=locres    
            #raise Exception('Not implemented yet')
    elif method=='filter':
        res = np.random.random(shape)
        res = ndimage.gaussian_filter(res, sigma=5)
    elif method=='direct':
        print('Experimental, failed')
        for dim in range(len(shape)):
            freqs = np.logspace(0, np.log(shape[dim] / 100.), samples)
            amps = freqs ** dprm.get('color', 1)
            phase = np.random.random(samples)
            xs = np.zeros(shape)
            dx = np.linspace(0, 1, shape[dim]).reshape(
                [1 for z in range(0, dim)] + [shape[dim]] + [1 for z in range(dim + 1, len(shape))])
            ps = np.multiply.outer(*[np.linspace(0, 1, sh) for sh in shape])
            xs = xs + dx
            dimres.append((np.sum([a * np.exp(1j * 2 * np.pi * (xs * f + p * np.cos(2 * np.pi * ps))) for f, a, p in
                                   zip(freqs, amps, phase)], axis=0)))
        res = np.real(np.multiply(*dimres))

    else:
        freqs = np.logspace(0, 2, samples)
        amps = freqs ** dprm.get('spectralexp', 0)
        phase = np.random.random((samples, len(shape)))
        xs = [np.zeros(shape) + np.linspace(0, 1, sh).reshape(
            [1 for z in range(0, dim)] + [sh] + [1 for z in range(dim + 1, len(shape))]) for dim, sh in
              enumerate(shape)]
        res = np.real(np.sum([a * np.exp(1j * 2 * np.pi * np.add(*[x * f + pp for x, pp in zip(xs, p)])) for f, a, p in
                              zip(freqs, amps, phase)], axis=0))
    return res


def generate_random(dprm):
    """ Generate a set of random parameters or values with prescribed properties. """
    shape=dprm['shape']
    if not hasattr(shape,'__iter__'):
        shape=[shape]

    dist=dprm['distribution']
    if 'relstd' in dprm:
        dprm['std']=dprm['relstd']*dprm['mean']
    if 'mu' in dprm:
        dprm['mean']=dprm['mu']*1./shape[0]
    if 'sigma' in dprm:
        dprm['std']=dprm['sigma']/np.sqrt(shape[0])
        
    if dist=='uniform':
        res=np.random.uniform(dprm['range'][0],dprm['range'][1],shape )
    elif dist=='normal':
        if dprm['std']>0:
            res=np.random.normal(dprm['mean'],dprm['std'],shape )
        else:
            res=np.ones(shape)*dprm['mean']
    elif dist=='power':
        xmin,xmax=dprm['range']
        res= powlaw(dprm['exponent'],xmin,xmax,shape)
    elif dist=='noise':
        # Generate noisy landscape
        res=generate_noise(shape=shape,**{i:j for i,j in list(dprm.items()) if i!='shape'} )
        rge=dprm['range']
        res=rge[0]+(res-np.min(res))*(rge[1]-rge[0])/(np.max(res)-np.min(res)) #Rescale generated noise

    if 'symmetry' in dprm:
        gamma=dprm['symmetry']
        res=matrix_symmetrize(res,gamma)
        print(np.corrcoef(res.ravel(),res.T.ravel())[0,1])
    if 'diagonal' in dprm:
        np.fill_diagonal(res,dprm['diagonal'])
    if 'sign' in dprm:
        sign=dprm['sign']
        if sign>0:
            res=np.abs(res)
        elif sign<0:
            res=-np.abs(res)
            
    if dprm.get('sorted',False):
        res1=res.copy()
        while len(res1.shape)>1:
            res1=res1[:,0]
        order=np.argsort(res1)
        res=res[order]
    return res
    
    
def dumps(fil,obj):
    """Write object to file"""
    fil.write(str(obj))

def loads(fil):
    """Load object from file"""
    txt=''
    for l in fil:
        txt+=l.strip()
    return eval(txt,{},{'array':np.array,'nan':np.nan})


def rebuild_filelist(path,verbose=True):
    """Run through directory tree to look for model results to add to files.csv"""
    if verbose:
        print('Rebuilding files.csv for {}'.format(path))
    final=None
    idx=0
    for root,dirs,files in os.walk(str(path)):
        if str(root)!=str(path):
            if 'files.csv' in files:
                pass
            elif 'model.dat' in files and 'results.csv' in files:
                print('Missing files.csv, regenerating')
                dic={'path':root }
                #dic.update( eval(open('model.dat','r' ) ) )
                df=pd.DataFrame([dic])
                df.to_csv(Path(root)+'files.csv')
            else:
                continue
            if verbose:
                print('{} Found files.csv in {}'.format(idx, root))
                idx+=1
            if final is None:
                final=pd.read_csv(Path(root)+'files.csv',index_col=0)
            else:
                final=final.append(pd.read_csv(Path(root)+'files.csv',index_col=0),ignore_index=1)
            final.loc[final.index[-1],'path']=str(Path(root).osnorm())

    if not final is None:
        final.to_csv(path+'files.csv')
    else:
        print('Error: No file found! {}'.format(path))


class Path(str):
    '''Strings that represent filesystem paths.
    Overloads __add__:
     - when paths are added, gives a path
     - when a string is added, gives a string'''
    def __add__(self,x):
        import os
        if isinstance(x,Path):
            return Path(os.path.normpath(os.path.join(str(self),x)))
        return os.path.normpath(os.path.join(str(self),x))

    def norm(self):
        import os
        return Path(os.path.normpath(str(self)))

    def osnorm(self):
        """Deal with different separators between OSes."""
        import os
        if os.sep=='/' and "\\" in str(self):
            return Path(os.path.normpath(str(self).replace('\\','/' )))
        elif os.sep=='\\' and "/" in str(self):
            return Path(os.path.normpath(str(self).replace('/','\\' )))
        else:
            return self.norm()

    def prev(self):
        import os
        lst=self.split()
        path=os.path.join(lst[:-1])
        return path.osnorm()

    def split(self,*args):
        """"""
        import os
        lst=[]
        cur=os.path.split(self.norm())
        while cur[-1]!='':
            lst.insert(0,cur[-1])
            cur=os.path.split(cur[0])
        return lst

    def mkdir(self,rmdir=False):
        """Make directories in path that don't exist. If rmdir, first clean up."""
        import os
        if rmdir:
            os.rmdir(str(self))
        cur=Path('./')
        for intdir in self.split():
            cur+=Path(intdir)
            if not os.path.isdir(cur):
                os.mkdir(cur)
        return self

    def copy(self):
        return Path(self)

    def strip(self,*args):
        '''Return string without final / or \\ to suffix/modify it.'''
        return str(self).strip('\/')



def auto_subplot(panel,nbpanels,rows=None,projection=None,return_all=0):
    """Helps deal with pyplot's subplots, automatically incrementing panel index."""
    i=int(panel)
    if rows is None:
        panels_per_row=np.ceil(np.sqrt(nbpanels) ).astype('int')
    else:
        panels_per_row=np.ceil(nbpanels/rows).astype('int')
    nbrows=int(np.ceil(nbpanels*1./panels_per_row))
    ax=plt.subplot(nbrows,panels_per_row,i +1 ,projection=projection)
    panel+=1
    if return_all:
        return ax,nbrows,panels_per_row
    return panel,ax


def draw_network(mat,ypos=None,xpos=None,directed=1,newfig=1,hold=0):
    edges=[(i,j) for i,j in zip(*np.where(mat!=0)) if mat[i,j]>mat[j,i] ]
    nodes=np.sort(np.unique(np.concatenate(edges)))
    N=len(nodes)
    if newfig:
        fig=plt.figure()
    else:
        fig=plt.gcf()
    if xpos is None:
        xpos=dict(list(zip(nodes,np.random.random(N) )))
    if ypos is None:
        ypos=dict(list(zip(nodes,np.random.random(N))))
    if not hasattr(xpos,'keys'):
        xpos=dict(list(zip(nodes,xpos)))
    if not hasattr(ypos,'keys'):
        ypos=dict(list(zip(nodes,ypos)))
    xs = [xpos[n] for n in nodes]
    ys = [ypos[n] for n in nodes]
    xs,ys=np.array(xs),np.array(ys)
    plt.scatter(xs,ys)
    ax = plt.gca()
    if directed:
        # X,Y,U,V=zip(*[(xpos[i],ypos[i],xpos[j]-xpos[i],ypos[j]-ypos[i]) for i,j in edges])
        # plt.quiver(X,Y,U,V)
        for i, j in edges:
            try:
                line=plt.FancyArrow(xpos[i],ypos[i],xpos[j]-xpos[i],ypos[j]-ypos[i] )
            except:
                line=plt.Arrow(xpos[i],ypos[i],xpos[j]-xpos[i],ypos[j]-ypos[i] ,width=0.02)
            ax.add_artist(line)
    else:
        for i,j in edges:
            line=plt.Line2D([xpos[i],xpos[j]],[ypos[i],ypos[j]] )
            ax.add_artist(line)
    if not hold:
        plt.show()
    else:
        return fig

def naive_trophic_score(A,r,z=.01):
    M=np.eye(A.shape[0])-z*A
    try:
        result= np.log(np.dot(np.linalg.inv(M),r ))/np.log(z)
    except Exception as e:
        print(e)
        result=np.zeros(r.shape)
    return result


