from landscapemodel import *
import matplotlib
import matplotlib.figure as mpfig, numpy.linalg as la
try:
    from matplotlib import rcParams
    mpfig.rcParams=rcParams
except:
    pass
import numpy.ma as ma
import scipy.stats as stats
from scipy.stats import linregress, spearmanr

# ============= MEASURES =================

def basic_measures(model,dic):
    """Measure various quantities for a given model, store summary in dictionary dic."""
    Nf = model.results['n'][-1]
    growth=model.data['growth']

    B=np.sum(Nf,axis=0)  #Total Biomass per patch
    basal=np.where(np.max(growth,axis=(1,2))>0 )[0] #Basal species
    Bbasal=np.sum(Nf[basal ],axis=0)  #Total producer biomass per patch
    prod=np.sum(growth*Nf,axis=0)  #Total production per patch

    alive=(Nf > model.prm['death'])
    n=Nf/(B+10**-15) # Relative biomass
    D = np.sum(alive, axis=0)  # Diversity
    DS = -np.sum(n * np.log(np.clip(n, 10 ** -15, None)), axis=0)  # Shannon diversity
    dic.update({ 'biomass_tot':np.sum(B),'biomass_basal':np.sum(Bbasal),
                 'production_tot':np.sum(prod),
                 'alpha_div':np.mean(D), 'alpha_div_min':np.min(D), 'alpha_div_max':np.max(D),
                 'alpha_shannon':np.mean(DS), 'alpha_shannon_min':np.min(DS), 'alpha_shannon_max':np.max(DS),
                 'gamma_div':np.sum(np.sum(alive,axis=(1,2))>0 ) })

    return B, Bbasal, D, DS

### I cannot think of a convenient way to measure stability (too high-dimensional for usual linear stability analysis
# def stability_measures(model,dic):
#     # try:
#     import numdifftools as nd
#     from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov, norm, inv
#     nf=model.results['n'][-1]
#     def fun(*args):
#         return model.get_dlogx(None, *args, calc_fluxes=0).ravel()
#     Jfun = nd.Jacobian(fun)
#     J= Jfun(nf,ravel()).reshape(nf.shape)
#     CIJ = [solve_lyapunov(J[:,i,j], -np.diag(nf)) for i,j in np.indices(J.shape[1:3])]
#     return np.mean([np.trace(x) for x in CIJ])
#     # except:
#     #     print 'ERROR: '

def hyperplane_kpower(N,k,death=10**-5,**kwargs):
    from scipy.optimize import least_squares
    S = N.shape[0]
    def link(x):
        return x
    mat = np.zeros((S, S)) #Returned matrix
    for sidx in range(S):
        notsidx= [s for s in range(S) if s!=sidx]
        alive=(N[sidx]>death)
        xsT = N[:,alive]
        xnomono =   N[:,(N[sidx]!=0)&(np.max(N[notsidx],axis=0)>death )]
        if not xnomono.shape[0]:
            print('hyperplane skipping', s, 'only present in monoculture')
            mat[sidx, sidx] = -1
            continue
        def costfeta(y, weights=None):
            yy = -np.ones(S)
            yy[notsidx] = link(y)
            return np.dot(yy, xsT) + N[sidx,alive]**(k-1)
        res = least_squares(costfeta, -np.zeros(S - 1))
        row = list(link(res.x))
        row.insert(sidx, -1)
        mat[sidx] = row
        mat[sidx, np.mean(np.abs(xsT), axis=1) <=death] = np.nan
    return mat


def hyperplane_light(N,K,death,mode='V',regularize=False,link=None,weights=None,**kwargs):
    if link is None:
        link = np.tanh
    from numpy.linalg import lstsq, norm as lanorm, inv as lainv
    from scipy.optimize import least_squares
    from scipy.stats import moment
    S = N.shape[0]
    mat = np.zeros((S, S)) #Returned matrix
    if mode == 'V':
        for sidx in range(S):
            res=None
            Ns=N[sidx]
            xsT = K[:,Ns >death]
            #xsT[xsT<0.1*np.median(xsT,axis=1).reshape((-1,1)) ]=0
            Nslive=Ns[Ns>death]
            vec1=np.zeros(S)
            vec1[sidx]=1
            def costfeta(y):
                if regularize:
                    return np.concatenate(( np.dot(y, xsT) -  Nslive  , y-vec1))
                return  np.dot(y, xsT) -  Nslive
            res=least_squares(costfeta,np.zeros(S) )
            mat[sidx]=res.x
    else:
        # # code_debugger()
        # if 'groundtruth' in kwargs:
        #         gg=kwargs['groundtruth']
        #         gg=gg[np.eye(S)==0]
        #         print moment(gg, 4) / 3 / moment(gg, 2) ** 2

        for sidx in range(S):
            res = None
            notsidx= [s for s in range(S) if s!=sidx]
            alive=(N[sidx]>death)
            xsT = N[:,alive]
            xnomono =   N[:,(N[sidx]!=0)&(np.max(N[notsidx],axis=0)>death )]
            if not xnomono.shape[0]:
                print('hyperplane skipping', s, 'only present in monoculture')
                mat[sidx, sidx] = -1
                continue
                # xsT = xs.T
            #
            def costfeta(y, weights=None):
                yy = -np.ones(S)
                yy[notsidx] = link(y)
                if weights is None:
                    return np.dot(yy, xsT) + K[sidx,alive] #, (moment(yy,4)/3/moment(yy,2)**2 - 1)*np.ones(S ) ) )
                else:
                    return (np.dot(yy, np.sum(weights * xsT, axis=1) / np.sum(weights)) + K[sidx,alive])
            #
            res = least_squares(costfeta, -np.zeros(S - 1))
            row = list(link(res.x))
            row.insert(sidx, -1)
            mat[sidx] = row
            mat[sidx, np.mean(np.abs(xsT), axis=1) <=death] = np.nan
            # code_debugger()

    return mat

def gaussianfromcorr(corr,Env,init=None):#,scale=1):
    from numpy.linalg import lstsq, norm as lanorm
    from scipy.optimize import least_squares
    S=corr.shape[0]
    def cost(args):
        ctest,wtest=args[:S],args[S:2*S]#,args[2*S:]
        Ngauss=np.exp(-(Env.reshape((1,-1))-ctest.reshape((-1,1)) )**2/(2*np.maximum(1,wtest).reshape((-1,1)) **2) )#*Nmax.reshape((-1,1))
        naive=np.corrcoef(Ngauss)#/scale
        return la.norm(naive-corr)
    if init is None:
        init=np.ones(2*S)
    args= least_squares(cost,init ).x
    ctest,wtest=args[:S],args[S:2*S]#,args[2*S:]
    return ctest,wtest#,Nmax



from numpy import sqrt, pi, exp, log, sin, cos
from scipy.special import erf, binom

def erfmom(mom, mean, var, lim=0):
    # Moment of error function
    var = max(var, 10 ** -5)
    xx = mean / sqrt(2 * var)
    mom0 = .5 * (erf(xx) + 1)
    mom1 = sqrt(var / 2 / pi) * exp(-xx ** 2)
    if mom == 0:
        return mom0
    elif mom == 1:
        return mean * mom0 + mom1
    elif mom == 2:
        return (var + mean ** 2) * mom0 + mean * mom1


def bunin_solve(S=100, mu=1, sigma=1, sigma_k=1, gamma=1, tol=10 ** -5, **kwargs):
    import scipy.optimize as sopt
    u = (1 - mu * 1. / S)
    def calc_v(phi):
        psg = np.clip(phi * sigma ** 2 * gamma, None, u ** 2 / 4.00000001)
        if np.abs(psg) < 10 ** -6:
            v = 1. / u
        else:
            v = (u - np.sqrt(u ** 2 - 4 * psg)) / (2 * psg)
        return v
    def eqs(vec):
        N1, N2 = np.abs(vec[:2])  # N1, N2 must be positive
        phi = np.exp(vec[2]) / (1 + np.exp(vec[2]))  # phi must be between 0 and 1
        v = calc_v(phi)
        mean = v * (1 - mu * N1 * phi)
        var = np.clip(v ** 2 * (sigma_k ** 2 + sigma ** 2 * N2 * phi), 0.001, None)
        m0, m1, m2 = erfmom(0, mean, var), erfmom(1, mean, var), erfmom(2, mean, var)
        eq1 = (phi - m0)
        eq2 = (phi * N1 - m1)
        eq3 = (phi * N2 - m2)
        res = np.array((eq1, eq2, eq3))
        return res
    root = sopt.root
    x0 = kwargs.get('x0', np.random.random(3))
    res = root(eqs, x0, tol=tol, method='df-sane')
    # EVERYTHING THAT FOLLOWS HAS TO DEAL WITH ROOTFINDING FINNICKINESS
    trials = 50
    while not res.success and trials > 0:
        # IF ROOT FINDING DOES NOT WORK (MAYBE DIVERGENCE, MAYBE BAD INITIAL CONDITION)
        if sigma > .7 and mu > -1 and trials < 30:
            # IF LARGE SIGMA, GET INITIAL CONDITION FROM SLIGHTLY LOWER SIGMA
            x0 = bunin_solve(S=S, mu=mu, sigma=sigma / 1.01, sigma_k=sigma_k, gamma=gamma, tol=tol)[:3]
            if np.isnan(x0).any():
                return np.nan * np.ones(5)
            res = root(eqs, x0, tol=tol)
            break
        else:
            x0 = np.random.random(3)
            res = root(eqs, x0, tol=tol)
            trials -= 1
    N1, N2, phi = res.x
    N1, N2 = np.abs(N1), np.abs(N2)
    phi = np.exp(phi) / (1 + np.exp(phi))
    if not res.success or N1 > 5:
        return np.nan * np.ones(5)
    return N1, N2, phi, calc_v(phi), res.fun

def paramsfromdist(Nflat,Kflat,together=1,nsamp=10,ntest=50,death=10**-5,tmax=1000,nepochs=10,**kwargs):
    def offdiag(x):
        return x[np.eye(x.shape[0]) == 0]

    import scipy.linalg as la
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV

    def persite(N,K,refS):
        # print N.shape,K.shape
        #N=N/np.maximum(K,death)
        #K=np.ones(K.shape)
        #N=N[K>10*death]
        #K=K[K>10*death]
        S=N.shape[0]
        # print S,refS
        if S<1:
            return np.zeros(N.shape)
        def perprm(A):
            from scipy.integrate import solve_ivp
            def eqs(t,x):
                dx= x*(K- np.dot(A,x))
                dx[x<death]=np.maximum(dx[x<death],0)
                return dx
            sol = solve_ivp(eqs, (0, tmax), np.maximum(1,K) )
            if not sol.success:
                return None
            #print sol.y.shape
            #plt.close('all')
            #plt.plot(sol.y.T)
            #plt.show()
            return sol.y[:,-1]
        comps=[]
        fails=[]
        rg=(0,refS),(0,3)
        bestN,bestdist=None,10**15
        for epoch in range(nepochs):
            for test in range(int(round(ntest/nepochs))):
                #print 'Paramsfromdist',test
                mu=np.random.uniform(*rg[0])
                sigma=np.random.uniform(*rg[1])
                dic={'mu':mu,'sigma':sigma,'alive':refS}
                A=np.random.normal( mu/S ,sigma/S**.5, (S,S) )
                np.fill_diagonal(A,1)
                Ncomp=perprm(A)
                if Ncomp is None:
                    fails.append(dic)
                    continue
                dist=la.norm(np.sort(N)-np.sort(Ncomp))
                dic['mnN']=np.mean(Ncomp)
                dic['mndist']=np.abs(dic['mnN']- np.mean(N))
                dic['sdN']=np.std(Ncomp)
                dic['sddist']=np.abs(dic['sdN']-np.std(N))
                dic['phiN']=np.mean(Ncomp>2*death)
                dic['phidist']=np.abs(dic['phiN']-np.mean(N>2*death))
                dic['dist']=dist
                dic['statdist']= dic['mndist']+ dic['sddist']+dic['phidist']
                comps.append(dic)
                if dic['dist']<bestdist:
                    bestdist=dic['dist']
                    bestN=Ncomp

            tmp=pd.DataFrame(comps)
            tmp=tmp[tmp.dist<np.percentile(tmp.dist,25)]
            if tmp.shape[0]>0:
                rg=(0,tmp.mu.max()),(0,tmp.sigma.max())
                print(rg)
            else:
                print(tmp.dist)
        comps=pd.DataFrame(comps)
        #print('Fails:{}'.format(len(fails)) )
        # code_debugger()
        if 0:
            plt.close('all')
            plt.hist(N, bins=50, cumulative=1)
            plt.hist(bestN, bins=50, alpha=.5, cumulative=1)
            plt.show()
        if 0:
            plt.close()
            plt.subplot(121)
            plt.colorbar(plt.scatter(comps.mu, comps.sigma, c=np.log10(comps.dist))),
            plt.subplot(122)
            plt.scatter(comps.dist,comps.statdist), plt.show()
        best=comps[comps.dist<= np.percentile(comps.dist,5) ]
        df=  pd.DataFrame({'mu':[best.mu.median() ],'sigma':[best.sigma.median()] ,'alive':[S] })
        #best2=comps[comps.statdist< np.percentile(comps.statdist,5) ]
        # best2=comps.iloc[np.argmin(comps.statdist)]
        #df2=  pd.DataFrame({'mu':[best2.mu.median() ],'sigma':[best2.sigma.median() ]})
        #print df.values,df2.values
        df['mnA']=df['mu']/refS
        df['sdA']=df['sigma']/np.sqrt(refS)
        if 'groundtruth' in kwargs:
            trueA = -kwargs['groundtruth']
            tmp=(K[:trueA.shape[0]]>2*death)
            trueA=trueA[np.ix_(tmp,tmp )]
            mnA,sdA=np.mean(offdiag(trueA)),np.std(offdiag(trueA))
            #print('true',mnA,sdA,'vs',df.mnA.values[0],df.sdA.values[0])
        #code_debugger()
        return df
    if together:
        Ns,Ks=list(zip(*[(Nflat[:,i],Kflat[:,i] ) for i  in np.random.randint(0,Kflat.shape[1],nsamp) ]))

        return persite(np.array(Ns).ravel(),np.array(Ks).ravel() ,Nflat.shape[0])
    else:
        results=[persite(Nflat[:,i],Kflat[:,i],Nflat.shape[0] ) for i  in np.random.randint(0,Kflat.shape[1],nsamp) ]
        return pd.concat(results)

def paramsfromstats(etamean,etasd,Ksd,alive,S,ntrials=3):
    from scipy.optimize import least_squares
    table=[]
    dft=np.ones(2)/4.
    for z in range(ntrials):
        idx=np.random.randint(len(etamean))
        mn,sd,ksd,al=etamean[idx],etasd[idx],Ksd[idx],alive[idx]  #np.random.normal(np.mean(etamean),np.std(etamean)),np.random.normal(np.mean(etasd),np.std(etasd))
        print(z,mn,sd,ksd,al)
        def cost(x):
            mu,sigma=np.abs(x)
            N1,N2,phi=bunin_solve(mu=mu,sigma=sigma,sigma_k=ksd,gamma=0)[:3]
            return np.array([N1-mn, N2 - mn**2 - sd**2, phi-al*1./S ])
        try:
            res=least_squares(cost,dft)
        except:
            continue
        if res.success:
            dft=np.abs(res.x)
            dic={'mu':dft[0],'sigma':dft[1],'mnA':dft[0]/S,'sdA':dft[1]/np.sqrt(S),'alive':al}
            table.append(dic)
    table=pd.DataFrame(table)
    return table


# ============= PLOTS =================



def detailed_plots(path,save=0,movie=0,**kwargs):
    def offdiag(x):
        x=np.array(x)
        return x[np.eye(x.shape[0]) ==0]
    if save:
        if isinstance(save, str) or isinstance(save, Path):
            outpath = save
        else:
            outpath = path + Path('plots')

    files=pd.read_csv(path+'files.csv')
    table=[]
    for idx,f in list(files.iterrows())[::-1]:
        if 'debug_predniche' in sys.argv:
            if not 'envniche_pos_range-(40.0, 50.0)' in f['path']:
                continue
            if not 'competition_mean-0.3' in f['path'] or not 'competition_relstd-1' in f['path']:
                continue
        model=LandscapeModel.load(f['path'])
        print('Plotting',f['path'])
        dic = dict(model.export_params())
        dic['path'] = f['path']
        B, Bbasal, D, DS=basic_measures(model, dic)
        table.append(dic)
        figs = {}

        prm,data=model.prm,model.data
        death=prm['death']

        if save:
            fpath=Path(outpath+'_'.join([n for n in Path(f['path']).split() if not n in Path(outpath).split() ]))
            fpath.mkdir()

        S=prm['species']
        Env=data['environment'][:,:,0].ravel()
        Env_discrete=np.round(Env,0)
        size=data['size']
        niche=data['envniche_pos'][:,0]
        nichewid=data['envniche_width'][:,0]

        nf=model.results['n'][-1]
        deth=5*death
        alive = [i for i in range(S) if np.max(nf[i]) > deth]
        if len(alive)==0:
            print("ALL SPECIES ARE DEAD")
            continue
        ordalive = [alive[z] for z in np.argsort(niche[alive])]
        Nalive = len(alive)
        Ks=np.array([ ( data['growth'][i] - data['mortality'][i])/data['competition'][i,i] for i in range(S)])
        Ns=nf[ordalive]
        Ks=Ks[ordalive]
        size,niche,nichewid=[z[ordalive] for z in (size,niche,nichewid)]

        if len(Ns.shape)<3:
            Ns=Ns.reshape((1,)+Ns.shape)
        if len(Ks.shape)<3:
            Ks=Ks.reshape((1,)+Ks.shape)
        mat=-data['competition'][np.ix_(ordalive,alive)]
        if prm['trophic']['ON']:
            tmat=data['trophic'][np.ix_(ordalive,ordalive)]
            mat-=tmat.T-prm['trophic']['efficiency']*tmat
        V=-la.inv(mat)
        Vii=np.diag(V)
        Vtree=-mat*np.multiply.outer(Vii,Vii)
        # V=Vtree

        #Copresence
        Nflat = np.array([Ns[i, :, :].ravel() for i in range(Ns.shape[0])  ])
        Kflat = np.array([Ks[i].ravel() for i in range(Ks.shape[0]) ])
        
        
        dic['normKN']=la.norm( (1- Kflat/(Nflat+0.01*np.max(Kflat) ) ))
        compositions={}
        refN=np.max(Nflat,axis=1)*0.1
        for val in set(Env_discrete):
            test=(Env_discrete==val)
            try:
                compositions[val]=tuple(np.where(np.mean(Nflat[:,test],axis=1)>refN )[0] )
            except:
                compositions[val]=()
        
        def fillv(content,pos):
            Vtmp=np.zeros((Nalive,Nalive))
            Vtmp[np.ix_(pos,pos)]=content
            return Vtmp

        complist=sorted(set(compositions.values()))

        V_comp={c:fillv(-la.inv(mat[np.ix_(c,c)] ),c) for c in complist}
        V_Env={E:V_comp[c] for E,c in list(compositions.items()) }
        Ntot = np.array([np.mean(nf[i][nf[i]>death ] ) for i in range(S)])
        copresence = np.dot((Nflat > deth) * 1., (Nflat > deth).T) / Nflat.shape[1]
        copnorm = np.multiply.outer(np.diag(copresence), np.diag(copresence)) ** .5
        copresence = copresence / copnorm
        gflat = np.array([ data['growth'][i].ravel() for i in ordalive])
        seenK = np.dot((Nflat > deth) * 1., (Kflat * (Nflat > deth)).T) / Nflat.shape[1] / copnorm
        
        Vcopnorm = (.001+np.sum([(V_Env[E]!=0)*np.sum(Env_discrete==E) for E in compositions],axis=0 ))
        Vcop = np.sum([V_Env[E]*np.sum(Env_discrete==E) for E in compositions],axis=0 )/ Vcopnorm

        Vstd =( np.sum([(V_Env[E]-Vcop)**2*np.sum(Env_discrete==E) for E in compositions],axis=0 )/Vcopnorm )**.5

        Viicop = np.diag(Vcop)


        covN=np.cov(Nflat)
        covK=np.cov(Kflat)
        corrN=np.corrcoef(Nflat)
        corrK=np.corrcoef(Kflat)
        
        xx,yy=(offdiag(corrN),offdiag(mat))
        figs["invcorr_vs_mat"]=plt.figure()
        plt.scatter(xx,yy),plt.xlabel('Inverse corr matrix'),plt.ylabel('True interactions')
        dic["R_invcorr_vs_mat"]=spearmanr(xx,yy)[0]

        Vresc=Vcop/np.diag(Vcop).reshape((-1,1))
        corrN_from_Vcop=0.5*(Vresc+Vresc.T )
        #Vsym_from_corrN=
        from scipy.optimize import least_squares
        def costcnK(y):
            return (offdiag(covN)-offdiag(np.multiply.outer(y,y) * covK) )

        coefsNK = least_squares(costcnK,np.ones(Nalive) ).x
        cov_residual=covN-np.multiply.outer(coefsNK,coefsNK) * covK

        def costcornK(y):
            return (offdiag(corrN)-offdiag(np.multiply.outer(y,y) * corrK) )
        try:
            coefscorNK = least_squares(costcornK,np.ones(Nalive)*.3 ).x
        except:
            code_debugger()
        corr_residual=corrN-np.multiply.outer(coefscorNK,coefscorNK) * corrK


        def mysd(xx):
            xx = xx[~np.isnan(xx)]
            xx = xx[~np.isinf(xx)]
            return (np.percentile(xx, 84) - np.percentile(xx, 16)) / 2.



        def mysym(xx):
            good = (np.eye(xx.shape[0])<1)&(~np.isnan(xx))&( ~np.isinf(xx)) & (~np.isnan(xx.T))&( ~np.isinf(xx.T))
            return np.corrcoef(xx[good],xx.T[good])[0,1]


        #================= PLOTS ===============================

        envdim=data['environment'].shape[2]
        figs['Environment'], subs = plt.subplots(envdim, 3, gridspec_kw={'width_ratios': [6, 1,4]},
                                               figsize=np.array(mpfig.rcParams['figure.figsize']) * (2, envdim))

        for dim in range(envdim):
            Envloc=data['environment'][:,:,dim]
            if envdim>1:
                a0,a1,a2=subs[dim]
            else:
                a0,a1,a2=subs
            # panel, ax = auto_subplot(panel, 3,rows=1)
            #plt.colorbar(
            a0.imshow(Envloc)#,ax=a0)
            a0.set_xticks([]),a0.set_yticks([])
            a0.set_title('Landscape')


            Emin,Emax=np.min(Envloc),np.max(Envloc)
            c,w=data['envniche_pos'][:,dim], data['envniche_width'][:,dim]
            norm = matplotlib.colors.Normalize(vmin=Emin, vmax=Emax)
            cmap = matplotlib.cm.get_cmap()


            _, bins, patches = a1.hist(Envloc.ravel(),bins=20,color='k',
                           orientation='horizontal',  )

            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - min(bin_centers)
            col /= max(col)
            for cp, p in zip(col, patches):
                plt.setp(p, "facecolor", cmap(cp))

            a1.set_xticks([])
            a1.set_title('Env.')
            #ax.set_aspect(4.)
            # panel, ax = auto_subplot(panel, 3,rows=1)

            for x,y in enumerate(c):
                a2.vlines(x,y-w[x],y+w[x] ,color=cmap(norm(y)))

            a2.scatter(np.arange(S),c,c=c, vmin=Emin,vmax=Emax)
            a2.set_title('Abiotic niche'),a2.set_xlabel('Species')
            a2.set_xticks(np.arange(0,S,5) )
            rg= min(Emin, np.min(data['envniche_pos'][:,dim]-data['envniche_width'][:,dim])),max(Emax, np.max(data['envniche_pos'][:,dim]+data['envniche_width'][:,dim]))
            a1.set_ylim(ymin=rg[0],ymax=rg[1])
            a2.set_ylim(ymin=rg[0],ymax=rg[1])
        figs['Environment'].tight_layout()




        #Plots of species abundance per patch
        if save and movie:
            ts=list(range(model.results['n'].shape[0]))
        else:
            ts=[0,-1]
        for t in ts:
            fig = plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (3, 3))
            if t==0:
                figs['N_initial']=fig
                plt.suptitle('Initial abundance')
            panel=0
            maxN=np.max(model.results['n'][t] / death)
            for i in range(S):
                panel,ax=auto_subplot(panel,S)
                val=model.results['n'][t][i]/death
                plt.title('Species {}'.format(i))
                #if np.max(val)<=2 or np.isnan(np.max(val)):
                #    continue
                plt.imshow(val ,vmin=0,vmax=maxN)
                plt.colorbar()
            if save and movie:
                print('Movie frame {}'.format(t))
                mpath = fpath + Path('movie')
                fig.savefig(mpath.mkdir() + 'fig{}.png'.format(t))
                if t != ts[0] and t != ts[-1]:
                    plt.close(fig)

        figs['N_final']=fig
        plt.suptitle('Final abundance')


        #Species trajectories,
        figs['N_traj']=plt.figure()
        plt.subplot(121)
        nsum=np.sum(model.results['n']/death, axis=(2, 3))
        for i in zip(*nsum):
            plt.plot(model.results['t'],i,alpha=0.5)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Time')
        plt.title('Total abundance per species')
        plt.subplot(122)
        for i in range(S):
            plt.plot(model.results['t'],model.results['n'][:,i,0,0]/death,alpha=0.5)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Time')
        plt.title('Species abundances in 1 patch')
        

        figs['growth']=plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (3, 3))
        plt.suptitle('Growth')
        panel=0
        for i in range(S):
            panel,ax=auto_subplot(panel,S)
            val=data['growth'][i]
            plt.title('Species {}'.format(i))
            #if np.max(val)<=2 or np.isnan(np.max(val)):
            #    continue
            plt.imshow(val ,vmin=0,vmax=np.max(data['growth']))
            plt.colorbar()

        
        # Environmental niche
        figs['infer_niche']=plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (2, 2))
        from scipy.optimize import minimize

        X=Env
        X_rg=np.max(X)-np.min(X)
        X_plot=np.linspace(np.min(X),np.max(X),20)
        X_ctr= (X_plot[:-1]+X_plot[1:])/2
        panel=0
        nicheprms=[]
        nicheprms2=[]
        for i in range(S):
            if i in alive:
                z=alive.index(i)
            else:
                continue
            bounds=[ (np.max(Nflat[z])/Nalive, np.max(Nflat[z])*Nalive), (np.min(X),np.max(X)), (X_rg/S,X_rg*2) ]
            panel,ax=auto_subplot(panel,S)
            #SECOND METHOD
            Ks_z=[]
            Nsmax_z=[]
            for X_bin in zip(X_plot[:-1],X_plot[1:]):
                loc=(X>=X_bin[0])&(X<=X_bin[1])
                y=Nflat[z][loc]
                ytot =np.sum(Nflat[:,loc],axis=0)-Nflat[z][loc]
                Ks_z.append( linregress(ytot,y)[1]  ) 
                softmax=np.mean(y**4)**0.25 
                softmax=np.percentile(y,90)
                Nsmax_z.append( softmax ) 
            Ks_z=np.array(Ks_z)
            Nsmax_z=np.array(Nsmax_z)

            # FIRST METHOD
            def gaussian(x,*popt):
                return popt[0]*np.exp(- (x-popt[1])**2/(2*popt[2]**2 ) )
            if 0:
                def fun(zz):
                    if (zz<0).all():
                        #w0=0.01
                        pos=np.zeros(1)
                    else:
                        #w0=0.01+  np.mean(zz[zz>0]) 
                        pos= zz[zz>0]
                    def negfun(w):
                        if not len(w):
                            return np.zeros(1)
                        return np.max(w**2) *np.ones(1) 
                        #return w**2/(1+w**2/w0**2)
                    return np.concatenate([  negfun( zz[zz<=0]), pos**2])
                def fit(popt):
                    return np.sum(fun( ( Nflat[z]- gaussian(X,*popt) )/np.mean(Nflat[z]) ))
            else:
                def fit(popt):
                    return np.sum(  (Nsmax_z - gaussian(X_ctr,*popt) )**2 / np.mean(Nsmax_z)**2 )
            succ=False
            X0=x0=np.ones(3)*[np.max(Nsmax_z),X_ctr[np.argmax(Nsmax_z)],np.std(X_ctr[np.where(Nsmax_z>np.max(Nsmax_z)/3)[0] ])  ]
            strials=0
            sol,ref=None,np.inf
            while not succ:
                lsol= minimize(fit,x0,bounds=bounds )
                #print(lsol)
                popt=lsol.x
                #popt=np.array([np.clip(popt[i],*bounds[i]) for i in range(len(popt))])
                succ=(lsol.fun<1)
                x0=np.array([np.random.uniform(b[0],b[1]) for b in bounds ])
                x0=np.mean([x0,X0],axis=0)
                if lsol.fun<ref:
                    ref=lsol.fun
                    sol=lsol
                strials+=1
                if strials>10 and ref <2:
                    break
                if strials>30:
                    break
            #PLOT
            print(strials)
            plt.title('Species {}'.format(alive[z]))
            #plt.plot(X_ctr,Nsmax_z,color='b',lw=2,linestyle='--',alpha=.3)
            ymax=max(np.max(Nflat),np.max(Kflat))
            plt.ylim(ymax=ymax*1.1,ymin=np.min(Kflat)-0.1*ymax)
            plt.scatter(X,Nflat[z],marker='x',alpha=.1)
            plt.plot(np.sort(X),Kflat[z][np.argsort(X)],color='g',lw=5,alpha=.5)
            plt.plot(X_plot,gaussian(X_plot,*popt),color='k',lw=2)
            #plt.show()
            error1=np.mean( (Kflat[z]-gaussian(X,*popt) )**2)**.5 
            error2= np.mean((Ks_z-gaussian(X_ctr,*popt))**2)**.5  
            #if error2< 1:
            #    plt.plot(X_plot[1:],Ks_z,color='r',linestyle='--')
            #    plt.show()
            r2=spearmanr(Kflat[z],gaussian(X,*popt))[0]**2
            nicheprms.append(tuple(popt) +   ( error1,error2,r2  ))
        nicheprms=np.array(nicheprms)
        plt.tight_layout()
        
        figs['infer_niche_prms']=plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (1.5, .75))
        plt.subplot(131)
        plt.title('Niche center')
        plt.ylabel('Inferred niche')
        rg=(np.min(niche),np.max(niche))
        plt.plot(rg,rg,linestyle='--',color='k')
        plt.scatter(niche,nicheprms[:,1])
        plt.subplot(132)
        plt.title('Niche width')
        rg=(np.min(nichewid),np.max(nichewid))
        plt.plot(rg,rg,linestyle='--',color='k')
        plt.scatter(nichewid,nicheprms[:,2])
        plt.xlabel('Fundamental niche')
        plt.subplot(133)
        plt.title('Maximum capacity')
        maxK=np.max(Kflat,axis=1)
        rg=(np.min(maxK),np.max(maxK))
        plt.plot(rg,rg,linestyle='--',color='k')
        plt.scatter(maxK,nicheprms[:,0])


        dic['infer_niche_error']=np.median( nicheprms[:,3] )
        dic['infer_niche_R2']=np.median( nicheprms[:,5] )
        dic['infer_niche_error_interceptmethod']=np.median( nicheprms[:,4] )
        dic['infer_niche_height_error']=np.median( (maxK-nicheprms[:,0])**2 )**.5
        dic['infer_niche_center_error']=np.median( (niche-nicheprms[:,1])**2 )**.5
        dic['infer_niche_width_error']=np.median( (nichewid-nicheprms[:,2])**2 )**.5
        dic['infer_niche_height_R2']=spearmanr(maxK,nicheprms[:,0])[0]**2
        dic['infer_niche_center_R2']=spearmanr(niche,nicheprms[:,1])[0]**2
        dic['infer_niche_width_R2']=spearmanr(nichewid,nicheprms[:,2])[0]**2


        # COMPOSITIONS


        figs['composition']=plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (2, 3))
        

        compdic={c:i for i,c in enumerate(complist)}
        compo=np.array([ compdic.get( tuple(np.where(Nflat[:,z]>refN )[0] ),0 ) for z in range(Nflat.shape[1]) ]).reshape(Ns[0].shape)
        compfreq={c:np.sum(compo==c) for c in list(compdic.values()) }
        #if 0 in compfreq:
            #del compfreq[0]
        compfreq[0]=0
        #goodcomps=[ c for c in compfreq if  compfreq[c]> np.mean(list(compfreq.values())) and  s1 in complist[c] and s2 in complist[c] ]
        #icompo1,icompo2=np.random.choice([ c for c in goodcomps if  compfreq[c]> np.median([compfreq[z] for z in goodcomps] ) ] ,size=2,replace=0)
        
        
        abundantspp=np.where(np.sum(Nflat,axis=1)>np.median(np.sum(Nflat,axis=1)))[0]
        ss=(0,1)
        ref=0
        for s1 in abundantspp:
            for s2 in abundantspp:
                if s2==s1:
                    continue
        #Vstdtmp=Vstd[np.ix_(abundantspp,abundantspp)].copy()
        #np.fill_diagonal(Vstdtmp,0)
        #s1,s2=[abundantspp[z] for z in np.unravel_index(np.argmax(Vstdtmp),Vstdtmp.shape)]
                goodcomps=[i for i in complist  if s1 in i and s2 in i and compfreq[complist.index(i)]>0 ]
                if not len(goodcomps):
                    continue
                Vs1s2=[V_comp[i][s1,s2] for i in goodcomps]
                new=np.max(Vs1s2)-np.min(Vs1s2)
                if new>ref:
                    ss=(s1,s2)
                    ref=new
        s1,s2=ss
        goodcomps=[i for i in complist  if s1 in i and s2 in i and compfreq[complist.index(i)]>0] 
        Vs1s2=[V_comp[i][s1,s2] for i in goodcomps]
        compo1,compo2=goodcomps[np.argmin(Vs1s2)],goodcomps[np.argmax(Vs1s2)]
        icompo1,icompo2=complist.index(compo1),complist.index(compo2)
        
        loc1=list(zip(*np.where(compo==icompo1)))
        loc2=list(zip(*np.where(compo==icompo2)))
        compo1=complist[icompo1]
        compo2=complist[icompo2]
        
        dic['#compositions']=np.max(compo)

        plt.subplot(3,3,1)
        plt.title('Number of surviving species')
        Salive=np.sum(Ns>deth,axis=0)
        plt.imshow(Salive ,vmin=0,vmax=S)
        dic['Salive']=np.mean(Salive)
        dic['ncompos']=len(np.unique(compo))
        plt.colorbar()
        plt.subplot(3,3,2)
        plt.title('Number of surviving species')
        plt.hist(Salive.ravel(),density=1)
        plt.subplot(3,3,3)
        plt.title('Composition')
        plt.imshow(compo )
        plt.colorbar()
        plt.scatter(*[(y,x) for x,y in loc1][len(loc1)//2],c='r',marker='x')
        plt.scatter(*[(y,x) for x,y in loc2][len(loc2)//2],c='w',marker='x')
        plt.subplot(3,3,4)
        plt.title('Set {}: {}'.format(icompo1,compo1))
        mat1=V_comp[compo1]
        plt.colorbar(plt.imshow(mat1, cmap='seismic_r', vmin=-np.max(np.abs(mat1)), vmax=np.max(np.abs(mat1))
                                ), ax=plt.gca())
        plt.subplot(3,3,5)
        plt.title('Set {}: {}'.format(icompo2,compo2))
        mat2=V_comp[compo2]
        plt.colorbar(plt.imshow(mat2, cmap='seismic_r', vmin=-np.max(np.abs(mat2)), vmax=np.max(np.abs(mat2))
                                ), ax=plt.gca())
        
        plt.subplot(3,3,6)
        #s1,s2=np.argsort(np.sum(Nflat,axis=1))[-2:]
        #s1,s2=sorted([s1,s2])
        plt.title('V_{},{}'.format(s1,s2))
        plt.hist([ V_comp[complist[c]][s1,s2] for c in compo.ravel() if s1 in complist[c] and s2 in complist[c]] ,density=1,bins=5)
        plt.axvline(mat[s1,s2],color='k',linestyle='--')
        plt.axvline(-la.inv(mat)[s1,s2],color='r',linestyle='--')


        plt.subplot(3,3,7)
        plt.title('Set {}: {}'.format(icompo1,compo1))
        mat1=np.zeros((Nalive,Nalive))
        mat1[np.ix_(compo1,compo1)]=mat[np.ix_(compo1,compo1)]
        plt.colorbar(plt.imshow(mat1, cmap='seismic_r', vmin=-np.max(np.abs(mat1)), vmax=np.max(np.abs(mat1))
                                ), ax=plt.gca())
        plt.subplot(3,3,8)
        plt.title('Set {}: {}'.format(icompo2,compo2))
        mat2=np.zeros((Nalive,Nalive))
        mat2[np.ix_(compo2,compo2)]=mat[np.ix_(compo2,compo2)]
        plt.colorbar(plt.imshow(mat2, cmap='seismic_r', vmin=-np.max(np.abs(mat2)), vmax=np.max(np.abs(mat2))
                                ), ax=plt.gca())
        #if np.max(compo)<40:
        #    plt.show()
        #    code_debugger()
        #else:
        #    plt.close('all')
        #continue

        ### CORRELATIONS AND COVARIANCES

        figs['analytical_approx']=plt.figure()
        plt.suptitle('Analytical approximations')
        plt.subplot(131),plt.title('corr N from V')
        plt.scatter(offdiag(corrN),offdiag(corrN_from_Vcop)),plt.xlabel('corr(Ni,Nj)')
        plt.subplot(132),plt.title('Vhat_ij from rho')
        invrho=np.diag(la.inv(corrN))
        Vhat=np.dot(np.diag(invrho)**.5, np.dot(corrN, np.diag(invrho)**.5) )
        plt.scatter(offdiag(Vcop),offdiag(Vhat) ),plt.xlabel("Vwei")
        plt.subplot(133),plt.title('Vhat_ii')
        plt.scatter(np.diag(Vcop),np.diag(Vhat) ),plt.xlabel("Vwei")

        for title,lst in [('covariance',[covK,covN,coefsNK,cov_residual]),
                          ('correlation',[corrK,corrN,coefscorNK,corr_residual]) ]:
            vecK,vecN,veccoef,vecres=lst
            figs[title] = plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (2,2))
            plt.suptitle('f='+title)
            plt.subplot(221)
            vmax = np.max(np.abs(vecN))
            if 'corr' in title:
                vmax=1
            def mask_diag(x):
                x=x.copy()
                x[np.eye(x.shape[0])!=0]=np.nan
                return x
            plt.colorbar(plt.imshow(mask_diag(vecN), cmap='seismic_r', vmin=-vmax, vmax=vmax),
                         ax=plt.gca())
            plt.title(r'$f(N)$')
            plt.subplot(222)
            vmax =np.max(np.abs(vecK))
            if 'corr' in title:
                vmax=1
            plt.title(r'$f(K)$')
            plt.colorbar(plt.imshow(vecK, cmap='seismic_r', vmin=-vmax, vmax=vmax,
                                    ), ax=plt.gca())
            plt.subplot(223)
            plt.scatter(Vii,veccoef),plt.xlabel('Vii'),plt.ylabel('ci'),plt.title('f(N) ~ ci cj f(K)')
            plt.subplot(224)
            plt.title(r'residual f(N) - ci cj f(K)')
            vmax = np.max(np.abs(vecres))
            if 'corr' in title:
                vmax=1
            plt.colorbar(plt.imshow(mask_diag(vecres), cmap='seismic_r', vmin=-vmax, vmax=vmax,
                                    ), ax=plt.gca())



        figs['infer_matrix']=plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (1,1.35))
        infermats={}
        tobreak=0
        for labl,Vuse in zip(('V','A'), (V,mat)):
            shift=0
            if 'A' in labl:
                shift=2
            plt.subplot(2,2,1+shift),plt.title(ifelse('A' in labl,'A',"V")+ '_ij from full Ki(x)')
            Vinfer= hyperplane_light(Nflat,Kflat,deth,mode=ifelse('A' in labl,'A','V'),regularize=not 'A' in labl)
            infermats[labl]=Vinfer
            xs,ys=offdiag(Vuse),offdiag(Vinfer)
            good=(~np.isnan(xs))&(~np.isnan(ys))&(~np.isinf(xs))&(~np.isinf(ys))
            xs,ys=xs[good],ys[good]
            slope, intercept, r, p, stderr = linregress(xs, ys)
            plt.scatter(xs,ys,alpha=.4,label=labl+' R:{:.2g}'.format(r)),plt.xlabel('True'), plt.ylabel('Inferred')
            rg=np.array((np.min(xs),np.max(xs)))
            #for k,v in zip('mnVij,sdVij,mnVijpred,sdVijpred'.split(','), (np.nanmean(xs),mysd(xs),np.nanmean(ys),mysd(ys))):
            #    dic[k+labl]=v
            plt.legend()#['True: mn {:.2g} sd {:.2g}\n'.format(np.mean(xs),mysd(xs))+ 'Inferred: mn {:.2g} sd {:.2g}'.format(np.mean(ys),mysd(ys)) ])
            plt.plot(rg,slope*rg+intercept,linestyle='--',lw=2)
            if shift:
                plt.subplot(2, 2, 2 + shift), plt.title('Aii hist')
                rg=min(np.min(xs),np.min(ys)),max(np.max(xs),np.max(ys))
                bins=np.linspace(rg[0],rg[1],11)
                plt.hist(xs,bins=bins,log=1);plt.hist(ys,bins=bins,alpha=.5,log=1)
                plt.legend(['True','Inferred'])
            else:
                plt.subplot(2,2,2+shift),plt.title('Vii')
                xs,ys=np.diag(Vuse),np.diag(Vinfer)
                good=(~np.isnan(xs))&(~np.isnan(ys))&(~np.isinf(xs))&(~np.isinf(ys))
                xs,ys=xs[good],ys[good]
                slope, intercept, r, p, stderr = linregress(xs, ys)
                plt.scatter(xs,ys,alpha=.4,label=labl+' R:{:.2g}'.format(r)),plt.xlabel('True')
                rg=np.array((np.min(xs),np.max(xs)))
                plt.legend()#['True: mn {:.2g} sd {:.2g}\n'.format(np.mean(xs),mysd(xs))+ 'Inferred: mn {:.2g} sd {:.2g}'.format(np.mean(ys),mysd(ys)) ])
                plt.plot(rg,slope*rg+intercept,linestyle='--',lw=2)
                #for k,v in zip('mnVii,sdVii,mnViipred,sdViipred'.split(','), (np.nanmean(xs),mysd(xs),np.nanmean(ys),mysd(ys))):
                #    dic[k+labl]=v
        plt.subplot(221)
        plt.legend()
        plt.subplot(222)
        plt.legend()
        if tobreak:
            continue

        def nonzero(x):
            return x[x!=0]
        Vstar=np.concatenate([nonzero(offdiag(z)) for z in list(V_Env.values())])
        for label, data in zip( ['Apred','Vpred','A','V','V*','covresidual','Vcop','cNK','corNK','covK','covN'],
                                [offdiag(-infermats['A']),offdiag(infermats['V']),offdiag(-mat),offdiag(V),Vstar,
                                       offdiag(cov_residual), offdiag(Vcop),coefsNK, coefscorNK,offdiag(covK),offdiag(covN)] ):
            dic[label+'_mn']=np.nanmean(data)
            dic[label+'_mnabs']=np.nanmean(np.abs(data))
            dic[label+'_std']=mysd(data)
        del Vstar
        
        for label, data in zip( ['Apred','Vpred','A','V','V*'],
                                [(-infermats['A']),(infermats['V']),(-mat),(V),
                                       np.nanmean(list(V_Env.values()),axis=0),] ):
            dic[label+'_symmetry']=mysym(data)

        def corr(x,y):
            good=(~np.isnan(x)) & (~np.isnan(y))
            return spearmanr(x[good],y[good])[0]#np.corrcoef(x[good],y[good])[0,1]
        
        dic['A_r']=ar2=corr(offdiag(-infermats['A']),offdiag(-mat))
        dic['V_r']=vr2=corr(offdiag(infermats['V']),offdiag(V))
        dic['Vcop_r']=vcr2=corr(offdiag(infermats['V']),offdiag(Vcop))
        # print dic
        
        if np.isnan([ar2,vr2,vcr2]).any():
            code_debugger()

        figs['matrix'] = plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (3, 2))
        plt.subplot(231)
        plt.title('Interaction matrix')
        plt.colorbar(plt.imshow(mat, cmap='seismic_r', vmin=-np.max(np.abs(mat)), vmax=np.max(np.abs(mat))
                                ), ax=plt.gca())
        plt.subplot(232)
        plt.title('Inferred interactions')
        plt.colorbar(plt.imshow(infermats['A'], cmap='seismic_r', vmin=-np.max(np.abs(mat)), vmax=np.max(np.abs(mat))
                                ), ax=plt.gca())
        plt.subplot(233)
        plt.title('V (inverse) matrix')
        plt.colorbar(plt.imshow(V, cmap='seismic_r', vmin=-np.max(np.abs(V)), vmax=np.max(np.abs(V))
                                ), ax=plt.gca())
        plt.subplot(234)
        plt.title('V weighted')
        plt.colorbar(plt.imshow(Vcop, cmap='seismic_r', vmin=-np.max(np.abs(V)), vmax=np.max(np.abs(V))
                                ), ax=plt.gca())


        plt.subplot(235)
        plt.title('V std')
        plt.colorbar(plt.imshow(Vstd, cmap='seismic_r', vmin=-np.max(np.abs(V)), vmax=np.max(np.abs(V))
                                ), ax=plt.gca())

        plt.subplot(236)
        plt.title('Inferred V')
        plt.colorbar(plt.imshow(infermats['V'], cmap='seismic_r', vmin=-np.max(np.abs(V)), vmax=np.max(np.abs(V))
                                ), ax=plt.gca())



        if save:
            for f in figs:
                imgformat='svg'
                if f in [ 'infer_niche']:
                #if 'N_' in f:
                    imgformat='png'
                figs[f].savefig(fpath+'{}.{}'.format(f,imgformat) )
            plt.close('all')
        else:
            plt.show()
    return table

def summary_plots(path,axes=None,save=0,values=None,x_axis='A_std',s_axis='A_mn',**kwargs):
    """Creates summary measurements for each simulation, store them all in summary.csv, along with all simulation parameters."""

    df=None
    if not kwargs.get('rerun',0):
        try:
            df=pd.read_json(path+'summary.csv')
        except:
            pass
    if df is None:
        table=detailed_plots(path,save,**kwargs)
        df=pd.DataFrame(table)
        df.to_json(path+'summary.csv')
    df['displim']=[z[0]<0 for z in df['n_range'].values]
    figs={}
    import seaborn as sns
    plt.style.use(['seaborn'])

    if 0:
        plt.style.use("seaborn-dark")
        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'  # bluish dark grey
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'  # very light grey
        ax=plt.gca()
        ax.grid(color='#2A3459')  # bluish dark grey, but slightly lighter than background

    cmaps=['Reds','Blues']
    markers=['o','X']
    def mks(s):
        s1,s2=np.min(s),np.max(s)
        return ( (s-s1)/(s2-s1) ) *120 +20
    for s in ['mn','std','mnabs','symmetry']:
        figs['stats_'+s]=plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (1.5, 1.3))
        def gets(data,label):
            return data[label+'_'+s].values
        for iz,z in enumerate(['V','V*','Vcop','covresidual','"R_invcorr_vs_mat','Apred']): #corNK
            plt.subplot(2,3,1+iz), plt.title(s+' '+z),plt.xlabel(s+' A')
            # print s,z, np.log10(gets(z))
            for displim,gp in df.groupby('displim'):
                try:
                    xs,ys=(gets(gp,'A')),(gets(gp,z))
                except:
                    continue
                zlab=z
                if (ys<0).all():
                    ys=-ys
                    zlab='|'+z+'|'
                if (xs>0).all() and not 'sym' in s:
                    plt.xlabel('log10 '+s + ' A')
                    xs=np.log10(xs)
                if (ys>0).all() and not 'sym' in s:
                    plt.title('log10 '+ s +' '+zlab)
                    ys=np.log10(ys)
                m=markers[displim]
                cmap=matplotlib.cm.get_cmap(cmaps[displim] )
                slope, intercept, r_value, p_value, std_err = stats.linregress(xs,ys)
                plt.plot(np.sort(xs),intercept+slope*np.sort(xs),label=r'$R^2$:{:.2g}'.format(r_value**2),color=cmap(.5),lw=5,alpha=.5 )
                plt.scatter(xs, ys, marker=m, edgecolor='k',lw=1,cmap=cmap,c=mks(np.log10(gp['A_std'].values)),s=mks(np.log10(gp['A_mn'].values)) )
            plt.legend()
        plt.tight_layout()

    
    figs['stats_r2' ] = plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (1.5, 1.5))
    try:
        df['covN/covK']=df['covN_mnabs']/df['covK_mnabs']
        for iz,z in enumerate(['A','V','Vcop']):
            for ix in (0,1):
                x_ax=[x_axis,'covN/covK'][ix]
                plt.subplot(2, 3, 3*ix + 1 + iz), plt.title(r'$R^2$ fit ' + z), plt.xlabel(x_ax)#'log10 std A')
                for displim, gp in df.groupby('displim'):
                    m=markers[displim]
                    cmap=matplotlib.cm.get_cmap(cmaps[displim] )
                    xs=gp[x_ax].values
                    if 'A_' in x_ax or 'dispersal_' in x_ax:
                        xs= np.log10(xs)
                    if s_axis is None:
                        ss=np.ones(xs.shape)*50
                    else:
                        ss=gp[s_axis].values
                        if 'A_' in s_axis:
                            ss=np.log10(ss)
                        elif np.min(ss)==0:
                            ss=np.log10(1+ss)
                        ss=mks(ss)
                    try:
                        rr=gp[z+'_r']
                    except:
                        rr=gp[z+'_r2']
                    plt.scatter(xs,rr.values**2,
                            marker=m,c=mks(xs), edgecolor='k',lw=1,cmap=cmap,s=ss )
                    plt.ylim(ymin=-0.1,ymax=1.1)
                
        plt.tight_layout()
    except:
        pass

    if save:
        if isinstance(save,str) or isinstance(save,Path):
            outpath=save
        else:
            outpath = path + Path('plots')
        for f in figs:
            figs[f].savefig(outpath+'{}.svg'.format(f) )
    else:
        plt.show()
    return df




