from landscapemodel import *
from plots import *


def gettingmore(path,rerun='rerun' in sys.argv):
    """ Main results """

    prm = deepcopy(LandscapeModel.dft_prm)
    prm['competition']['multiscale'] = 0
    prm['competition']['distribution'] = 'normal'
    prm['competition']['relstd'] = .3
    del prm['competition']['range']
    prm['mortality']['range'] = (0.002, .5)
    prm['envniche']['width']['range'] = (50., 60.)
    prm['envniche']['pos']['range'] = (0., 100.)
    prm['environment']['range'] = (-50., 150.)
    prm['environment']['cutoff']=.1
    prm['dispersal']['mean']=0.
    axes = [
        # ('species',[30,]),
        # ('envniche_pos_range', [(0., 100.), (40., 50.), ]),  # Environment heterogeneity
        # ('dispersal_mean', [0.0, 0.3]),  # Mean dispersal strength
        ('competition_symmetry', np.linspace(-.9,.9,5) ),  # 0.2, Interspecific competition strength
        ('competition_mean', [0.01,0.05, 0.3] ),  # 0.2, Interspecific competition strength
        ('competition_relstd', [0.001,0.01, .1, 1.]),  # Coefficient of variation
        ('n_range',[(.1,1),(-1,1)]),
        ('sys', range(1))
        # Dummy variable (change number in parentheses to make multiple runs with same parameters)
    ]
    loop(axes=axes, path=path, tmax=2000, nsample=20, rerun=rerun, species=20,
         reseed=0, use_Fourier=0, method='scipy', dft_prm=prm)

    df = summary_plots(path, save=1,  movie='movie' in sys.argv,
                       rerun='rerun' in sys.argv or 'replot' in sys.argv,
                       values=[
                               ]

                       , axes=[a[0] for a in axes[:-1]]
                       )
    plt.show()




def dispersal(path,rerun='rerun' in sys.argv):
    """ Continuously varying dispersal """
    
    prm = deepcopy(LandscapeModel.dft_prm)
    prm['competition']['multiscale'] = 0
    prm['competition']['distribution'] = 'normal'
    del prm['competition']['range']
    prm['competition']['mean']=0.3
    prm['competition']['relstd']=0.1
    prm['mortality']['range'] = (0.002, .5)
    prm['envniche']['width']['range'] = (100., 120.)
    prm['envniche']['pos']['range'] = (0., 100.)
    prm['environment']['range'] = (-50., 150.)
    prm['environment']['cutoff']=.1
    prm['dispersal']['mean']=0.
    prm['n']['range']=(-1,1)
    
    axes = [
        ('competition_mean', prm['competition']['mean']*np.linspace(1,1.1,10)),# Dummy variable (change number in parentheses to make multiple runs with same parameters)
         ('dispersal_mean', np.logspace(-4,-3,10)),  # Mean dispersal strength

    ]
    loop(axes=axes, path=path, tmax=2000, nsample=20, rerun=rerun, species=20,
         reseed=0, use_Fourier=0, methodleibold='scipy', dft_prm=prm)

    df = summary_plots(path, save=1, movie='movie' in sys.argv,x_axis='dispersal_mean',s_axis=None,
                       rerun='rerun' in sys.argv or 'replot' in sys.argv,
                       values=[
                       'infer_niche_error',
                       'infer_niche_height_error',#_interceptmethod',
                       'infer_niche_center_error',
                       'infer_niche_width_error',
                               ]
                       , axes=[a[0] for a in axes[:-1]]
                       )

    try:
        df_jsdm=pd.read_csv(Path(path)+'jsdm.csv')
        df_jsdm['abio/bio']=df_jsdm['abio']/df_jsdm['bio']
        plt.figure(),plt.suptitle('JSDM figure')
        idx=0
        ax=axes[0][0]
        for key in ['niche_corr','naive_niche_corr','V_corr','abio/bio']:
            plt.subplot(2,2,1+idx)
            idx+=1
            plt.xscale('log')
            plt.scatter(df_jsdm[ax].values,df_jsdm[key].values,)
            if 'corr' in key:
                plt.ylim(ymin=min(0,df_jsdm[ax].min()),ymax=1)
                plt.axhline(0,color='k')
            plt.xlabel("Dispersal"),plt.title(key)
    except Exception as e:
        print(e)
        pass
    plt.figure()
    plt.xscale('log')
    plt.scatter(df['dispersal_mean'].values,df['A_r'].values**2)
    plt.xlabel('Dispersal'),plt.ylabel(r'$R^2$ fit interactions')
    plt.figure()
    plt.xscale('log')
    plt.scatter(df['dispersal_mean'].values,df['#compositions'].values)
    plt.xlabel('Dispersal'),plt.ylabel(r'# compositions')
    
    plt.figure()
    idx=1    
    for z in [
                       'infer_niche_error',
                       'infer_niche_height_error',#_interceptmethod',
                       'infer_niche_center_error',
                       'infer_niche_width_error',
                               ]:
        plt.subplot(2,2,idx)
        idx+=1
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(df['dispersal_mean'].values,df[z].values)
        plt.xlabel('Dispersal'),plt.ylabel(z)
    
    plt.show()






path=Path('main_results')
gettingmore(path)

path=Path('dispersal_results')
dispersal(path)


