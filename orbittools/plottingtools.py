def seppa_plot(tmin, tmax, params, norbits, 
               plot_obs = False,
               observations = 0, observationdates = 0, 
               savefig = False, 
               filename='seppa_plot.png'):
    """ Make a plot of separation and position angle over time for orbits given orbital parameters, 
        and optionally overplot observational data points.
        Inputs:
            tmin,tmax (flt): date range over which to compute orbit tracks in decimal year
            params (arr,flt): array of orbital elements; assumes the following order and units:
                a [as]: semi-major axis in as
                T [yrs]: period
                to [yrs]: epoch of periastron passage (in same time structure as dates)
                e: eccentricity
                i [rad]: inclination
                w [rad]: argument of periastron
                O [rad]: longitude of nodes
            norbits (int): number of orbits in the params array to plot (allows you to select the 
                    first N orbits to plot)
            plot_obs (bool): set to True to overplot observations over orbit tracks
            observations (4xN arr): array of observations with:
                observations[0,N] = separations
                observations[1,N] = separation error
                observations[2,N] = position angle
                observations[3,N] = pa error
                Where N = number of observations
            observationdates (1xN array) = dates in decimal years
            savefig (bool): Set to True to save the figure
            filename (str): filename for figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
    import matplotlib.ticker

    if plot_obs == True:
        if observations == 0:
            print('Oops! Please include array of observations and observation dates')
            return
        
    # Compute seppa of orbits over date range:
    times = np.linspace(tmin,tmax,2000)

    X = np.zeros((norbits,len(times)))
    Y = np.zeros((norbits,len(times)))
    r = np.zeros((norbits,len(times)))
    for j in range(len(times)):
        pos = calc_XYZ(*params[:,:norbits],times[j])
        X[:,j] = pos[0]
        Y[:,j] = pos[1]

    r = np.sqrt((X**2)+(Y**2))
    theta=np.arctan2(X,-Y)
    theta=(np.degrees(theta)+270.)%360
    
    # Establish plot parameters:
    plt.style.use('supermongo')
    date_ticks = np.arange(tmin,tmax,10)
    ticksize = 15
    labelsize = 25
    alpha = 0.9
    fig = plt.figure(figsize=(8, 10))
    
    # Sep subplot:
    plt.subplot(2, 1, 1)
    plt.xlim(tmin,tmax)
    plt.ylabel(r'$\rho$ (mas)',fontsize=labelsize)
    majorLocator   = MultipleLocator(10)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(5)
    ax=plt.gca()
    ax.tick_params(labelsize=ticksize)
    ax.xaxis.set_major_locator(MaxNLocator(6,prune="both"))
    plt.grid(linestyle=':')
    
    #Plot orbits sep:
    for j in range(norbits):
        plt.plot(times,r[j]*1000,color='skyblue',alpha=0.5)
    
    # Plot observations sep:
    if plot_obs == True:
        plt.scatter(observationdates,observations[0], color='black',marker='o',zorder=10,s=80)
        plt.errorbar(observationdates,observations[0], yerr=observations[1], ls='none',
                color='black',elinewidth=2,capthick=2,zorder=10)
    # PA subplot:
    plt.subplot(2, 1, 2)
    plt.xlim(tmin,tmax)
    plt.ylabel(r'P.A. (deg)',fontsize=labelsize)
    plt.xlabel('Years',fontsize=labelsize)
    majorLocator   = MultipleLocator(10)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(5)
    ax=plt.gca()
    ax.tick_params(labelsize=ticksize)
    ax.xaxis.set_major_locator(MaxNLocator(6,prune="both"))

    for j in range(norbits):
        plt.plot(times,theta[j],color='skyblue',alpha=0.5)
    if plot_obs == True:
        plt.scatter(observationdates,observations[2], color='black',marker='o',zorder=10,s=80)
        plt.errorbar(observationdates,observations[2], yerr=observations[3], ls='none',
                    color='black',elinewidth=2,capthick=2,zorder=10)

    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.show()
    if savefig == True:
        plt.savefig(filename)

