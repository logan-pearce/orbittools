def eccentricity_anomaly(E,e,M):
    '''Eccentric anomaly function'''
    import numpy as np
    return E - (e*np.sin(E)) - M

def solve(f, M0, e, h):
    ''' Newton-Raphson solver for eccentricity anomaly
    from https://stackoverflow.com/questions/20659456/python-implementing-a-numerical-equation-solver-newton-raphson
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
    Returns: nextE (float): converged solution for eccentric anomaly
    '''
    import numpy as np
    if M0 / (1.-e) - np.sqrt( ( (6.*(1-e)) / e ) ) <= 0:
        E0 = M0 / (1.-e)
    else:
        E0 = (6. * M0 / e) ** (1./3.)
    lastE = E0
    nextE = lastE + 10* h  # "different than lastX so loop starts OK
    number=0
    while (abs(lastE - nextE) > h) and number < 1001:  # this is how you terminate the loop - note use of abs()
        newY = f(nextE,e,M0) # just for debug... see what happens
        lastE = nextE
        nextE = lastE - newY / (1.-e*np.cos(lastE))  # update estimate using N-R
        number=number+1
        if number >= 1000:
            nextE = float('NaN')#This truncates the calculation if a solution hasn't been reached by 1000 iter.
    return nextE

def to_si(mas,mas_yr,d):
    '''Convert from mas -> km and mas/yr -> km/s
        Input: 
         mas (array) [mas]: distance in mas
         mas_yr (array) [mas/yr]: velocity in mas/yr
         d (float) [pc]: distance to system in parsecs
        Returns:
         km (array) [km]: distance in km
         km_s (array) [km/s]: velocity in km/s
    '''
    import numpy as np
    import astropy.units as u
    km = ((mas*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = ((mas_yr*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = (km_s.value)*(u.km/u.yr).to(u.km/u.s)
    return km.value,km_s

def draw_priors(number):
    """Draw a set of orbital elements from proability distribution functions.
        Input: number - number of orbits desired to draw elements for
        Returns:
            a [as]: semi-major axis - set at 100 AU inital value
            T [yr]: period
            const: constant defining orbital phase of observation
            to [yr]: epoch of periastron passage
            e: eccentricity
            i [rad]: inclination in radians
            w [rad]: arguement of periastron
            O [rad]: longitude of nodes - set at 0 initial value
            m1 [Msol]: total system mass in solar masses
            dist [pc]: distance to system
    """
    import numpy as np
    #m1 = np.random.normal(m_tot,m_tot_err,number)
    #dist = np.random.normal(d_star,d_star_err,number)
    m1 = m_tot
    dist = d_star
    # Fixing and initial semi-major axis:
    a_au=100.0
    a_au=np.linspace(a_au,a_au,number)
    T = np.sqrt((np.absolute(a_au)**3)/np.absolute(m1))
    a = a_au/dist #semimajor axis in arcsec

    # Fixing an initial Longitude of ascending node in radians:
    O = np.radians(0.0)  
    O=[O]*number

    # Randomly generated parameters:
    #to = Time of periastron passage in years:
    const = np.random.uniform(0.0,1.0,number)
    #^ Constant that represents the ratio between (reference epoch minus to) over period.  Because we are scaling
    #semi-major axis, period will also scale, and epoch of periastron passage will change as a result.  This ratio
    #will remain constant however, so we can use it scale both T and to appropriately.
    to = d-(const*T)

    # Eccentricity:
    e = np.random.uniform(0.0,1.0,number)
    # Inclination in radians:
    cosi = np.random.uniform(-1.0,1.0,number)  #Draws sin(i) from a uniform distribution.  Inclination
    # is computed as the arccos of cos(i):
    i = np.arccos(cosi)
    # Argument of periastron in degrees:
    w = np.random.uniform(0.0,360.0,number)
    w = np.radians(w) #convert to radians for calculations
    return a,T,const,to,e,i,w,O,m1,dist

####### Scale and rotate to Gaia epoch:
def scale_and_rotate(X,Y):
    ''' Generates a new semi-major axis, period, epoch of peri passage, and long of peri for each orbit
        given the X,Y plane of the sky coordinates for the orbit at the date of the reference epoch
    '''
    import numpy as np
    r_model = np.sqrt((X**2)+(Y**2))
    rho_rand = np.random.normal(rho/1000.,rhoerr/1000.) #This generates a gaussian random to 
    #scale to that takes observational uncertainty into account.  #convert to arcsec
    #rho_rand = rho/1000. 
    a2 = a*(rho_rand/r_model)  #<- scaling the semi-major axis
    #New period:
    a2_au=a2*dist #convert to AU for period calc:
    T2 = np.sqrt((np.absolute(a2_au)**3)/np.absolute(m1))
    #New epoch of periastron passage
    to2 = d-(const*T2)

    # Rotate:
    # Rotate PA:
    PA_model = (np.degrees(np.arctan2(X,-Y))+270)%360 #corrects for difference in zero-point
    #between arctan function and ra/dec projection
    PA_rand = np.random.normal(pa,paerr) #Generates a random PA within 1 sigma of observation
    #PA_rand = pa
    #New omega value:
    O2=[]
    for PA_i in PA_model:
        if PA_i < 0:
            O2.append((PA_rand-PA_i) + 360.)
        else:
            O2.append(PA_rand-PA_i)
    # ^ This step corrects for the fact that the arctan gives the angle from the +x axis being zero,
    #while for RA/Dec the zero angle is +y axis.  

    #Recompute model with new rotation:
    O2 = np.array(O2)
    O2 = np.radians(O2)
    return a2,T2,to2,O2

def calc_XYZ(a,T,to,e,i,w,O,date):
    ''' Compute projected on-sky position only of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point. 
        Inputs:
            a [as]: semi-major axis in as
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
        Returns: X, Y, and Z coordinates [as] where +X is in the reference direction (north) and +Y is east, and +Z
            is towards observer
    '''
    import numpy as np
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    n = (2*np.pi)/T
    M = n*(date-to)
    try:
        nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
    except:
        E = solve(eccentricity_anomaly, M,e, 0.001)
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    # orbit plane radius in as:
    r = (a*(1.-e**2))/(1.+(e*cos(f)))
    X = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) )
    Y = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) )
    Z = r * sin(w+f)*sin(i)
    return X,Y,Z

def calc_velocities(a,T,to,e,i,w,O,date,dist):
    ''' Compute 3-d velocity of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point.  Uses my eqns derived from Seager 
        Exoplanets Ch2.
        Inputs:
            a [as]: semi-major axis in mas
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system
        Returns: X dot, Y dot, Z dot three dimensional velocities [km/s]
    '''
    import numpy as np
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    import astropy.units as u
    # convert to km:
    a_km = to_si(a/1000.,0.,dist)
    a_km = a_km[0]
    
    # Compute true anomaly:
    n = (2*np.pi)/T
    M = n*(date-to)
    try:
        nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
    except:
        E = solve(eccentricity_anomaly, M,e, 0.001)
    r1 = a*(1.-e*cos(E))
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    
    # Compute velocities:
    rdot = ( (n*a_km) / (np.sqrt(1-e**2)) ) * e*sin(f)
    rfdot = ( (n*a_km) / (np.sqrt(1-e**2)) ) * (1 + e*cos(f))
    Xdot = rdot * (cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i)) + \
           rfdot * (-cos(O)*sin(w+f) - sin(O)*cos(w+f)*cos(i))
    Ydot = rdot * (sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i)) + \
           rfdot * (-sin(O)*sin(w+f) + cos(O)*cos(w+f)*cos(i))
    Zdot = ((n*a_km) / (np.sqrt(1-e**2))) * sin(i) * (cos(w+f) + e*cos(w))
    
    Xdot = Xdot*(u.km/u.yr).to((u.km/u.s))
    Ydot = Ydot*(u.km/u.yr).to((u.km/u.s))
    Zdot = Zdot*(u.km/u.yr).to((u.km/u.s))
    return Xdot,Ydot,Zdot

def calc_accel(a,T,to,e,i,w,O,date,dist):
    ''' Compute 3-d acceleration of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point.  
        Inputs:
            a [as]: semi-major axis in as
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system in pc
        Returns: X ddot, Y ddot, Z ddot three dimensional accelerations [m/s/yr]
    '''
    import numpy as np
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    import astropy.units as u
    # convert to km:
    a_mas = a*u.arcsec.to(u.mas)
    try:
        a_mas = a_mas.value
    except:
        pass
    a_km = to_si(a_mas,0.,dist)[0]
    # Compute true anomaly:
    n = (2*np.pi)/T
    M = n*(date-to)
    try:
        nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
    except:
        E = solve(eccentricity_anomaly, M,e, 0.001)
    # r and f:
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    r = (a_km*(1-e**2))/(1+e*cos(f))
    # Time derivatives of r, f, and E:
    Edot = n/(1-e*cos(E))
    rdot = e*sin(f)*((n*a_km)/(sqrt(1-e**2)))
    fdot = ((n*(1+e*cos(f)))/(1-e**2))*((sin(f))/sin(E))
    # Second time derivatives:
    Eddot = ((-n*e*sin(f))/(1-e**2))*fdot
    rddot = a_km*e*cos(E)*(Edot**2) + a_km*e*sin(E)*Eddot
    fddot = Eddot*(sin(f)/sin(E)) - (Edot**2)*(e*sin(f)/(1-e*cos(E)))
    # Positional accelerations:
    Xddot = (rddot - r*fdot**2)*(cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i)) + \
            (-2*rdot*fdot - r*fddot)*(cos(O)*sin(w+f) + sin(O)*cos(w+f)*cos(i))
    Yddot = (rddot - r*fdot**2)*(sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i)) + \
            (2*rdot*fdot + r*fddot)*(sin(O)*sin(w+f) + cos(O)*cos(w+f)*cos(i))
    Zddot = sin(i)*((rddot - r*(fdot**2))*sin(w+f) + ((2*rdot*fdot + r*fddot)*cos(w+f)))
    return Xddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr)), Yddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr)), \
                    Zddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr))

def compute_observables(a,T,to,e,i,w,O,d,dist):
    """
    """
    import astropy.units as u
    a = a.to(u.mas)
    pos = calc_XYZ(a,T,to,e,i,w,O,d)
    vel = calc_velocities(a,T,to,e,i,w,O,d,dist)
    acc = calc_accel(a,T,to,e,i,w,O,d,dist)


def calc_OFTI(a,T,to,e,i,w,O,d,dist):
    import numpy as np
    import astropy.units as u
    X1,Y1,Z1 = calc_XYZ(a,T,to,e,i,w,O,d)
    a2,T2,to2,O2 = scale_and_rotate(X1,Y1)
    X2,Y2,Z2 = calc_XYZ(a2,T2,to2,e,i,w,O2,d)
    X2,Y2,Z2 = (X2*u.arcsec).to(u.mas).value, (Y2*u.arcsec).to(u.mas).value, (Z2*u.arcsec).to(u.mas).value
    Xdot,Ydot,Zdot = calc_velocities(a2,T2,to2,e,i,w,O2,d,dist)
    Xddot,Yddot,Zddot = calc_accel(a2,T2,to2,e,i,w,O2,d,dist)
    i,w,O2 = np.degrees(i),np.degrees(w),np.degrees(O2)
    return X2,Y2,Z2,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2


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

