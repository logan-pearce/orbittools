import numpy as np
import astropy.units as u

def period(sma,mass):
    """ Given semi-major axis in AU and mass in solar masses, return the period in years of an orbit using 
        Kepler's third law.
        Written by Logan Pearce, 2019
    """
    import numpy as np
    import astropy.units as u
    # If astropy units are given, return astropy unit object
    try:
        sma = sma.to(u.au)
        mass = mass.to(u.Msun)
        period = np.sqrt(((sma)**3)/mass).value*(u.yr)
    # else return just a value.
    except:
        period = np.sqrt(((sma)**3)/mass)
    return period

def semimajoraxis(period,mass):
    """ Given period in years and mass in solar masses, return the semi-major axis in au of an orbit using 
        Kepler's third law.
        Written by Logan Pearce, 2019
    """
    import numpy as np
    import astropy.units as u
    # If astropy units are given, return astropy unit object
    try:
        period = period.to(u.yr)
        mass = mass.to(u.Msun)
        sma = ((mass * period**2) ** (1/3)).value*u.au
    # else return just a value.
    except:
        sma = (mass * period**2) ** (1/3)
    return sma

def distance(parallax,parallax_error):
    '''Computes distance from Gaia parallaxes using the Bayesian method of Bailer-Jones 2015.
    Input: parallax [mas], parallax error [mas]
    Output: distance [pc], 1-sigma uncertainty in distance [pc]
    '''
    import numpy as np
    # Compute most probable distance:
    L=1350 #parsecs
    # Convert to arcsec:
    parallax, parallax_error = parallax/1000., parallax_error/1000.
    # establish the coefficients of the mode-finding polynomial:
    coeff = np.array([(1./L),(-2),((parallax)/((parallax_error)**2)),-(1./((parallax_error)**2))])
    # use numpy to find the roots:
    g = np.roots(coeff)
    # Find the number of real roots:
    reals = np.isreal(g)
    realsum = np.sum(reals)
    # If there is one real root, that root is the  mode:
    if realsum == 1:
        gd = np.real(g[np.where(reals)[0]])
    # If all roots are real:
    elif realsum == 3:
        if parallax >= 0:
            # Take the smallest root:
            gd = np.min(g)
        elif parallax < 0:
            # Take the positive root (there should be only one):
            gd = g[np.where(g>0)[0]]
    
    # Compute error on distance from FWHM of probability distribution:
    from scipy.optimize import brentq
    rmax = 1e6
    rmode = gd[0]
    M = (rmode**2*np.exp(-rmode/L)/parallax_error)*np.exp((-1./(2*(parallax_error)**2))*(parallax-(1./rmode))**2)
    lo = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), 0.001, rmode)
    hi = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), rmode, rmax)
    fwhm = hi-lo
    # Compute 1-sigma from FWHM:
    sigma = fwhm/2.355
            
    return gd[0],sigma

def to_polar(RAa,RAb,Deca,Decb):
    ''' Converts RA/Dec [deg] of two binary components into separation and position angle of B relative 
        to A [mas, deg]
    '''
    import numpy as np
    import astropy.units as u
    dRA = (RAb - RAa) * np.cos(np.radians(np.mean([Deca,Decb])))
    dRA = (dRA*u.deg).to(u.mas)
    dDec = (Decb - Deca)
    dDec = (dDec*u.deg).to(u.mas)
    r = np.sqrt( (dRA ** 2) + (dDec ** 2) )
    p = (np.degrees( np.arctan2(dDec.value,-dRA.value) ) + 270.) % 360.
    p = p*u.deg
    return r, p

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

def parallax(d):
    """
    Returns parallax in arcsec given distances.
    Args:
        d (float): distance
    Return:
        parallax in arcsecs
    Written by: Logan Pearce, 2017
    """
    from astropy import units as u
    d = d.to(u.pc)
    x = (1*u.au)/(d)
    return x.to(u.arcsec, equivalencies=u.dimensionless_angles())

def physical_separation(d,theta):
    """
    Returns separation between two objects in the plane of the sky in AU given distance and parallax
    Distance and parallax must be astropy unit objects.
    Args:
        d (float): distance
        theta (float): parallax
    Return:
        separation in AU
    Written by: Logan Pearce, 2017
    """
    from astropy import units as u
    d = d.to(u.pc)
    theta = theta.to(u.arcsec)
    a = (d)*(theta)
    return a.to(u.au, equivalencies=u.dimensionless_angles())

def angular_separation(d,a):
    """
    Returns separation between two objects in the plane of the sky in angle given distance and 
    physical separation in AU.
    Distance and separation must be astropy unit objects.
    Args:
        d (float): distance
        a (float): separation
    Return:
        theta in arcsec
    Written by: Logan Pearce, 2017
    """
    from astropy import units as u
    d = d.to(u.pc)
    a = a.to(u.au)
    theta = a / d
    return theta.to(u.arcsec, equivalencies=u.dimensionless_angles())

def tisserand(a,e,i):
    """ Return Tisserand parameter for given orbital parameters relative to
           Jupiter
        Inputs:
            a : flt
                semi-major axis in units of Jupiter sma
            e : flt
                eccentricity
            i : flt
                inclination in radians relative to J-S orbit plane
        Returns:
            T : flt
                Tisserand parameter define wrt Jupiter's orbit
    """
    return 1/(a) + 2*np.sqrt(a*(1-e**2))*np.cos(i)

def keplersconstant(m1,m2):
    '''Compute Kepler's constant for two gravitationally bound masses k = G*m1*m2/(m1+m2) = G + (m1+m2)
        Inputs:
            m1,m2 (arr,flt): masses of the two objects in solar masses.  Must be astropy objects
        Returns:
            Kepler's constant in m^3 s^(-2)
    '''
    import astropy.constants as c
    import astropy.units as u
    m1 = m1.to(u.Msun)
    m2 = m2.to(u.Msun)
    mu = c.G*m1*m2
    m = (1/m1 + 1/m2)**(-1)
    kep = mu/m
    return kep.to((u.m**3)/(u.s**2))

def M(T,T0,obsdate = 2015.5):
    """ Given period, ref date, and observation date,
            return the mean anomaly

       Parameters:
       -----------
       T : flt
           period
       T0 : flt
           time of periastron passage
       obsdate : flt
           observation date.  Default = 2015.5 (Gai DR2 ref date)
       Returns:
       -----------
       mean anomaly [radians]
    """
    return (2*np.pi/T)*(obsdate-T0)


def draw_orbits(number):
    ''' Semi-major axis is fixed at 100 au and long. of asc. node is fixed at 0 deg.
    Written by Logan Pearce, 2019
    '''
    import astropy.units as u
    import numpy as np
    sma = 100.*u.au
    sma = np.array(np.linspace(sma,sma,number))
    # Eccentricity:
    ecc = np.random.uniform(0.0,1.0,number)
    # Inclination in radians:
    cosi = np.random.uniform(-1.0,1.0,number)  #Draws sin(i) from a uniform distribution.  Inclination
    # is computed as the arccos of cos(i):
    inc = np.degrees(np.arccos(cosi))
    # Argument of periastron in degrees:
    argp = np.random.uniform(0.0,360.0,number)
    # Long of nodes:
    lon = np.degrees(0.0)
    lon = np.array([lon]*number)
    # orbit fraction (fraction of orbit completed at observation date since reference date)
    orbit_fraction = np.random.uniform(0.0,1.0,number)
    return sma, ecc, inc, argp, lon, orbit_fraction

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
        Written by Logan Pearce, 2019
    '''
    import numpy as np
    from orbittools.orbittools import eccentricity_anomaly
    if M0 / (1.-e) - np.sqrt( ( (6.*(1-e)) / e ) ) <= 0:
        E0 = M0 / (1.-e)
    else:
        E0 = (6. * M0 / e) ** (1./3.)
    lastE = E0
    nextE = lastE + 10* h 
    number=0
    while (abs(lastE - nextE) > h) and number < 1001: 
        new = f(nextE,e,M0) 
        lastE = nextE
        nextE = lastE - new / (1.-e*np.cos(lastE)) 
        number=number+1
        if number >= 1000:
            nextE = float('NaN')
    return nextE

def danby_solve(M0, e, h, maxnum=50):
    ''' Newton-Raphson solver for eccentricity anomaly based on "Danby" method in 
        Wisdom textbook
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
        maxnum (int): if it takes more than maxnum iterations,
            use the Mikkola solver instead.
    Returns: nextE (float): converged solution for eccentric anomaly
        Written by Logan Pearce, 2020
    '''
    import numpy as np
    from orbittools.orbittools import eccentricity_anomaly
    f = eccentricity_anomaly
    k = 0.85
    E0 = M0 + np.sign(np.sin(M0))*k*e
    lastE = E0
    nextE = lastE + 10* h 
    number=0
    delta_D = 1
    while (delta_D > h) and number < maxnum+1: 
        fx = f(nextE,e,M0) 
        fp = (1.-e*np.cos(lastE)) 
        fpp = e*np.sin(lastE)
        fppp = e*np.cos(lastE)
        lastE = nextE
        delta_N = -fx / fp
        delta_H = -fx / (fp + 0.5*fpp*delta_N)
        delta_D = -fx / (fp + 0.5*fpp*delta_H + (1./6)*fppp*delta_H**2)
        nextE = lastE + delta_D
        number=number+1
        if number >= maxnum:
            from orbittools.orbittools import mikkola_solve
            nextE = mikkola_solve(M0,e)
    return nextE


def mikkola_solve(M,e):
    ''' Analytic solver for eccentricity anomaly from Mikkola 1987. Most efficient
        when M near 0/2pi and e >= 0.95.
    Inputs: 
        M (float): mean anomaly
        e (float): eccentricity
    Returns: eccentric anomaly
        Written by Logan Pearce, 2020
    '''
    # Constants:
    alpha = (1 - e) / ((4.*e) + 0.5)
    beta = (0.5*M) / ((4.*e) + 0.5)
    ab = np.sqrt(beta**2. + alpha**3.)
    z = np.abs(beta + ab)**(1./3.)

    # Compute s:
    s1 = z - alpha/z
    # Compute correction on s:
    ds = -0.078 * (s1**5) / (1 + e)
    s = s1 + ds

    # Compute E:
    E0 = M + e * ( 3.*s - 4.*(s**3.) )

    # Compute final correction to E:
    sinE = np.sin(E0)
    cosE = np.cos(E0)

    f = E0 - e*sinE - M
    fp = 1. - e*cosE
    fpp = e*sinE
    fppp = e*cosE
    fpppp = -fpp

    dx1 = -f / fp
    dx2 = -f / (fp + 0.5*fpp*dx1)
    dx3 = -f / ( fp + 0.5*fpp*dx2 + (1./6.)*fppp*(dx2**2) )
    dx4 = -f / ( fp + 0.5*fpp*dx3 + (1./6.)*fppp*(dx3**2) + (1./24.)*(fpppp)*(dx3**3) )

    return E0 + dx4

###################################################################
# a set of functions for working with position, velocity, acceleration,
# and orbital elements for Keplerian orbits.

def calc_XYZ(a,T,to,e,i,w,O,date, solvefunc = solve):
    ''' Compute projected on-sky position only of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point. 
        Inputs:
            a [as]: semi-major axis
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
    from lofti_gaiaDR2.loftifittingtools import solve
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    
    n = (2*np.pi)/T
    M = n*(date-to)
    try:
        nextE = [solvefunc(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
    except:
        E = solvefunc(eccentricity_anomaly, M,e, 0.001)
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    # orbit plane radius in as:
    r = (a*(1.-e**2))/(1.+(e*cos(f)))
    X = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) )
    Y = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) )
    Z = r * sin(w+f)*sin(i)
    return X,Y,Z

def calc_velocities(a,T,to,e,i,w,O,date,dist, solvefunc = solve):
    ''' Compute 3-d velocity of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point.  Uses my eqns derived from Seager 
        Exoplanets Ch2.
        Inputs:
            a [as]: semi-major axis
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            m_tot [Msol]: total system mass
        Returns: X dot, Y dot, Z dot three dimensional velocities [km/s]
    '''
    import numpy as np
    import astropy.units as u
    from lofti_gaiaDR2.loftifittingtools import to_si, solve
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    
    # convert to km:
    a_km = to_si(a*1000.,0.,dist)
    a_km = a_km[0]
    
    # Compute true anomaly:
    n = (2*np.pi)/T
    M = n*(date-to)
    try:
        nextE = [solvefunc(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
    except:
        E = solvefunc(eccentricity_anomaly, M,e, 0.001)
    #E = solve(eccentricity_anomaly, M,e, 0.001)
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

def calc_accel(a,T,to,e,i,w,O,date,dist, solvefunc = solve):
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
    from lofti_gaiaDR2.loftifittingtools import to_si, solve
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
        nextE = [solvefunc(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
    except:
        E = solvefunc(eccentricity_anomaly, M,e, 0.001)
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

###################################################################
# OFTI specific functions:

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

def calc_OFTI(a,T,const,to,e,i,w,O,d,m1,dist,rho,pa, solvefunc = solve):
    '''Perform OFTI steps to determine position/velocity/acceleration predictions given
       orbital elements.
        Inputs:
            a [as]: semi-major axis
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system in pc
            rho [mas] (tuple, flt): separation and error
            pa [deg] (tuple, flt): position angle and error
        Returns: 
            X, Y, Z positions in plane of the sky [mas],
            X dot, Y dot, Z dot three dimensional velocities [km/s]
            X ddot, Y ddot, Z ddot 3d accelerations in [m/s/yr]
    '''
    import numpy as np
    import astropy.units as u
    
    # Calculate predicted positions at observation date:
    X1,Y1,Z1 = calc_XYZ(a,T,to,e,i,w,O,d)
    # scale and rotate:
    a2,T2,to2,O2 = scale_and_rotate(X1,Y1,rho,pa,a,const,m1,dist,d, solvefunc = solvefunc)
    # recompute predicted position:
    X2,Y2,Z2 = calc_XYZ(a2,T2,to2,e,i,w,O2,d)
    # convert units:
    X2,Y2,Z2 = (X2*u.arcsec).to(u.mas).value, (Y2*u.arcsec).to(u.mas).value, (Z2*u.arcsec).to(u.mas).value
    # Compute velocities at observation date:
    Xdot,Ydot,Zdot = calc_velocities(a2,T2,to2,e,i,w,O2,d,dist, solvefunc = solvefunc)
    # Compute accelerations at observation date:
    Xddot,Yddot,Zddot = calc_accel(a2,T2,to2,e,i,w,O2,d,dist, solvefunc = solvefunc)
    # Convert to degrees:
    i,w,O2 = np.degrees(i),np.degrees(w),np.degrees(O2)
    return X2,Y2,Z2,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2
    

###################################################################
# a different set of functions for working with position, velocity, acceleration,
# and orbital elements for Keplerian orbits.

def rotate_z(vector,theta):
    """ Rotate a 3D vector about the +z axis
        Inputs:
            vector: 3d vector array
            theta [rad]: angle to rotate the vector about
        Returns: rotated vector
    """
    import numpy as np
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]
              ])
    if np.ndim(vector) == 1:
        out = np.zeros(3)
        out[0] = R[0,0]*vector[0] + R[0,1]*vector[1] + R[0,2]*vector[2]
        out[1] = R[1,0]*vector[0] + R[1,1]*vector[1] + R[1,2]*vector[2]
        out[2] = R[2,0]*vector[0] + R[2,1]*vector[1] + R[2,2]*vector[2]
        
    else:
        out = np.zeros((3,vector.shape[1]))
        out[0] = R[0,0]*vector[0,:] + R[0,1]*vector[1,:] + R[0,2]*vector[2,:]
        out[1] = R[1,0]*vector[0,:] + R[1,1]*vector[1,:] + R[1,2]*vector[2,:]
        out[2] = R[2,0]*vector[0,:] + R[2,1]*vector[1,:] + R[2,2]*vector[2,:]
    
    return out

def rotate_x(vector,theta):
    """ Rotate a 3D vector about the +z axis
        Inputs:
            vector: 3d vector array
            theta [rad]: angle to rotate the vector about
        Returns: rotated vector
    """
    import numpy as np
    if np.ndim(vector) == 1:
        R = np.array([[1., 0., 0.],
              [0., np.cos(theta), -np.sin(theta)],
              [0., np.sin(theta), np.cos(theta)]  
              ])
        out = np.zeros(3)
        out[0] = R[0,0]*vector[0] + R[0,1]*vector[1] + R[0,2]*vector[2]
        out[1] = R[1,0]*vector[0] + R[1,1]*vector[1] + R[1,2]*vector[2]
        out[2] = R[2,0]*vector[0] + R[2,1]*vector[1] + R[2,2]*vector[2]
        
    else:
        R = np.array([[[1.]*len(theta), 0., 0.],
              [0., np.cos(theta), -np.sin(theta)],
              [0., np.sin(theta), np.cos(theta)]  
              ])
        out = np.zeros((3,vector.shape[1]))
        out[0] = R[0,0]*vector[0,:] + R[0,1]*vector[1,:] + R[0,2]*vector[2,:]
        out[1] = R[1,0]*vector[0,:] + R[1,1]*vector[1,:] + R[1,2]*vector[2,:]
        out[2] = R[2,0]*vector[0,:] + R[2,1]*vector[1,:] + R[2,2]*vector[2,:]
    return out

def keplerian_to_cartesian(sma,ecc,inc,argp,lon,meananom,kep, solvefunc = solve):
    """ Given a set of Keplerian orbital elements, returns the observable 3-dimensional position, velocity, 
        and acceleration at the specified time.  Accepts and arbitrary number of input orbits.  Semi-major 
        axis must be an astropy unit object in physical distance (ex: au, but not arcsec).  The observation
        time must be converted into mean anomaly before passing into function.
        Inputs:
            sma (1xN arr flt) [au]: semi-major axis in au, must be an astropy units object
            ecc (1xN arr flt) [unitless]: eccentricity
            inc (1xN arr flt) [deg]: inclination
            argp (1xN arr flt) [deg]: argument of periastron
            lon (1xN arr flt) [deg]: longitude of ascending node
            meananom (1xN arr flt) [radians]: mean anomaly 
            kep (1xN arr flt): kepler constant = mu/m where mu = G*m1*m2 and m = [1/m1 + 1/m2]^-1 . 
                        In the limit of m1>>m2, mu = G*m1 and m = m2
        Returns:
            pos (3xN arr) [au]: position in xyz coords in au, with 
                        x = pos[0], y = pos[1], z = pos[2] for each of N orbits
                        +x = +Dec, +y = +RA, +z = towards observer
            vel (3xN arr) [km/s]: velocity in xyz plane.
            acc (3xN arr) [km/s/yr]: acceleration in xyz plane.
        Written by Logan Pearce, 2019, inspired by Sarah Blunt
    """
    import numpy as np
    import astropy.units as u
    
    # Compute mean motion and eccentric anomaly:
    meanmotion = np.sqrt(kep / sma**3).to(1/u.s)
    try:
        E = solve(eccentricity_anomaly, meananom, ecc, 0.001)
    except:
        nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(meananom, ecc)]
        E = np.array(nextE)

    # Compute position:
    try:
        pos = np.zeros((3,len(sma)))
    # In the plane of the orbit:
        pos[0,:], pos[1,:] = (sma*(np.cos(E) - ecc)).value, (sma*np.sqrt(1-ecc**2)*np.sin(E)).value
    except:
        pos = np.zeros(3)
        pos[0], pos[1] = (sma*(np.cos(E) - ecc)).value, (sma*np.sqrt(1-ecc**2)*np.sin(E)).value
        
    # Rotate to plane of the sky:
    pos = rotate_z(pos, np.radians(argp))
    pos = rotate_x(pos, np.radians(inc))
    pos = rotate_z(pos, np.radians(lon))
    
    # compute velocity:
    try:
        vel = np.zeros((3,len(sma)))
        vel[0], vel[1] = (( -meanmotion * sma * np.sin(E) ) / ( 1- ecc * np.cos(E) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(E) ) / ( 1 - ecc * np.cos(E) )).to(u.km/u.s).value
    except:
        vel = np.zeros(3)
        vel[0], vel[1] = (( -meanmotion * sma * np.sin(E) ) / ( 1- ecc * np.cos(E) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(E) ) / ( 1 - ecc * np.cos(E) )).to(u.km/u.s).value
    vel = rotate_z(vel, np.radians(argp))
    vel = rotate_x(vel, np.radians(inc))
    vel = rotate_z(vel, np.radians(lon))
    
    # Compute accelerations numerically
    # Generate a nearby future time point(s) along the orbit:
    deltat = 0.002*u.yr
    try:
        acc = np.zeros((3,len(sma)))
        futurevel = np.zeros((3,len(sma)))
    except:
        acc = np.zeros(3)
        futurevel = np.zeros(3)
    # Compute new mean anomaly at future time:
    futuremeananom = meananom + meanmotion*((deltat).to(u.s))
    # Compute new eccentricity anomaly at future time:
    try:
        futureE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(futuremeananom.value, ecc)]
        futureE = np.array(futureE)
    except:
        futureE = solve(eccentricity_anomaly, futuremeananom.value, ecc, 0.001)
    # Compute new velocity at future time:
    futurevel[0], futurevel[1] = (( -meanmotion * sma * np.sin(futureE) ) / ( 1- ecc * np.cos(futureE) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(futureE) ) / ( 1 - ecc * np.cos(futureE) )).to(u.km/u.s).value
    futurevel = rotate_z(futurevel, np.radians(argp))
    futurevel = rotate_x(futurevel, np.radians(inc))
    futurevel = rotate_z(futurevel, np.radians(lon))
    acc = (futurevel-vel)/deltat.value
    
    return np.transpose(pos)*u.au, np.transpose(vel)*(u.km/u.s), np.transpose(acc)*(u.km/u.s/u.yr)

def cartesian_to_keplerian(pos, vel, kep):
    """Given observables XYZ position and velocity, compute orbital elements.  Position must be in
       au and velocity in km/s.  Returns astropy unit objects for all orbital elements.
        Inputs:
            pos (3xN arr) [au]: position in xyz coords in au, with 
                        x = pos[0], y = pos[1], z = pos[2] for each of N orbits
                        +x = +Dec, +y = +RA, +z = towards observer
                        Must be astropy unit array e.g: [1,2,3]*u.AU, ~NOT~ [1*u.AU,2*u.AU,3*u,AU]
            vel (3xN arr) [km/s]: velocity in xyz plane.  Also astropy unit array
            kep (flt) [m^3/s^2] : kepler's constant.  From output of orbittools.keplersconstant. Must be
                        astropy unit object.
        Returns:
            sma (1xN arr flt) [au]: semi-major axis in au, must be an astropy units object
            ecc (1xN arr flt) [unitless]: eccentricity
            inc (1xN arr flt) [deg]: inclination
            argp (1xN arr flt) [deg]: argument of periastron
            lon (1xN arr flt) [deg]: longitude of ascending node
            meananom (1xN arr flt) [radians]: mean anomaly 
        Written by Logan Pearce, 2019, inspired by Sarah Blunt
    """
    import numpy as np
    import astropy.units as u
    # rvector x vvector:
    rcrossv = np.cross(pos, vel)*u.au*(u.km/u.s)
    # specific angular momentum:
    h = np.sqrt(rcrossv[0]**2 + rcrossv[1]**2 + rcrossv[2]**2)
    # normal vector:
    n = rcrossv / h
    
    # inclination:
    inc = np.arccos(n[2])
    
    # longitude of ascending node:
    lon = np.arctan2(n[0],-n[1])
    
    # semi-major axis:
    r = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    v = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
    sma = 1/((2./r) - ((v)**2/kep))
    
    # ecc and f:
    rdotv = pos[0]*vel[0] + pos[1]*vel[1] + pos[2]*vel[2]
    rdot = rdotv/r
    parameter = h**2 / kep
    ecosf = parameter/r - 1
    esinf = (h)*rdot / (kep.to(u.m**3/u.s**2))
    ecc = np.sqrt(ecosf**2 + esinf**2)
    f = np.arctan2(esinf,ecosf)
    f = f.value%(2.*np.pi)
    
    # E and M:
    E = 2. * np.arctan( np.sqrt( (1 - ecc.value)/ (1 + ecc.value) ) * ( np.tan(f/2.) ) )
    M = E - ecc * np.sin(E)
    
    # argument of periastron:
    rcosu = pos[0] * np.cos(lon) + pos[1] * np.sin(lon)
    rsinu = (-pos[0] * np.sin(lon) + pos[1] * np.cos(lon)) / np.cos(inc)
    uangle = np.arctan2(rsinu,rcosu)
    argp = uangle.value - f
    
    return sma.to(u.au), ecc, np.degrees(inc), (np.degrees(argp)%360.)*u.deg, (np.degrees(lon.value)%360.)*u.deg, M


def kepler_advancer(ro, vo, t, k, to = 0):
    ''' Initial value problem solver.  Given an initial position and
        velocity vector (in 2 or 3 dimensions; in any reference frame
        [plane of the sky, plane of the orbit]) at an initial time to,
        compute the position and velocity vector at a later time t in 
        that same frame.

        Written by Logan A. Pearce, 2020
        
        Parameters:
       -----------
       ro : flt, arr
           initial position vector at time = to; astropy unit object
       vo : flt, arr
           initial velocity vector at time = to; astropy unit object
       t : flt
           future time at which to compute new r,v vectors; 
           astropy unit object
       k : flt
           "Kepler's constant", k = G*(m1+m2); astropy unit object
       to : flt
           initial time for initial values.  Default = 0; 
           astropy unit object
       
       Returns:
       --------
       new_r : flt, arr
           new position vector at time t in m
       new_v : flt, arr
           new velocity vector at time t in m/s
    '''
    from orbittools.orbittools import danby_solve
    import numpy as np
    # Convert everything to mks:
    ro = ro.to(u.m).value
    vo = vo.to(u.m/u.s).value
    k = k.to((u.m**3)/(u.s**2)).value
    t = t.to(u.s).value
    if to != 0:
        to = to.to(u.s).value
    # Compute magnitude of position vector:
    r = np.linalg.norm(ro)
    # Compute v^2:
    v2 = np.linalg.norm(vo)**2
    # Compute ang mom h^2:
    h2 = np.linalg.norm(np.cross(ro,vo))**2
    # find a [m] from vis-viva:
    a = (2/r - v2/k)**(-1)
    # mean motion:
    n = np.sqrt(k/(a**3))
    # ecc:
    e = np.sqrt( 1 - h2/(k*a) )
    # Eo:
    E0 = np.arccos(1/e*(1-r/a))
    # M0:
    M0 = E0 - e*np.sin(E0)
    # M(t = t):
    M = M0 + n*(t-to)
    # E:
    E = danby_solve(M, e, 1.e-9)
    # delta E
    deltaE = E-E0
    cosDE = np.cos(deltaE)
    sinDE = np.sin(deltaE)
    # f, g:
    f = 1 + (a/r)*(cosDE - 1)
    g = t - to + (1/n)*(sinDE - deltaE)
    # new r:
    new_r = f*ro + g*vo
    # fdot, gdot:
    fprime = 1 - (cosDE * e * np.cos(E)) + (sinDE * e * np.sin(E))
    fdot = -n*a*sinDE / (r*fprime)
    gdot = 1 + (cosDE - 1) / fprime
    # new v:
    new_v = fdot*ro + gdot*vo
    
    return new_r*u.m, new_v*(u.m/u.s)


def kepler_advancer2(ro, vo, t, k, to = 0):
    ''' Initial value problem solver using Wisdom-Holman
        numerically well-defined expressions

        Written by Logan A. Pearce, 2020
        
        Parameters:
       -----------
       ro : flt, arr
           initial position vector at time = to; astropy unit object
       vo : flt, arr
           initial velocity vector at time = to; astropy unit object
       t : flt
           future time at which to compute new r,v vectors; 
           astropy unit object
       k : flt
           "Kepler's constant", k = G*(m1+m2); astropy unit object
       to : flt
           initial time for initial values.  Default = 0; 
           astropy unit object
       
       Returns:
       --------
       new_r : flt, arr
           new position vector at time t in m
       new_v : flt, arr
           new velocity vector at time t in m/s
    '''
    from orbittools.orbittools import danby_solve
    import numpy as np
    # Convert everything to mks:
    ro = ro.to(u.m).value
    vo = vo.to(u.m/u.s).value
    k = k.to((u.m**3)/(u.s**2)).value
    t = t.to(u.s).value
    if to != 0:
        to = to.to(u.s).value
    # Compute magnitude of position vector:
    r = np.linalg.norm(ro)
    # Compute v^2:
    v2 = np.linalg.norm(vo)**2
    # Compute ang mom h^2:
    h2 = np.linalg.norm(np.cross(ro,vo))**2
    # find a [m] from vis-viva:
    a = (2/r - v2/k)**(-1)
    # mean motion:
    n = np.sqrt(k/(a**3))
    # ecc:
    e = np.sqrt( 1 - h2/(k*a) )
    # Eo:
    E0 = np.arccos(1/e*(1-r/a))
    # M0:
    M0 = E0 - e*np.sin(E0)
    # M(t = t):
    M = M0 + n*(t-to)
    # E:
    E = danby_solve(M, e, 1.e-9)
    # Delta E:
    deltaE = E - E0
    # s2:
    s2 = np.sin(deltaE/2)
    sinE = np.sin(E)
    # c2:
    c2 = np.cos(deltaE/2)
    cosE = np.cos(E)
    # s:
    s = 2*s2*c2
    # c:
    c = c2*c2 - s2*s2
    # f prime:
    fprime = 1 - c*e*cosE + s*e*sinE
    # f, g:
    f = 1 - 2*s2*s2*a/r
    g = 2*s2*(s2*e*sinE + c2*r/a)*(1./n)
    # new r:
    new_r = f*ro + g*vo
    # fdot, gdot:
    fdot = -(n*a*s) / (r*fprime)
    gdot = 1 - (2*s2*s2 / fprime)
    # new v:
    new_v = fdot*ro + gdot*vo
    
    return new_r*u.m, new_v*(u.m/u.s)



################################################################
# a set of functions for estimating how much of an orbit you
# need to observe to get a good handle on the orbit's velocity,
# acceleration, and 3rd derivative.

def orbit_fraction(sep, seperr, snr = 5):
    """ What fraction of an orbital period do you need to observe to characterize
        the velocity, acceleration, and 3rd derivatives to a given SNR?  That
        is, v/sigma_v = 5, etc.  This is a rough estimate derived from assuming 
        a circular face-on orbit.

      Parameters:
       -----------
       sep, seperr : flt
           observed separation and error in separation, any unit
       snr : flt
           desired signal to noise ratio.  Default = 5
        
       Returns:
       --------
        of_for_vel, of_for_acc, of_for_jerk : flt
            orbit fraction required to be observed to achieve desired snr 
            for understanding the velocity, acceleration, and jerk. 
            In decimal fraction.  ex: 0.01 = 1% of orbit
    """
    of_for_vel = (snr/5)*seperr/sep
    of_for_acc = (snr/5)*np.sqrt((9*seperr)/(5*sep))
    of_for_jerk = (snr/5)*(seperr/sep)**(1/3)
    return of_for_vel, of_for_acc, of_for_jerk


def orbit_fraction_observing_time(sep, seperr, period, snr = 5):
    """ Given a fractional postional uncertainty and a given orbital 
        period, what timespace do your observations need to cover to 
        achieve a desired SNR on velocity, acceleration, and jerk?
        Inputs:
            sep, seperr : observed separation and error in separation, any unit
            snr : desired signal to noise ratio.  Default = 5
            period : orbital period in years
        Returns:
            time needed to observe vel, acc, jerk to desired SNR in years.
    """
    from orbittools.orbittools import orbit_fraction
    v,a,j = orbit_fraction(sep, seperr, snr=snr)
    return v*period,a*period,j*period

def orbit_fraction_postional_uncertainty(time, period, sep = None, snr = 5):
    """ Given a certain observing timespan, what measurement precision
        is needed to obtain the desired SNR on vel, acc, and jerk?
        The orbit fraction is the ratio of observed time span to the period,
        and is also defined by the scale-free positional uncertainty given
        in the orbit_fraction() function.
        Inputs:
            time : observed time span in years
            period : orbital period in years
            sep : separation (optional)
            snr : desired signal-to-noise on measurements.  Currently set at 5.
        Returns:
            if separation is given, returns the required astrometric precision for
                snr = 5 for vel, acc, jerk in same units as sep input.
            if no separation if given, the scale-free positional uncertainty is 
                returned.
    """
    of = time/period
    v_scalefree_uncert = of
    a_scalefree_uncert = (of**2)*(5/9)
    j_scalefree_uncert = of**3
    if sep:
        return v_scalefree_uncert*sep, a_scalefree_uncert*sep, j_scalefree_uncert*sep
    else:
        return v_scalefree_uncert, a_scalefree_uncert, j_scalefree_uncert


#########################################################



