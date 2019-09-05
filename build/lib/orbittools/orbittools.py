def period(sma_au,mass):
    """ Given semi-major axis in AU and mass in solar masses, return the period in years of an orbit using 
        Kepler's third law.
        Written by Logan Pearce, 2019
    """
    import numpy as np
    return np.sqrt(((sma_au)**3)/mass).value*u.yr

def draw_orbits(number):
    ''' Draw a set of trial orbits as part of the OFTI procedure
    Written by Logan Pearce, 2019
    '''
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
    lon = np.radians(0.0)  
    lon = np.array([lon]*number)
    # orbit fraction (fraction of orbit completed at observation date since reference date)
    orbit_fraction = np.random.uniform(0.0,1.0,number)
    return sma, ecc, inc, argp, lon, orbit_fraction

def eccentricity_anomaly(E,e,M):
    '''Eccentric anomaly function'''
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

def rotate_z(vector,theta):
    """ Rotate a 3D vector about the +z axis
        Inputs:
            vector: 3d vector array
            theta [rad]: angle to rotate the vector about
        Returns: rotated vector
    """
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]
              ])
    
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
    R = np.array([[[1.]*len(theta), 0., 0.],
              [0., np.cos(theta), -np.sin(theta)],
              [0., np.sin(theta), np.cos(theta)]  
              ])
    
    out = np.zeros((3,vector.shape[1]))
    out[0] = R[0,0]*vector[0,:] + R[0,1]*vector[1,:] + R[0,2]*vector[2,:]
    out[1] = R[1,0]*vector[0,:] + R[1,1]*vector[1,:] + R[1,2]*vector[2,:]
    out[2] = R[2,0]*vector[0,:] + R[2,1]*vector[1,:] + R[2,2]*vector[2,:]
    return out


def rotate_z_matmul(vector,theta):
    """ Rotate a 3D vector about the +z axis
        Inputs:
            vector: 3d vector array
            theta [rad]: angle to rotate the vector about
        Returns: rotated vector
    """
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]
              ])
    return np.matmul(Rz,vector)

def rotate_x_matmul(vector,theta):
    """ Rotate a 3D vector about the +z axis
        Inputs:
            vector: 3d vector array
            theta [rad]: angle to rotate the vector about
        Returns: rotated vector
    """
    Rx = np.array([[1, 0, 0],
               [0, np.cos(theta), -np.sin(theta)],
               [0, np.sin(theta), np.cos(theta)],      
              ])
    return np.matmul(Rx,vector)


def keplerian_to_cartesian(sma,ecc,inc,argp,lon,meananom,kep):
    """ Given Keplerian elements, return the position and velocity of an orbiting body in
        the plane of the sky.
        Inputs:
            sma (flt) [au]: semi-major axis in au, must be an astropy units object
            ecc (flt) [unitless]: eccentricity
            inc (flt) [deg]: inclination
            argp (flt) [deg]: argument of periastron
            lon (flt) [deg]: longitude of ascending node
            meananom (flt) [radians]: mean anomaly 
            kep (flt): kepler constant = mu/m where mu = G*m1*m2 and m = [1/m1 + 1/m2]^-1 . 
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
    nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(meananom, ecc)]
    E = np.array(nextE)
    #E = solve(eccentricity_anomaly, meananom, ecc, 0.001)

    # Compute position:
    pos = np.zeros((3,len(sma)))
    # In the plane of the orbit:
    pos[0,:], pos[1,:] = (sma*(np.cos(E) - ecc)).value, (sma*np.sqrt(1-ecc**2)*np.sin(E)).value
    # Rotate to plane of the sky:
    pos = rotate_z(pos, np.radians(argp))
    pos = rotate_x(pos, np.radians(inc))
    pos = rotate_z(pos, np.radians(lon))
    
    # compute velocity:
    vel = np.zeros((3,len(sma)))
    vel[0], vel[1] = (( -meanmotion * sma * np.sin(E) ) / ( 1- ecc * np.cos(E) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(E) ) / ( 1 - ecc * np.cos(E) )).to(u.km/u.s).value
    vel = rotate_z(vel, np.radians(argp))
    vel = rotate_x(vel, np.radians(inc))
    vel = rotate_z(vel, np.radians(lon))
    
    # Compute accelerations numerically
    acc = np.zeros((3,len(sma)))
    # Generate a nearby future time point(s) along the orbit:
    deltat = 0.002*u.yr
    # Compute new mean anomaly at future time:
    futuremeananom = meananom + meanmotion*((deltat).to(u.s))
    # Compute new eccentricity anomaly at future time:
    futureE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(futuremeananom.value, ecc)]
    futureE = np.array(futureE)
    # Compute new velocity at future time:
    futurevel = np.zeros((3,len(sma)))
    futurevel[0], futurevel[1] = (( -meanmotion * sma * np.sin(futureE) ) / ( 1- ecc * np.cos(futureE) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(futureE) ) / ( 1 - ecc * np.cos(futureE) )).to(u.km/u.s).value
    futurevel = rotate_z(futurevel, np.radians(argp))
    futurevel = rotate_x(futurevel, np.radians(inc))
    futurevel = rotate_z(futurevel, np.radians(lon))
    acc = (futurevel-vel)/deltat.value
    
    return np.transpose(pos)*u.au, np.transpose(vel)*(u.km/u.s), np.transpose(acc)*(u.km/u.s/u.yr)

def cartesian_to_keplerian(pos, vel, kep):
    """ Given observables XYZ position and velocity, compute orbital elements
        Inputs:
            pos (3xN arr) [au]: position in xyz coords in au, with 
                        x = pos[0], y = pos[1], z = pos[2] for each of N orbits
                        +x = +Dec, +y = +RA, +z = towards observer
            vel (3xN arr) [km/s]: velocity in xyz plane.
        Returns:
            sma (flt) [au]: semi-major axis in au, must be an astropy units object
            ecc (flt) [unitless]: eccentricity
            inc (flt) [deg]: inclination
            argp (flt) [deg]: argument of periastron
            lon (flt) [deg]: longitude of ascending node
            meananom (flt) [radians]: mean anomaly 
        Written by Logan Pearce, 2019, inspired by Sarah Blunt
    """
    import numpy as np
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

