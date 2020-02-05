def hyperbolic_anomaly(H,e,M):
    '''Eccentric anomaly function'''
    import numpy as np
    return H - (e*np.sin(H)) - M

def hyperbolic_solve(f, M0, e, h):
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
    H0 = M0
    lastH = H0
    nextH = lastH + 10* h 
    number=0
    while (abs(lastH - nextH) > h) and number < 1001: 
        new = f(nextH,e,M0) 
        lastH = nextH
        nextH = lastH - new / (1.-e*np.cos(lastH)) 
        number=number+1
        if number >= 1000:
            nextH = float('NaN')
    return nextH

def sma(vinf,M):
    """ Given hyperbolic excess velocity, compute semi-major axis for
        hyperbolic orbit around object of mass M.  vinf and M must
        be astropy unit objects
    """
    vinf = vinf.to(u.m/u.s)
    mu = c.G*(M.to(u.kg))
    return -mu/(vinf**2)

def ecc(vinf,M,R):
    """ Given hyperbolic excess velocity, compute eccentricity for
        hyperbolic orbit around object of mass M to periastron distance R.  
        vinf, R and M must be astropy unit objects
    """
    a = sma(vinf,M,R)
    e = 1 - R/a
    return e

def compute_psi(vinf,M,R):
    """ Given hyperbolic excess velocity, compute maximum deflection angle for
        hyperbolic orbit around object of mass M to periastron distance R.  
        vinf, R and M must be astropy unit objects.
    """
    e = ecc(vinf,M,R)
    psi = 2*np.arcsin(1/e)
    return psi

def impact_parameter(vinf,M,R):
    """ Given hyperbolic excess velocity, compute impact parameter required for
        hyperbolic orbit around object of mass M to periastron distance R.  
        vinf, R and M must be astropy unit objects.
    """
    a = sma(vinf,M,R)
    e = ecc(vinf,M,R)
    return -a*np.sqrt(e**2-1)
