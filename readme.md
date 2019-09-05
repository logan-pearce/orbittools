# Functions for computing quantities in orbital dynamics.
Written by Logan Pearce, 2019 (with inspiration from Sarah Blunt)

**period:** given semi-major axis in au and central object mass in solar masses, return the period in years using Kepler's 3rd law.  If astropy unit objects are passed, returns an astropy unit object, otherwise returns a float.

**distance:** Computes distance from Gaia parallaxes using the Bayesian method of Bailer-Jones 2015.
    Input: parallax [mas], parallax error [mas]
    Returns: distance [pc], 1-sigma uncertainty in distance [pc]

**draw_orbits:** Draw a set of N trial orbits as part of the OFTI procedure.  Semi-major axis is fixed at 100 au and long. of asc. node is fixed at 0 deg.

**to_polar:** Converts RA/Dec [deg] of two binary components into separation and position angle of B relative 
        to A [mas, deg]

**parallax:** Given distance, compute the observed parallax.  Distance must be an astropy unit object.

**physical_separation:** Given distance and angular separation, return physical separation in AU.  Distance and angular sep must be astropy unit objects.

**angular_separation:** Given physical separation and distance, return the angular on-sky separation.  Distance and physical separation must be astropy unit open.

**keplerian_to_cartesian:** Given a set of Keplerian orbital elements, returns the observable 3-dimensional position, velocity, and acceleration at the specified time.  Accepts and arbitrary number of input orbits.  Semi-major axis must be an astropy unit object in physical distance (ex: au, but not arcsec).  The observation time must be converted into mean anomaly before passing into function.

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

**cartesian_to_keplerian:** Given observables XYZ position and velocity, compute orbital elements.  Position must be in
au and velocity in km/s.  Returns astropy unit objects for all orbital elements.

        Inputs:
            pos (3xN arr) [au]: position in xyz coords in au, with 
                        x = pos[0], y = pos[1], z = pos[2] for each of N orbits
                        +x = +Dec, +y = +RA, +z = towards observer
            vel (3xN arr) [km/s]: velocity in xyz plane.
        Returns:
            sma (1xN arr flt) [au]: semi-major axis in au, must be an astropy units object
            ecc (1xN arr flt) [unitless]: eccentricity
            inc (1xN arr flt) [deg]: inclination
            argp (1xN arr flt) [deg]: argument of periastron
            lon (1xN arr flt) [deg]: longitude of ascending node
            meananom (1xN arr flt) [radians]: mean anomaly 

