"""Tools for calculating the reconnection rate using ionospheric plasma/field
measurements and solar wind data.

Uses method adapted from Lockwood 1992.
"""
from astropy import constants, units
from datetime import datetime, timedelta
import math
from matplotlib.dates import date2num
import numpy as np
import pytz

# Length of path the particle precipation travels down along the cusp field
# line. In Lockwood 1992, it is suggested the true length is between 15-30 Re.
# This is also represented by the the d' parameter.
PRECIP_TRAVEL_PATHS = np.array([10., 20., 30.,])

# TRACERS Satellite Altitude
TRACERS_ALTITUDE = 600

# Earth Dipole Moment
B0 = 30e3 * units.nT


def estimate_reconn_rate(t, Eic, mlat, alpha, ignore_uncertain=True, Bmp=50, altitude=TRACERS_ALTITUDE, Bs=None, Vs=7.8, precip_travel_paths=PRECIP_TRAVEL_PATHS, return_error=False):
    """Estimate reconnection rate using in-situ measurements and
    solar wind data. 

    Args
      t : array
        Times as datetimes
      Eic: array with units
        Eic low energy cutoff, units of energy
      mlat: array
        Magnetic latitude, in degrees
      alpha: float
         angle between satellite and merging gap in degrees deg
    Returns
      precip_travel_paths: precipitation travel paths (d' in paper)
      Ey_sat: reconnection rate at satellite (Ey in paper)
      Ey_mpause: reconnetion rate at magnetopause (Ey' in paper)
    """
    # Calculate required parameters. Interpolate everything to the time axis
    # of Eic.
    # ------------------------------------------------------------------------
    # dEic/dt
    dEic = np.diff(Eic.value).tolist()
    dEic.append(dEic[-1])
    dEic = np.array(dEic) * units.eV

    dt = [delta.total_seconds() for delta in np.diff(t)]
    dt.append(dt[-1])
    dt = np.array(dt) * units.s
    
    dEicdt = dEic/dt
    
    # Magnetc field at MP
    Bmp *= units.nT

    # Magnetic field at satellite
    if Bs is None:
        r = constants.R_earth  + altitude * units.km
        colat = np.deg2rad(90 - mlat.mean())
        Bs = B0 * (constants.R_earth / r)**3 * np.sqrt(1 + 3 * np.cos(colat)**2)
    print('Bs at satellite', Bs)
    
    # Satellite velocity
    Vs = Vs * units.km / units.s

    # Misc
    alpha = np.deg2rad(alpha)
    m = constants.m_p
    
    # Calculate the Reconnection Rate using equations derived in the paper
    # ------------------------------------------------------------------------
    # reconnection rate at satellite
    Ey_sat = {
        'nominal': [],
        'err_low': [],
        'err_high': []
    }

       # reconnection rate at magnetopause
    Ey_mpause = {
        'nominal': [],
        'err_low': [],
        'err_high': []
    }

    for precip_travel_path in precip_travel_paths:
        d = precip_travel_path * constants.R_earth

        for sf, key in [(1.0, 'nominal'), (0.5, 'err_low'), (1.5, 'err_high')]:            
            Ey = (
                (Bs * Vs * np.cos(alpha))
                /
                (1 + (d/2) * np.sqrt(m/2) * (Eic)**(-3/2) * sf * np.abs(dEicdt))
            )
            dy = np.sqrt(Bs / Bmp)
            
            Ey_final = Ey / dy
        
            Ey_sat[key].append(Ey.to(units.mV/units.m))
            Ey_mpause[key].append(Ey_final.to(units.mV/units.m))

    for key in Ey_sat.keys():
        Ey_sat[key] = np.array(Ey_sat[key])
        Ey_mpause[key] = np.array(Ey_mpause[key])

    # Remove points where DeltaEic=0
    
    if ignore_uncertain:
        for key in Ey_sat.keys():
            for i in range(len(precip_travel_paths)):
                Ey_sat[key][i, :-1][np.diff(Eic)==0] = np.nan
                Ey_mpause[key][i, :-1][np.diff(Eic)==0] = np.nan

    if return_error:
        return precip_travel_paths, Ey_sat, Ey_mpause
    else:
        return precip_travel_paths, Ey_sat['nominal'], Ey_mpause['nominal']

