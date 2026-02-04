from rbinvariantslib import models
import numpy as np
import PyGeopack as gp
import pandas as pd
import astropy
from spacepy import pycdf
from astropy.constants import R_earth
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def calc_transit_dist(time, xline_file, ead_file, plot=True, lon_nbrhood_size=30):
    """Estimate the transit distance from a X-line precipitation source to
    TRACERS.

    Arguments
    ---------
    time: time of the event 
    xline_file: Output from Trattner's Maximum Magnetic Shield Model X-line code
    ead_file: TRACERS EAD data file
    plot: set to True to make a 3D plot of the connectivity
    lon_nbrhood_size: Constraints the search; you may be asked to make this larger
      if the search ends up on edge of the neighborhood.

    Algorithm description
    ---------------------  
    Calculating the distances requires merging T96 with Trattner's Xline model, 
    which don't always agree on where the magnetopause is. To calculate the
    distance, I start traces along each point on Trattner's X-line within a 
    neighbor of the spacecraft's longitude. If that position is not within the
    T96 magnetopause, then I move the starting position inwards by an additional 
    2.5% and restart the trace until it is. Out of all these traces, I select
    the one that comes closest to the satellite position and then calculate its
    length.
    """
    df_xline = load_xline_file(xline_file, time)
    sc_pos = load_sc_pos(ead_file, time)
    model = get_tsyganenko_model(time)

    trace_points = do_traces(model, df_xline, sc_pos, lon_nbrhood_size)    
    best_points = find_closest_trace(trace_points, sc_pos, lon_nbrhood_size)
    length = calc_length(best_points)
    
    if plot:
        plot_traces(trace_points, df_xline)

    return length


def calc_length(best_points):
    dx = np.diff(best_points[:, 0])
    dy = np.diff(best_points[:, 1])
    dz = np.diff(best_points[:, 2])
    dr = np.sqrt(dx**2 + dy**2 + dz**2)
    
    length = np.sum(dr)
    
    print(f'The Length is {length}')    

    return length
    
    
def plot_traces(trace_points, df_xline):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x,y,z in trace_points:
        ax.plot(x,y,z)
    ax.plot(df_xline.x_sm, df_xline.y_sm, df_xline.z_sm, 'r-')

    # Create a sphere of radius 1
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    sphere_x = 1 * np.outer(np.cos(u), np.sin(v))
    sphere_y = 1 * np.outer(np.sin(u), np.sin(v))
    sphere_z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the sphere surface
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='b', alpha=0.3)
    
    ax.set_aspect('equal')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    
def find_closest_trace(trace_points, sc_pos, lon_nbrhood_size):
    best_points = None
    best_norm = np.inf
    best_idx = -1
    
    print(f'Number of traces: {len(trace_points)}')

    for i, points in enumerate(trace_points):
        points = np.array(points).T
        norms = np.linalg.norm(points - sc_pos, axis=1)
        idx_closest = np.argmin(norms)
        if norms[idx_closest] < best_norm:
            best_points = points
            best_norm = norms[idx_closest]
            best_idx = i
            print(f'Found better trace in trace_points[{i}]')

    if best_idx in (0, len(trace_points) - 1):
        raise RuntimeError(
            'The best trace is on the edge of the neighborhood. Please '
            'rerun the algorithm with a higher settings for '
            f'lon_nbrhood_size (current value {lon_nbrhood_size} deg)'
        )

    return best_points

            
def do_traces(
        model, df_xline, sc_pos, lon_nbrhood_size,
        d_sf=0.025, min_good_trace_len=10, max_good_r=1.1,
        trace_step_size=1e-3
):
    # Select points in neighborhood of satellite longitude
    sc_r, sc_lat, sc_lon = astropy.coordinates.cartesian_to_spherical(*sc_pos)
    mask = np.abs(df_xline.lon_sm - sc_lon.value) < np.deg2rad(lon_nbrhood_size)
    print(f'Will perform search for {mask.sum()} xline positions')

    # Do trace searches
    traces = []
    trace_starts = []
    rows = list(df_xline[mask].iterrows())
    
    for _, row in tqdm(rows, desc='Performing Trace Searches'):
        for i in itertools.count():
            sf = (1 - d_sf * i)
            pos = (row.x_sm * sf, row.y_sm * sf, row.z_sm * sf)
            trace = model.trace_field_line(pos, trace_step_size)
            min_r = np.linalg.norm(trace.points, axis=1).min()

            if trace.points.shape[0] > min_good_trace_len and min_r < max_good_r:
                break

        trace_starts.append(pos)
        traces.append(trace)

    # Take part of field line going down to earth
    trace_points = []
    iterable = zip(traces, trace_starts, df_xline[mask].iterrows())
    
    for trace, trace_start, (_, row) in iterable:
        i = np.argmin(np.linalg.norm(trace.points[1:] - trace_start, axis=1))
        x = trace.points[:i, 0]
        y = trace.points[:i, 1]
        z = trace.points[:i, 2]
        trace_points.append((x, y, z))

    return trace_points


def get_tsyganenko_model(time):
    # Get Tsyganenko Params from CDAWeb
    params = models.get_tsyganenko_params(time)

    print(params)
    
    # Setup cartesian grid
    xaxis = np.arange(0, 20, 0.1)
    yaxis = np.arange(-10, 10, 0.1)
    zaxis = np.arange(-10, 10, 0.1)    
    x, y, z = np.meshgrid(xaxis, yaxis, zaxis)

    # Evaluate model on the grid
    print('Computing T96 on grid...')
    
    model = models.get_tsyganenko(
        "T96", params, time,
        x_re_sm_grid=x,
        y_re_sm_grid=y,
        z_re_sm_grid=z,
        inner_boundary=1
    )

    return model

    
def load_sc_pos(ead_file, time): 
    # Load SM coordinates from EAD CDF file using SpacePy
    cdf = pycdf.CDF(ead_file)
    ead_time = cdf['Epoch'][:]
    ead_sm_pos = cdf['ts2_ead_r_sm'][:]
    cdf.close()   

    # Find closest point to target time
    i = np.argmin(np.abs(ead_time - time))
    sc_pos = ead_sm_pos[i] / R_earth.to_value('km')
    sc_alt = (np.linalg.norm(sc_pos) * R_earth - R_earth).to_value('km')
    
    print(f'EAD file index: {i}')
    print('Spacecraft Position (SM):', sc_pos)
    print('Spacecraft altitude:', sc_alt, 'km')

    return sc_pos
    

def load_xline_file(xline_file, time):
    # Load CSV using Pandas
    df_xline = pd.read_csv(xline_file, sep='\s+', skiprows=2)

    # Convert coordinates to SM 
    date = int('%04d%02d%02d' % (time.year, time.month, time.day))
    ut = time.hour + time.minute / 60 
    df_xline['x_sm'], df_xline['y_sm'], df_xline['z_sm'] = gp.Coords.GSMtoSM(
        df_xline.XlinePosX.values,
        df_xline.XlinePosY.values,
        df_xline.XlinePosZ.values,
        date,
        ut
    )

    # Add r, lon and lat in sm coordinates
    r, lat, lon = astropy.coordinates.cartesian_to_spherical(
        df_xline['x_sm'], df_xline['y_sm'], df_xline['z_sm'],
    )

    df_xline['r'] = r.value
    df_xline['lat_sm'] = lat.value
    df_xline['lon_sm'] = lon.value
    
    print(df_xline.head().to_string())
    
    return df_xline
