from dataclasses import dataclass

from astropy import units
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from spacepy import pycdf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import lib_lockwood1992

CHAN_CUTOFF = 10


@dataclass
class TRACERSData:
    time: np.ndarray
    energies: np.ndarray
    flux: np.ndarray
    spect: np.ndarray

    def subset(self, stime, etime):
        i = np.searchsorted(self.time, stime)
        j = np.searchsorted(self.time, etime)
        
        return TRACERSData(
            time=self.time[i:j],
            energies=self.energies,
            flux=self.flux[i:j],
            spect=self.spect[i:j],
        )

    def plot_spect(self, fig=None, ax=None, Eic=False, Eic_times=None):
        if ax is None:
            fig = plt.figure(figsize=(12, 4))
            ax = plt.gca()
            
        im = ax.pcolor(self.time, self.energies[CHAN_CUTOFF:], self.spect.T[CHAN_CUTOFF:], norm=LogNorm())
        ax.set_yscale('log')
        ax.set_ylabel('Energy')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical').set_label('Summed Pitch Angle DEF')

        if Eic and Eic_times:
            dispersion_subset = self.subset(Eic_times[0], Eic_times[1])
            Eic = dispersion_subset.find_Eic()
            ax.plot(dispersion_subset.time, Eic, 'b*-', label='Eic: Low Energy Cutoff')
            ax.legend(loc='upper right', framealpha=1)
                
    def find_Eic(self, smooth=True, Eic_frac=0.1, window_size=5):
        Eic = np.zeros(self.time.size)
        Eic[:] = np.nan
        Eic_frac = 0.1
        
        for i in range(Eic.size):            
            idx_peak_energy = np.argmax(self.spect[i, CHAN_CUTOFF:]) + CHAN_CUTOFF
            j = idx_peak_energy
            
            while j >  CHAN_CUTOFF:
                if self.spect[i, j] < Eic_frac * self.spect[i, idx_peak_energy]:
                    Eic[i] = self.energies[j]
                    break
                j -= 1

        if smooth:
            Eic = self.smooth_Eic(Eic, window_size)
        
        return Eic 

    def smooth_Eic(self, Eic, window_size=5):
        """Smooth Eic with a mask of points to include in moving average.
        
        Args
          Eic: array
          window_size: integer, must be odd
        Returns
          Smoothed Eic array
        """            
        assert (window_size is None) or (window_size % 2 == 1),\
            'Window size must be odd'
    
        Eic_clean = Eic.copy()
        
        for i in range(Eic.size):
            total = 0.0
            count = 0
            
            for di in range(-window_size//2, window_size//2 + 1):
                if i + di > 0 and i + di < Eic.size:
                    total += Eic[i + di]
                    count += 1
            
            if count > 0:  # else left as nan
                Eic_clean[i] = total / count
                    
        return Eic_clean

    def calculate_recon_rate(self, mlat, alpha, d):
        """Calculate the reconnection rate

        Returns
            recon_rate, err_low, err_high
        """
        Eic = self.find_Eic()
        
        if mlat is None:
            mlat = -75 * np.ones(Eic.size)
            
        Eic_units = Eic * units.eV
        
        Ey_sat, Ey_mp = lib_lockwood1992.estimate_reconn_rate(
            self.time,
            Eic_units,
            mlat,
            alpha,
            d,
            return_error=True,
            ignore_uncertain=True,
        )

        return (
            Ey_mp['nominal'],
            Ey_mp['err_low'],
            Ey_mp['err_high'],
        )
        
    def plot_recon_rate(self, stime, etime, mlat=None, alpha=0, d=12):
        # Subset dispersion data and calculate reconnection rate
        dispersion_subset = self.subset(stime, etime)
        Eic = dispersion_subset.find_Eic()

        recon_rate, err_low, err_high = dispersion_subset.calculate_recon_rate(
            mlat=mlat,
            alpha=alpha,
            d=d,
        )
    
        # Do plotting -----------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
        im = ax1.pcolor(self.time, self.energies[CHAN_CUTOFF:], self.spect.T[CHAN_CUTOFF:], norm=LogNorm())
        ax1.plot(dispersion_subset.time, Eic, 'b*-', label='Eic: Low Energy Cutoff')
        ax1.set_yscale('log')
        ax1.set_ylabel('Energy')
        ax1.legend(loc='upper right', framealpha=1)
        
        ax2.plot(dispersion_subset.time, recon_rate.value, label='Lockwood Reconnection Rate')
        ax2.fill_between(dispersion_subset.time, err_low.value, err_high.value, alpha=0.25)
        ax2.set_ylim(0, 2)
        ax2.set_ylabel('Reconnection Rate\n(mV/m)')
        ax2.grid(color='#ccc', linestyle='dashed')
        plt.legend()
        
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical').set_label('Diff En Flux\nSummed over P.A.')
        
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        
        return fig, (ax1, ax2)


def load_data(aci_file):
    # Load data from ACI file
    cdf = pycdf.CDF(aci_file)
    
    flux = cdf['ts1_l2_aci_tscs_def'][:]
    spect = flux.sum(axis=-1)
    energies = cdf['ts1_l2_aci_energy'][:]
    time = cdf['Epoch'][:]

    cdf.close()

    # Return TRACERSData instance
    return TRACERSData(
        time=time,
        energies=energies,
        flux=flux,
        spect=spect,
    )

    