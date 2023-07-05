import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
mpl.use('Agg')  # Don't display the plot
from matplotlib import colors
import tables

filename = sys.argv[1]
# filename = "D://Puffin_BUILD/Puffin_BIN/examples/simple/3D/CEP/rectbeam_FEL_R0.96_up/0.2L_aperp/6microns_aperp_41_P_800.h5"

base_name = os.path.basename(filename)

# remove the extension
name_without_ext = os.path.splitext(base_name)[0]

# find the number before the extension
number = re.search(r'\d+$', name_without_ext)

h5 = tables.open_file(filename, 'r')
fieldin = h5.root.aperp.read()

# TArea = h5.root.runInfo._v_attrs.transArea
nx = h5.root.runInfo._v_attrs.nX
ny = h5.root.runInfo._v_attrs.nY
nz = h5.root.runInfo._v_attrs.nZ2
Lc = h5.root.runInfo._v_attrs.Lc
Lg = h5.root.runInfo._v_attrs.Lg
gamma_0 = h5.root.runInfo._v_attrs.gamma_r
kappa = h5.root.runInfo._v_attrs.kappa

c = 299792458 # [m/s] speed of light in vacuum
eps0 = 8.8541878128e-12 # [F/m] vacuum permittivity
qe = 1.60217662e-19 # [C] electron charge
me = 9.10938356e-31 # kg

meshsizeX = h5.root.runInfo._v_attrs.sLengthOfElmX
meshsizeY = h5.root.runInfo._v_attrs.sLengthOfElmY
meshsizeZ2 = h5.root.runInfo._v_attrs.sLengthOfElmZ2
meshsizeXSI = meshsizeX*np.sqrt(Lc*Lg)
meshsizeYSI = meshsizeY*np.sqrt(Lc*Lg)
meshsizeZSI = meshsizeZ2*Lc
dxbar = h5.root.runInfo._v_attrs.sLengthOfElmX
dybar = h5.root.runInfo._v_attrs.sLengthOfElmY
dz2 = h5.root.runInfo._v_attrs.sLengthOfElmZ2
wavelengths = h5.root.runInfo._v_attrs.lambda_r
lenz2 = (nz-1)*dz2

xaxis = np.arange(0,nx)*dxbar
xaxisplot = np.linspace((1-nx)/2*dxbar*np.sqrt(Lc*Lg)*1000,(nx-1)/2*dxbar*np.sqrt(Lc*Lg)*1000,nx) # unit of (mm)

yaxis = np.arange(0,ny)*dybar
yaxisplot = np.linspace((1-ny)/2*dybar*np.sqrt(Lc*Lg)*1000,(ny-1)/2*dybar*np.sqrt(Lc*Lg)*1000,ny) # unit of (mm)

z2axis = np.arange(0,nz)*dz2
taxis = z2axis*Lc/c
z = z2axis*Lc/wavelengths

aperp_x = np.array(fieldin[0])
intensity = fieldin[0]**2+fieldin[1]**2 # scaled intensity

scaled_power3 = np.trapz(intensity,axis = 2 , dx=meshsizeX)
scaled_power2 = np.trapz(scaled_power3,axis = 1, dx=meshsizeY)

scaledPeakPower = np.max(scaled_power2)
scaledPulseEnergy = np.trapz(scaled_power2, dx=meshsizeZ2)

zc = int((aperp_x.shape[0]-1)/2)
yc = int((aperp_x.shape[1]-1)/2)
xc = int((aperp_x.shape[2]-1)/2)

# central node
aperp_1D = aperp_x[:, yc, xc] # index order by (z2, yb, xb)
max_aperp = np.argmax(aperp_1D)
aperp_limit = np.max(np.abs(aperp_x))
aperp_trans = aperp_x[zc,:,:]
aperp_xz2 = aperp_x[:,yc,:]
aperp_yz2 = aperp_x[:,:,xc]


intens = aperp_1D**2
mean_int = np.average(intensity, axis = 0)

bprofile_x = np.average(mean_int, axis = 0)
bprofile_y = np.average(mean_int, axis = 1)

# calculate for mean and sigma of Guassian for pre-fitting
mean = sum(xaxisplot * bprofile_x)*1. / sum(bprofile_x) # weighted arithmetic mean
sigma = np.sqrt(sum(bprofile_x * (xaxisplot - mean)**2)*1./ sum(bprofile_x))

def gaussianIntensity(xdata, I0, mean, waist):
    return I0*np.exp(-2*(xdata-mean)**2/(waist)**2)

# Gaussian fit
popt, pcov = curve_fit(gaussianIntensity, xaxisplot, bprofile_x, p0=[max(bprofile_x), mean, sigma]) # p0=initial 
# print('intensity = ' + str(popt[0]))
# print('beam centre = ' + str(popt[1]) +' mm')
# print('beam waist = ' + str(popt[2]) +' mm')

# output format: fileNumber AverageIntensity WaistSize(mm) ScaledPeakPower ScaledPulseEnergy  
print(number.group(0), str(popt[0]), str(popt[2]), scaledPeakPower, scaledPulseEnergy, sep='\t')
h5.close()


# mpl.rcParams['font.sans-serif'] = "Arial"
# mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.size'] = 16  # adjust as needed

fig = plt.figure(figsize=(9, 9))

gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

ax_beam = fig.add_subplot(gs[1, 0])

# Normalize the mean_int data to range [0,1]
mean_int_norm = (mean_int - np.min(mean_int)) / (np.max(mean_int) - np.min(mean_int))

norm = colors.Normalize(vmin=0, vmax=1)  # Normalization for colorbar
contourf_plot = ax_beam.contourf(xaxisplot,yaxisplot,mean_int_norm, 512, cmap='jet', norm=norm)
# contourf_plot=ax_beam.contourf(xaxisplot,yaxisplot,mean_int, 512, cmap='jet')
ax_beam.tick_params('x')
ax_beam.set_xlabel('$x$ (mm)')
ax_beam.set_ylabel('$y$ (mm)')
ax_beam.set_aspect('equal')  # Set aspect ratio to equal

# Add colorbar, specifying contour plot and axes
# cbar_ax = fig.add_axes([0.86, ax_beam.get_position().y0, 0.02, ax_beam.get_position().height])
# cbar = fig.colorbar(contourf_plot, ax=cbar_ax)
# cbar.set_label('Colorbar Label')
# cbar.set_ticks([0, 0.5, 1])  # Set specific ticks on the colorbar

ax_bp_x = fig.add_subplot(gs[0, 0], sharex=ax_beam)
ax_bp_x.plot(xaxisplot, bprofile_x, label='beam profile')
ax_bp_x.plot(xaxisplot, gaussianIntensity(xaxisplot, *popt), 'r--', label='Gaussian fit:\n$w$ =%5.2f mm' % tuple(popt)[2])
ax_bp_x.tick_params('x', labelbottom=False)
ax_bp_x.set_ylabel('Intensity (a.u.)')
ax_bp_x.legend(frameon=False, prop={'size': 14}, loc='upper right',  bbox_to_anchor=(1, 1))

ax_bp_y = fig.add_subplot(gs[1, 1], sharey=ax_beam)
ax_bp_y.plot(bprofile_y, yaxisplot,)
ax_bp_y.tick_params('y', labelleft=False)
ax_bp_y.set_xlabel('Intensity (a.u.)')

fig.savefig(filename[:-3] + '_BeamProfile.png', dpi=300)

# plt.show
