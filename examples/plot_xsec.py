from matplotlib.colors import LogNorm
from pyglimer.ccp.ccp import read_ccp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs

# Load Stack
ccp = read_ccp("US_P_0.58_minrad_3D_it_f2.pkl")

# Conclude and keep water, nicer for the illumination plots
ccp.conclude_ccp(keep_water=True)

# Peter, figure, 5.18
lat, lon = np.array([42.5, 42.5]), np.array([-127, -63])

# Get Cross section
slat, slon, sdists, qlat, qlon, qdists, qz, qillum, qccp, epi_area = \
    ccp.get_profile(lat, lon)

# Get depth slice (mainly for the illumination)
zqlat, zqlon, zqill, zqccp, zextent, z0 = ccp.get_depth_slice(z0=410)
zalpha = np.where(zqill == 0, 0, 0.5)

# Define norms
# Define norms
vmin=-0.25
vmax=0.25

snorm = Normalize(vmin=0, vmax=sdists[-1])
rfnorm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0.0)
illumnorm = LogNorm(vmin=1, vmax=zqill.max())

# Set illumination boundaries for section plotting
alpha1 = 50
minillum = 25
alpha = qillum/alpha1
alpha = np.where(alpha >= 1, 1.0, alpha)


# ############### Plot map ###################
figure()
geoax = axes(projection=ccrs.PlateCarree())
plot_map(fill=False)
geoax.tick_params(labelright=False, labeltop=False)

# Plot illumination
im = geoax.imshow(zqill, alpha=zalpha, extent=zextent, norm=illumnorm, transform=ccrs.PlateCarree(),
                  cmap='magma', origin='lower')

# Plot cross section
geoax.plot(qlon, qlat, 'k', zorder=0, transform=ccrs.PlateCarree())

# Plot waypoints
geoax.scatter(slon, slat, c=sdists, s=50, cmap='Greys',
              norm=snorm, marker='o', edgecolor='k', zorder=10, transform=ccrs.PlateCarree())

# Plot buffer that shows where we got cross section stuff from
_ = plot_line_buffer(qlat, qlon, delta=epi_area, zorder=5, linestyle='--',
                     linewidth=1.0, alpha=1.0, facecolor='none', edgecolor='k')

# Create colorbar from artifical scalarmappable (alpha is problematic)
c = colorbar(ScalarMappable(cmap=get_cmap('magma'), norm=illumnorm),
             orientation='vertical', aspect=40)

# Set colormap alpha manually
c.solids.set(alpha=0.5)


# ############### Plot section ###################

# Plot section
rfcmap = 'seismic'

figure()
ax = axes(facecolor=(0.8, 0.8, 0.8))

# Plot section
imshow(qccp, cmap='seismic', norm=rfnorm,
       extent=[0, np.max(qdists), np.max(qz), np.min(qz)], aspect='auto',
       alpha=alpha, rasterized=True)

c = colorbar(ScalarMappable(cmap=rfcmap, norm=rfnorm), aspect=40)
c.set_label('A', rotation=0)
# Plot waypoints
ax.scatter(sdists, np.min(qz) * ones_like(sdists), c=sdists, s=50, cmap='Greys',
           norm=snorm, marker='o', zorder=10, edgecolor='k', clip_on=False)
xlabel('Distance along cross section [$^\circ$]')
ylabel('Depth [km]')
