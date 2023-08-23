from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('default')

# %% Input settings

# Settings
fa_applied = 25  # degrees
n_excitations = 28  # number of excitations/spiral interleaves
TH = 15  # mm, slice thickness
m_0 = 1  # a.u., initial magnetization

# Load in profile
gz = np.load("gz_philips.npy")
gz /= np.max(gz)

# Find the peak's maximum value and index
max_index = np.argmax(gz)
peak_maximum = gz[max_index]

# Calculate half maximum value
half_maximum = peak_maximum / 2

# Find the indices where data crosses half maximum value
left_idx = np.argmin(np.abs(gz[:max_index] - half_maximum))
right_idx = max_index + np.argmin(np.abs(gz[max_index:] - half_maximum))

# Calculate FWHM, and normalize z to it
fwhm_index = right_idx-left_idx
fwhm = TH
z_res = fwhm/fwhm_index  # mm occupied by each z point
z = np.linspace(-len(gz)*z_res/2, len(gz) * z_res/2, len(gz))

# Trim the variables for better visualization
gz = gz[abs(z) < 3*TH/2]
z = z[abs(z) < 3*TH/2]

# Ideal slice selective case
gz_ideal = np.zeros_like(gz)
gz_ideal[abs(z) <= 0.5+TH/2] = 1  # Adjust for visualization

# Non-selective case
gz_non_selective = np.ones_like(gz)

# Independent variables
n = np.array(range(n_excitations))
n_mesh, z_mesh = np.meshgrid(n, z)
M_0 = m_0 * np.ones_like(n_mesh)

# Plot gz
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(z, gz_ideal, 'k-', label="Slice selective (ideal)")
ax1.plot(z, gz, 'k--', label="Slice-selective (real)")
ax1.set_title('g(z) slice-select profile', size=16)
ax1.set_xlabel(r'$z$', size=16)
ax1.set_ylabel(r'$g(z)$', size=16)
ax1.set_ylim((0, 1.25))
ax1.legend(fontsize=12, loc="upper right")


def s_n_gapped(gz, fa):

    # Convert FA to radians
    alpha = np.deg2rad(fa)

    # Signal equation
    s_n = np.zeros_like(z_mesh)
    for i in range(z.shape[0]):
        s_n[i, :] = M_0[i, :] * np.sin(alpha * gz[i])
        for j in range(n_excitations):
            s_n[i, j] *= (np.cos(alpha * gz[i]))**j
            s_n += 1e-19

    return s_n


def s_n_contiguous(gz, fa):

    # Convert FA to radians
    alpha = np.deg2rad(fa)

    # Signal equation
    s_n = np.zeros_like(z_mesh)
    for i in range(z.shape[0]):
        s_n[i, :] = M_0[i, :] * np.sin(alpha * gz[i])
        s_n[i, :] *= np.cos(alpha * gz[i-fwhm_index])**n_excitations
        for j in range(n_excitations):
            s_n[i, j] *= (np.cos(alpha * gz[i]))**j
            s_n += 1e-19

    return s_n


# Surface plot
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.set_box_aspect(aspect=None, zoom=0.9)
ax2.plot_surface(z_mesh, n_mesh, s_n_gapped(gz=gz_non_selective, fa=fa_applied), cmap=cm.jet,
                 linewidth=0, antialiased=False)
ax2.set_ylim([n_excitations, 0])
ax2.set_title("Non-selective", size=16)
ax2.set_xlabel("$z$", size=16)
ax2.set_ylabel("$n$", size=16)
ax2.set_zlabel(r"$M_{xy}$", size=16)

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.set_box_aspect(aspect=None, zoom=0.9)
ax3.plot_surface(z_mesh, n_mesh, s_n_gapped(gz=gz, fa=fa_applied), cmap=cm.jet,
                 linewidth=0, antialiased=False)
ax3.set_ylim([n_excitations, 0])
ax3.set_title("Slice-selective (gapped)", size=16)
ax3.set_xlabel("$z$", size=16)
ax3.set_ylabel("$n$", size=16)
ax3.set_zlabel(r"$M_{xy}$", size=16)

ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.set_box_aspect(aspect=None, zoom=0.9)
ax4.plot_surface(z_mesh, n_mesh, s_n_contiguous(gz=gz, fa=fa_applied), cmap=cm.jet,
                 linewidth=0, antialiased=False)
ax4.set_ylim([n_excitations, 0])
ax4.set_title("Slice-selective (contiguous)", size=16)
ax4.set_xlabel("$z$", size=16)
ax4.set_ylabel("$n$", size=16)
ax4.set_zlabel(r"$M_{xy}$", size=16)


def s_n_mean_z(s_n):

    # Find mean over z
    s_n_mean_z = np.mean(s_n, axis=0)

    return s_n_mean_z


# Visualize
fig, (ax1) = plt.subplots(figsize=(7, 7),
                          nrows=1, ncols=1)
# Do not include the first excitation
ax1.plot(n, s_n_mean_z(s_n_gapped(gz=gz_ideal, fa=fa_applied)),
         'k-', label="Slice-selective \n(ideal)")
ax1.plot(n, s_n_mean_z(s_n_gapped(gz=gz, fa=fa_applied)),
         'ko-', label="Slice-selective \n(gapped")
ax1.plot(n, s_n_mean_z(s_n_contiguous(gz=gz, fa=fa_applied)),
         'k--', label="Slice-selective \n(contiguous)")
ax1.set_title("Signal measurements", size=16)
ax1.set_xlabel(r"Excitation number $n$", size=16)
ax1.set_ylabel(r"Signal $\int s_n(z) dz$", size=16)
ax1.legend(fontsize=16)


def fa_meas(s_n, gz, fa):
    S1 = np.mean(s_n(gz=gz, fa=fa)[:, :n_excitations//2])
    S2 = np.mean(s_n(gz=gz, fa=fa)[:, n_excitations//2:])
    att = S2/S1
    fa_meas = np.arccos(att**(2/n_excitations))

    return fa_meas


# Range of flip angles
min_fa = 3
max_fa = 40
fa_range = np.linspace(min_fa, max_fa, 20)
fa_meas_gapped = np.zeros_like(fa_range)
fa_meas_contiguous = np.zeros_like(fa_range)

# Calculate the flip angle using the ideal pulse assumption equation
for i in range(fa_range.shape[0]):
    fa_meas_gapped[i] = np.rad2deg(
        fa_meas(s_n=s_n_gapped, gz=gz, fa=fa_range[i]))
    fa_meas_contiguous[i] = np.rad2deg(
        fa_meas(s_n=s_n_contiguous, gz=gz, fa=fa_range[i]))

# Visualize
fig, (ax1, ax2) = plt.subplots(figsize=(12, 5),
                               nrows=1, ncols=2)
ax1.plot(fa_range, fa_meas_gapped, 'ko-', label=r"Gapped")
ax1.plot(fa_range, fa_meas_contiguous, 'k*--', label=r"Contiguous")
ax1.plot(fa_range, fa_range, 'k-', label="True flip angle")
ax1.set_title(r"$cos^{-1}(S_2/S_1)^{2/N_s}$ vs. True", size=16)
ax1.set_xlabel("Applied flip angle (degrees)", size=16)
ax1.set_ylabel("Measured flip angle (degrees)", size=16)
ax1.legend(fontsize=14)

# Calculate correction factor
fa_correction_gapped = fa_range/fa_meas_gapped
fa_correction_contiguous = fa_range/fa_meas_contiguous

# Fitting a 2-dimensional polynomial to the data
degree = 10  # Choose the degree of the polynomial
coeffs_gapped = np.polyfit(fa_meas_gapped, fa_correction_gapped, degree)
poly_fa_gapped = np.poly1d(coeffs_gapped)
coeffs_contiguous = np.polyfit(
    fa_meas_contiguous, fa_correction_contiguous, degree)
poly_fa_contiguous = np.poly1d(coeffs_contiguous)

# Generate the x values for the polynomial fit curve
fit_x_gapped = np.linspace(min(fa_meas_gapped), max(fa_meas_gapped), 100)
fit_y_gapped = poly_fa_gapped(fit_x_gapped)
fit_x_contiguous = np.linspace(
    min(fa_meas_contiguous), max(fa_meas_contiguous), 100)
fit_y_contiguous = poly_fa_contiguous(fit_x_contiguous)

# Plot polynomial fit
ax2.plot(fa_meas_gapped, fa_correction_gapped, 'ko',
         label="Gapped")
ax2.plot(fa_meas_contiguous, fa_correction_contiguous, 'k*',
         label="Contiguous")
ax2.plot(fit_x_gapped, fit_y_gapped, 'm-', label="Poly. fit - gapped")
ax2.plot(fit_x_contiguous, fit_y_contiguous,
         'm--', label="Poly. fit - contiguous")
ax2.legend(fontsize=14)
ax2.set_title(r"$FA_{meas}$ / $FA_{app}$", size=16)
ax2.set_xlabel("Measured flip angle (degrees)", size=16)
ax2.set_ylabel(r"$FA_{meas}$ / $FA_{app}$", size=16)
ax2.text(0.05, 0.5, r"$FA_{app} = \phi \times FA_{meas}$",
                    transform=ax2.transAxes,
                    bbox=dict(facecolor='white', alpha=0.5),
                    size=16)


# %% Simulate the correction process

def integral_gapped(x, gz):
    integrand = np.zeros_like(gz)
    for i in range(gz.shape[0]):
        integrand[i] = np.sin(x * gz[i])
        integrand[i] *= ((1 - (np.cos(x * gz[i])) ** n_excitations) /
                         (1 - np.cos(x * gz[i])))

    integral = np.mean(integrand)
    return integral


def integral_contiguous(x, gz):
    integrand = np.zeros_like(gz)
    for i in range(gz.shape[0]):
        integrand[i] = np.sin(x * gz[i]) * np.cos(x *
                                                  gz[i - fwhm_index]) ** n_excitations
        integrand[i] *= ((1 - (np.cos(x * gz[i])) ** n_excitations) /
                         (1 - np.cos(x * gz[i])))

    integral = np.mean(integrand)
    return integral


def integral_non_selective(x, gz):
    gz = np.ones_like(gz)
    integrand = np.zeros_like(gz)
    for i in range(gz.shape[0]):
        integrand[i] = np.sin(x * gz[i])
        integrand[i] *= ((1 - (np.cos(x * gz[i])) ** n_excitations) /
                         (1 - np.cos(x * gz[i])))

    integral = np.mean(integrand)
    return integral


def correction_factor(x, integral):
    # correction = (1/x)*(n_excitations) # Uncomment for real applications
    correction = (n_excitations)
    xr = np.deg2rad(x)
    correction /= integral(xr, gz)

    return correction


# Print settings
print("Input settings:")
print("Applied flip angle = " + str(fa_applied) + " deg")
print("Number of excitations = " + str(n_excitations))
print("Slice thickness = " + str(TH))
print("Initial magnetization = " + str(m_0))

# Print measurements
print()
print("Measurements:")
fa_measured = np.rad2deg(
    fa_meas(s_n=s_n_contiguous, gz=gz, fa=fa_applied))
print("Measured flip angle = " + str(np.round(fa_measured, 3)) + " deg")
fa_corrected = poly_fa_contiguous(fa_measured)*fa_measured
print("Measured and corrected flip angle = " +
      str(np.round(fa_corrected, 3)) + " deg")

# Print signals
print()
print("Signal estimations using analytical methods:")
S_mean = m_0 * \
    integral_contiguous(np.deg2rad(fa_corrected), gz) / (n_excitations)
print("Mean signal using measured and corrected flip angle of " +
      str(np.round(fa_corrected, 3)) + " degrees = " + str(np.round(S_mean, 3)))
gamma = correction_factor(fa_corrected, integral_contiguous)
print("Correction factor = " + str(np.round(gamma, 3)))

m_0_recovered = gamma * S_mean
print("Recovered magnetization after corrections = " +
      str(np.round(m_0_recovered, 3)))

print()
print("Signal estimations using numerical methods:")


def s_n_contiguous_no_decay(gz, fa):

    # Convert FA to radians
    alpha = np.deg2rad(fa)

    # Signal equation
    s_n = np.zeros_like(z_mesh)
    for i in range(z.shape[0]):
        s_n[i, :] = M_0[i, :] * np.sin(alpha * gz[i])
        s_n[i, :] *= np.cos(alpha * gz[i-fwhm_index])**n_excitations

    return s_n


def s_n_gapped_no_decay(gz, fa):

    # Convert FA to radians
    alpha = np.deg2rad(fa)

    # Signal equation
    s_n = np.zeros_like(z_mesh)
    for i in range(z.shape[0]):
        s_n[i, :] = M_0[i, :] * np.sin(alpha * gz[i])

    return s_n


def integral_contiguous_no_decay(x, gz):
    integrand = np.zeros_like(gz)
    for i in range(gz.shape[0]):
        integrand[i] = np.sin(x * gz[i])
        integrand[i] *= np.cos(x * gz[i - fwhm_index]) ** n_excitations
    integral = np.mean(integrand)
    return integral


def integral_gapped_no_decay(x, gz):
    integrand = np.zeros_like(gz)
    for i in range(gz.shape[0]):
        integrand[i] = np.sin(x * gz[i])
    integral = np.mean(integrand)
    return integral


S_mean = np.mean(s_n_mean_z(s_n_gapped(gz=gz, fa=fa_corrected)))
print("Mean signal using measured and corrected flip angle = " +
      str(np.round(S_mean, 3)))
S_0_mean = np.mean(s_n_mean_z(s_n_gapped_no_decay(gz=gz, fa=fa_corrected)))
print("Mean S0 using measured and corrected flip angle = " +
      str(np.round(S_0_mean, 3)))

fa = np.deg2rad(fa_corrected)
numerator = integral_gapped_no_decay(fa, gz)
denominator = integral_gapped(fa, gz) / n_excitations
gamma = numerator / denominator
print("Correction factor = " + str(np.round(gamma, 3)))

S_0_recovered = gamma * S_mean
print("Unattenuated image after corrections = " +
      str(np.round(S_0_recovered, 3)))

m_0_recovered = S_0_recovered/integral_gapped_no_decay(fa, gz)
print("Recovered M0 = " +
      str(np.round(m_0_recovered, 3)))
