from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('default')

# %% Input settings

# Settings
fa_applied = 25.16  # degrees
n_excitations = 29  # number of excitations/spiral interleaves
TH = 15  # mm, slice thickness
m_0 = 1  # a.u., initial magnetization

# Independent variables
n = np.array(range(n_excitations))
z = np.arange(-1.5*TH, 1.5*TH, 0.25)
n_mesh, z_mesh = np.meshgrid(n, z)
M_0 = m_0 * np.ones_like(n_mesh)

# RF functions


def gz_rect(z, TH):
    gz = np.where(np.abs(z) <= TH / 2, 1, 0)
    return gz


def gz_sinc(z, TH):
    sigma = TH / (4 * np.sqrt(2 * np.log(2)))
    gz = abs(np.sinc(z / (sigma * np.pi)))
    return gz


def gz_sinc(z, TH):
    gz = np.where(np.abs(z) <= TH / 2, 1, 0) * 1.0
    mp = z < 0
    mn = z >= 0
    gz_p = 1.25*mp
    gz_n = 0.75*mn
    gz *= gz_p + gz_n
    return gz


def gz_sinc(x, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-(x)**2 / (2 * sigma**2))
    return gaussian


plt.figure(figsize=(7, 5))
plt.plot(z, gz_rect(z, TH), 'k')
plt.plot(z, gz_sinc(z, TH), 'k--')
plt.xlabel(r'$z$', size=16)
plt.ylabel(r'$g(z)$', size=16)
# plt.ylim((0,1))
plt.legend(("Non-selective", "Slice-selective"),
           fontsize=12, loc="upper right")
# plt.title('Simulated RF pulse shape', size=20)


# %% Signal simulations

# Signal function
def s_n(gz, z, fa_applied):

    # Convert FA to radians
    fa = np.deg2rad(fa_applied)

    # Signal equation
    s_n = np.zeros_like(z_mesh)
    for i in range(z.shape[0]):
        s_n[i, :] = M_0[i, :] * np.sin(fa * gz(z[i], TH))
        s_n[i, :] *= (np.cos(fa * gz(z[i] - TH, TH)))**n_excitations
        for j in range(n_excitations):
            s_n[i, j] *= (np.cos(fa * gz(z[i], TH)))**j
            s_n += 1e-19

    return s_n


# Signal function with slice gap
def s_n_gapped(gz, z, fa_applied):

    # Convert FA to radians
    fa = np.deg2rad(fa_applied)

    # Signal equation
    s_n = np.zeros_like(z_mesh)
    for i in range(z.shape[0]):
        s_n[i, :] = M_0[i, :] * np.sin(fa * gz(z[i], TH))
        for j in range(n_excitations):
            s_n[i, j] *= (np.cos(fa * gz(z[i], TH)))**j
            s_n += 1e-19

    return s_n


# fig, ([ax1, ax2],[ax3, ax4]) = plt.subplots(figsize=(12, 12),
#                                  nrows=2, ncols=2, subplot_kw={"projection": "3d"})

fig = plt.figure(figsize=(12, 12))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(z, gz_rect(z, TH), 'k')
ax1.plot(z, gz_sinc(z, TH), 'k--')
ax1.set_xlabel(r'$z$', size=16)
ax1.set_ylabel(r'$g(z)$', size=16)
ax1.set_title("Pulse shape", size=16)
ax1.legend(("Non-selective", "Slice-selective"),
           fontsize=12, loc="upper right")

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(z_mesh, n_mesh, s_n(gz_rect, z, fa_applied), cmap=cm.jet,
                 linewidth=0, antialiased=False)
ax2.set_ylim([n_excitations, 0])
ax2.set_zlim([0, np.max(s_n(gz_sinc, z, fa_applied))])
ax2.set_title("Non-selective", size=16)
ax2.set_xlabel("$z$", size=16)
# ax2.set_xticks(ticks=[-TH,-TH/2, 0, TH/2,TH])
# ax2.set_xticklabels(['-TH','-TH/2', '0', 'TH/2', 'TH'])
ax2.set_ylabel("$n$", size=16)
ax2.set_zlabel(r"$M_{xy}$", size=16)

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(z_mesh, n_mesh, s_n_gapped(gz_sinc, z, fa_applied), cmap=cm.jet,
                 linewidth=0, antialiased=False)
ax3.set_ylim([n_excitations, 0])
ax3.set_zlim([0, np.max(s_n(gz_sinc, z, fa_applied))])
ax3.set_title("Slice-selective (gapped)", size=16)
ax3.set_xlabel("$z$", size=16)
ax3.set_ylabel("$n$", size=16)
ax3.set_zlabel(r"$M_{xy}$", size=16)

ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot_surface(z_mesh, n_mesh, s_n(gz_sinc, z, fa_applied), cmap=cm.jet,
                 linewidth=0, antialiased=False)
ax4.set_ylim([n_excitations, 0])
ax4.set_zlim([0, np.max(s_n(gz_sinc, z, fa_applied))])
ax4.set_title("Slice-selective (contiguous)", size=16)
ax4.set_xlabel("$z$", size=16)
ax4.set_ylabel("$n$", size=16)
ax4.set_zlabel(r"$M_{xy}$", size=16)

# %% Flip angle measurements

# Flip angle function


def fa_meas(gz, z, fa_applied):
    S1 = np.mean(s_n(gz, z, fa_applied)[:, :n_excitations//2])
    S2 = np.mean(s_n(gz, z, fa_applied)[:, n_excitations//2:])
    att = S2/S1
    fa_meas = np.arccos(att**(2/n_excitations))

    return fa_meas


print("Ideal flip angle measurement: " +
      str(np.rad2deg(fa_meas(gz_rect, z, fa_applied))))
print("Slice selective flip angle measurement: " +
      str(np.rad2deg(fa_meas(gz_sinc, z, fa_applied))))
print("Ratio: " +
      str(fa_meas(gz_rect, z, fa_applied) / fa_meas(gz_sinc, z, fa_applied)))


# %% Signal simulations

# Signal function
def s_n_mean_z(s_n):

    # Find mean over z
    s_n_mean_z = np.mean(s_n, axis=0)

    return s_n_mean_z


# Visualize
fig, (ax1) = plt.subplots(figsize=(7, 7),
                          nrows=1, ncols=1)
ax1.plot(n, s_n_mean_z(s_n(gz_rect, z, fa_applied)), 'k')
ax1.plot(n, s_n_mean_z(s_n_gapped(gz_sinc, z, fa_applied)), 'ko--')
ax1.plot(n, s_n_mean_z(s_n(gz_sinc, z, fa_applied)), 'k--')
# ax1.set_title("Signal measurements", size=16)
ax1.set_xlabel(r"Excitation number $n$", size=16)
ax1.set_ylabel(r"Signal $\int s_n(z) dz$", size=16)
ax1.legend(
    ("Non-selective", "Slice-selective \n (gapped)", "Slice-selective \n (contiguous slices)"), fontsize=12)


# %% Range of flip angles

# Range of flip angles
fa_range = np.linspace(1, 40, 20)
fa_meas_rect = np.zeros_like(fa_range)
fa_meas_sinc = np.zeros_like(fa_range)

# Calculate the flip angle using the ideal pulse assumption equation
for i in range(fa_range.shape[0]):
    fa = fa_range[i]
    fa_meas_rect[i] = np.rad2deg(fa_meas(gz_rect, z, fa))
    fa_meas_sinc[i] = np.rad2deg(fa_meas(gz_sinc, z, fa))

# Visualize
fig, ([ax1, ax2]) = plt.subplots(figsize=(12, 5),
                                 nrows=1, ncols=2)
ax1.plot(fa_range, fa_meas_rect, 'k*-')
ax1.plot(fa_range, fa_meas_sinc, 'r*--')
ax1.set_title("Flip angle measurements", size=16)
ax1.set_xlabel("Applied flip angle (degrees)", size=16)
ax1.set_ylabel("Measured flip angle (degrees)", size=16)
ax1.legend(("True flip angle", r"$cos^{-1}(S_2/S_1)^{2/N_s}$"), fontsize=14)

fa_correction = fa_meas_rect/fa_meas_sinc

# Fitting a 2-dimensional polynomial to the data
degree = 10  # Choose the degree of the polynomial
coeffs = np.polyfit(fa_meas_sinc, fa_correction, degree)
poly_fit = np.poly1d(coeffs)

# Generate the x values for the polynomial fit curve
fit_x = np.linspace(min(fa_meas_sinc), max(fa_meas_sinc), 100)
fit_y = poly_fit(fit_x)

# Plot polynomial fit
ax2.plot(fa_meas_sinc, fa_correction, 'k*',
         label=r"$FA_{meas}$ / $FA_{app}$")
ax2.plot(fit_x, fit_y, 'r', label="Polynomial Fit")
ax2.legend(fontsize=14)
ax2.set_title("Ratio of flip angle measurments", size=16)
ax2.set_xlabel("Measured flip angle (degrees)", size=16)
ax2.set_ylabel(r"$FA_{meas}$ / $FA_{app}$", size=16)
ax2.text(0.05, 0.5, r"$FA_{app} = \phi \times FA_{meas}$",
                    transform=ax2.transAxes,
                    bbox=dict(facecolor='white', alpha=1),
                    size=16)

exp_fit = False
if exp_fit:
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c

    # Perform the exponential fit
    params, covariance = curve_fit(
        exponential_func, fa_meas_sinc, fa_correction)

    # Get the optimized parameters
    a_fit, b_fit, c_fit = params

    # Generate the y values for the exponential fit curve
    fit_y = exponential_func(fit_x, a_fit, b_fit, c_fit)
    ax2.plot(fit_x, fit_y, 'y', label="Exponential Fit")
    ax2.legend(fontsize=14)


# %% Simulate the correction process

# Print settings
print("Input settings:")
print("Applied flip angle = " + str(fa_applied) + " deg")
print("Number of excitations = " + str(n_excitations))
print("Slice thickness = " + str(TH))
print("Initial magnetization = " + str(m_0))

# Print measurements
print()
print("Measurements:")
fa_measured = np.rad2deg(fa_meas(gz_sinc, z, fa_applied))
print("Measured flip angle = " + str(np.round(fa_measured, 3)) + " deg")
fa_corrected = poly_fit(fa_measured)*fa_measured
print("Measured and corrected flip angle = " +
      str(np.round(fa_corrected, 3)) + " deg")

# Print signals
print()
print("Signal estimations:")
fa = np.deg2rad(fa_corrected)
S_mean_z = m_0 * np.sin(fa * gz_sinc(z, TH)) * \
    (np.cos(fa * gz_sinc(z - TH, TH)))**n_excitations
S_mean_z /= n_excitations
S_mean_z /= (1 - np.cos(fa))
S_mean_z *= (1 - (np.cos(fa)**n_excitations))
S_mean = np.mean(S_mean_z)
print("Mean signal using measured and corrected flip angle of " +
      str(np.round(fa_corrected, 3)) + " degrees = " + str(np.round(S_mean, 3)))
S_0_z = m_0 * np.sin(fa * gz_sinc(z, TH)) * \
    (np.cos(fa * gz_sinc(z - TH, TH)))**n_excitations
S_0_mean = np.mean(S_0_z)
print("Mean S0 using measured and corrected flip angle of " +
      str(np.round(fa_corrected, 3)) + " degrees = " + str(np.round(S_0_mean, 3)))
gamma = S_0_mean/S_mean
print("Correction factor = " + str(np.round(gamma, 3)))
denominator_z = np.sin(fa * gz_sinc(z, TH)) * \
    (np.cos(fa * gz_sinc(z - TH, TH)))**n_excitations
denominator_z /= n_excitations
denominator_z /= (1 - np.cos(fa))
denominator_z *= (1 - (np.cos(fa)**n_excitations))
denominator = np.mean(denominator_z)
m_0_recovered = S_mean
m_0_recovered /= denominator
# m_0_recovered /= fa
print("Recovered magnetization after corrections = " +
      str(np.round(m_0_recovered, 3)))
