# %%
import numpy as np
import matplotlib.pyplot as plt

print("my name")
def Rx(flip):
    # rotation matrix about the x-axis
    # Note these are left-handed rotation matrices for MRI

    cos_flip = np.cos(flip)
    sin_flip = np.sin(flip)

    Rx = np.array([[1, 0, 0],
                   [0, cos_flip, sin_flip],
                   [0, -sin_flip, cos_flip]])

    return Rx


def Ry(flip):
    # rotation matrix about the y-axis
    # Note these are left-handed rotation matrices for MRI

    cos_flip = np.cos(flip)
    sin_flip = np.sin(flip)

    Ry = np.array([[cos_flip, 0, sin_flip],
                   [0, 1, 0],
                   [-sin_flip, 0, cos_flip]])

    return Ry


def Rz(flip):
    # rotation matrix about the z-axis
    # Note these are left-handed rotation matrices for MRI

    cos_flip = np.cos(flip)
    sin_flip = np.sin(flip)

    Rz = np.array([[cos_flip, -sin_flip, 0],
                   [sin_flip, cos_flip, 0],
                   [0, 0, 1]])

    return Rz


def bloch_rotateXe(Mstart, T, B):
    # bloch_rotate - compute the rotation of the net magnetization for a given magnetic field
    #
    # INPUTS
    # Mstart - initial magnetization
    # T - duration [ms]
    # B = [Bx, By, Bz] - magnetic field [mT]
    # OUTPUTS
    # Mend - final magnetization

    GAMMA = -11.78  # kHz/mT

    flip = 2 * np.pi * GAMMA * np.linalg.norm(B) * T

    eta = np.arccos(B[2] / (np.linalg.norm(B) + np.finfo(float).eps))

    theta = np.arctan2(B[1], B[0])

    Mend = Rz(-theta).dot(Ry(-eta).dot(Rz(flip).dot(Ry(eta).dot(Rz(theta).dot(Mstart)))))

    return Mend, flip


gammabar = -11.78  # kHz/mT     Xenon129
M0 = 1
M_initial = np.array([0, 0, M0]).T
# mT/cm  Slice selection gradient (set for 1 cm slice with current RF shape)
Gs = 4.98e-2

RF_shape = np.array([66, -138, -359, -595, -845, -1107, -1378, -1658, -1943, -2230, -2517, -2800, -3076, -3341, -3592, -3824, -4034, -4217, -4370, -4488, -4567, -4602, -4591, -4530, -4414, -4240, -4007, -3710, -3348, -2918, -2421, -1853, -1216, -510, 266, 1110, 2019, 2991, 4024, 5114, 6256, 7446, 8678, 9947, 11247, 12571, 13911, 15262, 16614, 17960,
                    19293, 20604, 21885, 23129, 24326, 25471, 26554, 27570, 28511, 29371, 30144, 30826, 31410, 31894, 32274, 32547, 32712, 32767, 32712, 32547, 32274, 31894, 31410, 30826, 30144, 29371, 28511, 27570, 26554, 25471, 24326, 23129, 21885, 20604, 19293, 17960, 16614, 15262, 13911, 12571, 11247, 9947, 8678, 7446, 6256, 5114, 4024, 2991, 2019, 1110, 266])
Trf = 1  # pulse length in ms
slcgap = 0  # slice gap in percentage of slice thickness

N = len(RF_shape)  # number of samples in RF pulse
Np = 24  # number of RF excitations
flip = 25  # nominal flip angle
BWplot = 8  # excitation bandwidth in kHz for plot
Nf = 200  # number of points in profile plot
df = np.linspace(-BWplot, BWplot, Nf)

Gz = Gs
z = df / Gz / gammabar
dz = 2 * BWplot / Nf

indx = np.arange(-np.floor(N/2), np.floor((N-1)/2) + 1)
IN = indx / (N-1)

t = IN * Trf
dt = Trf / Nf

RF = (flip * np.pi / 180) * RF_shape / \
    np.sum(RF_shape) / (2 * np.pi * gammabar * dt)

# M = np.tile(M_initial, (Nf, 1)).T
M = np.zeros((3, Nf))
M[2, :] = 1

Mxy = np.zeros((Nf, Np), dtype=complex)
Mxy2 = np.zeros((Nf, Np), dtype=complex)
Mz = np.zeros((Nf, Np))
Mz2 = np.zeros((Nf, Np))

Snrect = np.zeros(Np)
leftTH1 = np.zeros(Np, dtype=int)
rightTH1 = np.zeros(Np, dtype=int)

for i in range(Np):
    M[(0, 1), :] = 0  # x and y components of M set to zero before RF pulse
    for n in range(len(t)):
        for f in range(Nf):
            M_rotated, temp = bloch_rotateXe(M[:, f], dt, [np.real(
                RF[n]), np.imag(RF[n]), df[f] / gammabar])
            M[0, f] = M_rotated[0]
            M[1, f] = M_rotated[1]
            M[2, f] = M_rotated[2]
    Mxy[:, i] = M[0, :] + 1j * M[1, :]
    Mz[:, i] = M[2, :]
    leftTH1[i] = np.argmin(
        np.abs(np.abs(Mxy[0:round(Nf/2), i]) / np.max(np.abs(Mxy[:, i])) - 0.5))
    rightTH1[i] = np.argmin(
        np.abs(np.abs(Mxy[-1:-round(Nf/2)-1:-1, i]) / np.max(np.abs(Mxy[:, i])) - 0.5))
    leftZ = z[leftTH1[i]]
    rightZ = z[Nf - rightTH1[i] + 1]

# Signal after each excitation for first slice (no contiguous slice effect)
Sn1 = np.sum(np.abs(Mxy), axis=0)

flipRect = np.zeros(df.shape)
flipRect[leftTH1[0]:-rightTH1[0]] = 1
Snrect = np.zeros(Np)

for i in range(Np):
    Snrect[i] = np.sum(flipRect * np.cos(np.radians(flip))**(i)
                       * np.sin(np.radians(flip)))  # Signal for ideal slice profile

gap = round(slcgap * (Nf - leftTH1[0] - rightTH1[0]) / 100)

M = np.zeros((3, Nf))
M[2, :] = 1
M[2, leftTH1[0] + gap:] = Mz[0:-(leftTH1[0] + gap), -1]
RF = (flip * np.pi / 180) * RF_shape / \
    np.sum(RF_shape) / (2 * np.pi * gammabar * dt)

for i in range(Np):
    M[0:2, :] = 0  # x and y components of M set to zero before RF pulse
    for n in range(len(t)):
        for f in range(Nf):
            M_rotated, temp = bloch_rotateXe(M[:, f], dt, [np.real(
                RF[n]), np.imag(RF[n]), df[f] / gammabar])
            M[0, f] = M_rotated[0]
            M[1, f] = M_rotated[1]
            M[2, f] = M_rotated[2]
    Mxy2[:, i] = M[0, :] + 1j * M[1, :]
    Mz2[:, i] = M[2, :]

Sn2 = np.sum(np.abs(Mxy2), axis=0)

efliprect = np.arccos(np.sum(Snrect[Np//2:]) / np.sum(Snrect[:Np//2])) * 2
eflip1 = np.arccos(np.sum(Sn1[Np//2:]) / np.sum(Sn1[:Np//2])) * 2
eflip2 = np.arccos(np.sum(Sn2[Np//2:]) / np.sum(Sn2[:Np//2])) * 2

plt.figure(1)
plt.subplot(311)
plt.plot(t, RF)
plt.grid(True)
plt.xlabel('time (ms)')
plt.ylabel('RF (mT)')
plt.title('RF Pulse')

plt.subplot(3, 1, 2)
plt.plot(df, flipRect * flip)
plt.plot(df, np.rad2deg(np.arcsin(np.abs(Mxy[:, 0]))))
plt.title('Flip angle profile')
plt.xlabel('f (kHz)')
plt.ylabel('Flip angle (degrees)')
plt.grid(True)

plt.subplot(313)
plt.plot(df, flipRect * np.sin(np.radians(flip)))
plt.plot(df, flipRect * np.cos(np.radians(flip))
         ** (Np-1) * np.sin(np.radians(flip)))
plt.plot(df, flipRect * np.cos(np.radians(flip))
         ** (Np//2) * np.sin(np.radians(flip)))
plt.plot(df, np.abs(Mxy[:, 0]))
plt.plot(df, np.abs(Mxy2[:, 0]))
plt.plot(df, np.abs(Mxy2[:, Np//2]))
plt.plot(df, np.abs(Mxy2[:, -1]))
plt.title('Transverse Magnetization')
plt.xlabel('f (kHz)')
plt.ylabel('abs(Mxy)')
plt.grid(True)

plt.figure(2)
plt.subplot(211)
X, Y = np.meshgrid(np.arange(1, Np + 1), 14 * np.linspace(-1, 1, Nf))
s1 = plt.contourf(X, Y, np.abs(Mxy), cmap='viridis')
plt.xlabel('Excitation number')
plt.ylabel('Z (mm)')
plt.title('abs(Mxy)')

plt.subplot(212)
s2 = plt.contourf(X, Y, np.abs(Mxy2), cmap='viridis')
plt.xlabel('Excitation number')
plt.ylabel('Z (mm)')
plt.title('abs(Mxy)')

plt.figure(3)
plt.plot(Snrect)
plt.plot(Sn1)
plt.plot(Sn2)
plt.xlabel('Excitation number')
plt.ylabel('Signal (arbitrary units)')
plt.legend(['Ideal pulse', 'Non-contiguous Slices', 'Contiguous Slices'])
plt.show()
