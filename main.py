import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load the CSV Data
# Referencing "E Plane.csv" and "H Plane.csv" verbatim
e_plane_csv = pd.read_csv('E Plane.csv')
h_plane_csv = pd.read_csv('H Plane.csv')

# Extracting CSV values for plotting
e_csv_theta = e_plane_csv['Theta [deg]'].values
e_csv_gain = e_plane_csv['normalize(GainTotal) []'].values
h_csv_theta = h_plane_csv['Theta [deg]'].values
h_csv_gain = h_plane_csv['normalize(GainTotal) []'].values

# 2. Antenna Parameters
c = 3e8
f = 6e9
k = 2 * np.pi * f / c
l = 10e-3
dx = 5e-3
dy = 25e-3
h = 12e-3

def patt(theta, phi):
    cos_psix = np.sin(theta) * np.cos(phi)
    cos_psiy = np.sin(theta) * np.sin(phi)
    cos_psiz = np.cos(theta)

    # Individual dipole radiation function
    sin_psiy = np.sqrt(1 - cos_psiy**2 + 1e-12)
    F0 = (np.cos(k * l * cos_psiy) - np.cos(k * l)) / sin_psiy

    # Group function X-axis (3 elements)
    Fs1 = 1 - 2 * np.cos(k * dx * cos_psix)

    # Group function Y-axis (5 elements)
    Fs2 = 1 + 2 * np.cos(k * dy * cos_psiy) + 2 * np.cos(k * 2 * dy * cos_psiy)

    # Reflector
    Fr = 2 * np.sin(k * h * cos_psiz)

    return F0 * Fs1 * Fs2 * Fr

# 3. Calculation of Resistance and Gain
N = 400
theta_grid = np.linspace(0, np.pi/2, N)
phi_grid = np.linspace(0, 2*np.pi, N)
d_theta = theta_grid[1] - theta_grid[0]
d_phi = phi_grid[1] - phi_grid[0]

TH, PH = np.meshgrid(theta_grid, phi_grid)
F_all = patt(TH, PH)

Rm = (30 / np.pi) * np.sum(np.abs(F_all)**2 * np.sin(TH)) * d_theta * d_phi
Fmax = np.max(np.abs(F_all))
D = 120 * (Fmax**2) / Rm
G = 10 * np.log10(D)

print(f"--- Vysledky bodu 2 ---")
print(f"R_sigma = {Rm:.2f} Ohm")
print(f"D_max   = {D:.2f}")
print(f"G_max   = {G:.2f} dB")

# 4. FIGURE 1: Cartesian and Polar Comparison
th_deg = np.linspace(-90, 90, 1000)
th_rad = np.deg2rad(th_deg)

Fe_norm = np.abs(patt(th_rad, np.pi/2)) / np.max(np.abs(patt(th_rad, np.pi/2)))
Fh_norm = np.abs(patt(th_rad, 0)) / np.max(np.abs(patt(th_rad, 0)))

plt.figure(1, figsize=(10, 8))
# E-Plane Cartesian
plt.subplot(2, 2, 1)
plt.plot(th_deg, Fe_norm, label='Python')
plt.plot(e_csv_theta, e_csv_gain, 'r--', label='Ansys')
plt.title("Rovina pola E"); plt.grid(True); plt.legend()

# H-Plane Cartesian
plt.subplot(2, 2, 2)
plt.plot(th_deg, Fh_norm, label='Python')
plt.plot(h_csv_theta, h_csv_gain, 'g--', label='Ansys')
plt.title("Rovina pola H"); plt.grid(True); plt.legend()

# E-Plane Polar
plt.subplot(2, 2, 3, projection='polar')
plt.plot(th_rad, Fe_norm, label='Python')
plt.plot(np.deg2rad(e_csv_theta), e_csv_gain, 'r--', label='Ansys')
plt.title("Rovina E")

# H-Plane Polar
plt.subplot(2, 2, 4, projection='polar')
plt.plot(th_rad, Fh_norm, label='Python')
plt.plot(np.deg2rad(h_csv_theta), h_csv_gain, 'g--', label='Ansys')
plt.title("Rovina H")
plt.tight_layout()

# 5. FIGURE 2: Polar Layout with 0.707 markers
fig2 = plt.figure(2, figsize=(10, 5))
ticks_deg = np.arange(-90, 91, 15)
for i, (model_data, csv_th, csv_g, title, col) in enumerate([
    (Fe_norm, e_csv_theta, e_csv_gain, "E", 'r'), 
    (Fh_norm, h_csv_theta, h_csv_gain, "H", 'g')
]):
    ax = fig2.add_subplot(1, 2, i+1, projection='polar')
    ax.plot(th_rad, model_data, lw=2, label='Model')
    ax.plot(np.deg2rad(csv_th), csv_g, color=col, linestyle='--', label='CSV')
    circle = np.linspace(-np.pi/2, np.pi/2, 100)
    ax.plot(circle, 0.707*np.ones_like(circle), 'k:', alpha=0.5, label='0.707')
    ax.set_thetagrids(ticks_deg)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_title(f"Rovina {title}")
    ax.legend(loc='lower right', fontsize='small')
plt.tight_layout()

# 6. FIGURE 3: 3D Radiation Pattern
N3 = 100
T3, P3 = np.meshgrid(np.linspace(0, np.pi/2, N3), np.linspace(0, 2*np.pi, N3))
F3 = np.abs(patt(T3, P3))
R3 = F3 / np.max(F3)
X = R3 * np.sin(T3) * np.cos(P3)
Y = R3 * np.sin(T3) * np.sin(P3)
Z = R3 * np.cos(T3)
fig3 = plt.figure(3, figsize=(7, 6))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax3.set_title("3D funkcia vyzarovania")

# 7. Beam Steering (30 Degrees)
theta_0 = np.deg2rad(30)
delta = -k * dy * np.sin(theta_0)

print(f"\n--- Vysledky bodu 3 ---")
print(f"Fazovy posun delta: {delta:.4f} rad ({np.rad2deg(delta):.2f}°)")

def patt_30(theta, phi):
    cos_psix = np.sin(theta) * np.cos(phi)
    cos_psiy = np.sin(theta) * np.sin(phi)
    cos_psiz = np.cos(theta)

    sin_psiy = np.sqrt(1 - cos_psiy**2 + 1e-12)
    F0 = (np.cos(k * l * cos_psiy) - np.cos(k * l)) / sin_psiy
    Fs1 = 2 * np.cos(0.5 * k * dx * cos_psix)

    psi_y = k * dy * cos_psiy
    Fs2_steer = 1 + 2*np.cos(1 * (psi_y + delta)) + 2*np.cos(2 * (psi_y + delta)) + 2*np.cos(3 * (psi_y + delta))

    Fr = 2 * np.sin(k * h * cos_psiz)
    return F0 * Fs1 * Fs2_steer * Fr

Fe_30_norm = np.abs(patt_30(th_rad, np.pi/2)) / np.max(np.abs(patt_30(th_rad, np.pi/2)))

# 8. FIGURE 4: Steering Comparison
plt.figure(4, figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(th_deg, Fe_norm, '--', label='Povodna (0°)', alpha=0.6)
plt.plot(th_deg, Fe_30_norm, 'r', label='Odklonena (30°)')
plt.title("Porovnanie roviny E (Kartézsky)")
plt.xlabel("theta [°]"); plt.ylabel("F norm [-]"); plt.grid(True); plt.legend()

ax_pol = plt.subplot(1, 2, 2, projection='polar')
ax_pol.plot(th_rad, Fe_norm, '--', label='0°', alpha=0.6)
ax_pol.plot(th_rad, Fe_30_norm, 'r', label='30°')
ax_pol.set_theta_zero_location("N"); ax_pol.set_theta_direction(-1)
ax_pol.set_thetamin(-90); ax_pol.set_thetamax(90)
ax_pol.set_title("Porovnanie roviny E (polarne)"); ax_pol.legend()

plt.tight_layout()
plt.show()