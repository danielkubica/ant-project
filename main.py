import numpy as np
import matplotlib.pyplot as plt

# Parametry anteny
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

    # Funkcia ziarenia jedneho dipolu
    sin_psiy = np.sqrt(1 - cos_psiy**2 + 1e-12)
    F0 = (np.cos(k * l * cos_psiy) - np.cos(k * l)) / sin_psiy

    # Skupinova funkcia v ose X (3 prvky)
    Fs1 = 1 - 2 * np.cos(k * dx * cos_psix)

    # Skupinova funkcia v ose Y (5 prvkov)
    Fs2 = 1 + 2 * np.cos(k * dy * cos_psiy) + 2 * np.cos(k * 2 * dy * cos_psiy)

    # Reflektor
    Fr = 2 * np.sin(k * h * cos_psiz)

    return F0 * Fs1 * Fs2 * Fr

# Vypocet odporu a zisku
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

# Grafy
th_deg = np.linspace(-90, 90, 1000)
th_rad = np.deg2rad(th_deg)

Fe = np.abs(patt(th_rad, np.pi/2))
Fe_norm = Fe / np.max(Fe)

Fh = np.abs(patt(th_rad, 0))
Fh_norm = Fh / np.max(Fh)

plt.figure(1, figsize=(10, 8))
plt.subplot(2, 2, 1); plt.plot(th_deg, Fe_norm); plt.title("Rovina pola E"); plt.grid(True)
plt.subplot(2, 2, 2); plt.plot(th_deg, Fh_norm); plt.title("Rovina pola H"); plt.grid(True)
plt.subplot(2, 2, 3, projection='polar'); plt.plot(th_rad, Fe_norm); plt.title("Rovina E")
plt.subplot(2, 2, 4, projection='polar'); plt.plot(th_rad, Fh_norm); plt.title("Rovina H")
plt.tight_layout()

fig2 = plt.figure(2, figsize=(10, 5))
ticks_deg = np.arange(-90, 91, 15)
for i, (data, title) in enumerate([(Fe_norm, "E"), (Fh_norm, "H")]):
    ax = fig2.add_subplot(1, 2, i+1, projection='polar')
    ax.plot(th_rad, data, lw=2)
    circle = np.linspace(-np.pi/2, np.pi/2, 100)
    ax.plot(circle, 0.707*np.ones_like(circle), 'r--', label='0.707')
    ax.set_thetagrids(ticks_deg)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_title(f"Rovina {title}")
plt.tight_layout()

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

# Odklon zvazku o 30 stupnov
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


Fe_30 = np.abs(patt_30(th_rad, np.pi/2))
Fe_30_norm = Fe_30 / np.max(Fe_30)

# FIGURE 4
plt.figure(4, figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(th_deg, Fe_norm, '--', label='Povodna (0°)', alpha=0.6)
plt.plot(th_deg, Fe_30_norm, 'r', label='Odklonena (30°)')
plt.title("Porovnanie roviny E (Kartézsky)")
plt.xlabel("theta [°]"); plt.ylabel("F norm [-]")
plt.grid(True); plt.legend()

ax_pol = plt.subplot(1, 2, 2, projection='polar')
ax_pol.plot(th_rad, Fe_norm, '--', label='0°', alpha=0.6)
ax_pol.plot(th_rad, Fe_30_norm, 'r', label='30°')
ax_pol.set_theta_zero_location("N")
ax_pol.set_theta_direction(-1)
ax_pol.set_thetamin(-90)
ax_pol.set_thetamax(90)
ax_pol.set_title("Porovnanie roviny E (polarne)")
ax_pol.legend()

plt.tight_layout()
plt.show()