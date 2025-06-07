import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# TRANSFORMATION DES UNITÉS : définition de la conversion
# =====================================================================

hbar     = 1.054571817e-34     # J·s
mass_He  = 6.646476e-27        # kg (masse d’un atome de 4He)
e_charge = 1.602176634e-19     # C = charge élémentaire (J/eV)
mu       = mass_He / 2         # masse réduite pour deux atomes identiques

# On fixe la demi-largeur “réelle” du puits en mètres (3 Å)
a_real   = 3e-10               # m 

# Conversion d’énergie : 1 unité-modèle = ħ² / (2 m_He * a_real²) en J
E_unit   = (hbar**2) / (2 * mass_He * a_real**2)  # J par “unité-modèle”

# =====================================================================
# BASE DE DÉPART : calcul de ψ(x) et V(x) pour les états n = 1..5
#  (on conserve le formalisme en unités-modèle, 
#   mais on convertit et affiche ensuite en J et en eV)
# =====================================================================

# Paramètres du puits (unités-modèle)
V0_model    = 20.0        # profondeur du puits en unités-modèle
a_model     = 1.0         # demi-largeur du puits en unités-modèle
hbar2_over_2m = 1.0       # ħ²/(2m) = 1 dans ces unités

# Énergies analytiques (puits carré) en unité-modèle pour n=1..5
n_vals      = np.arange(1, 6)
E_n_model   = (n_vals**2 * np.pi**2) / (4 * a_model**2) - V0_model

# Conversion des E_n en Joules puis en eV
E_n_real    = E_n_model * E_unit     # en J
E_n_eV      = E_n_real / e_charge    # en eV

print("Énergies analytiques (E_n) pour n=1..5 :")
for n, En_mod, En_J, En_eV in zip(n_vals, E_n_model, E_n_real, E_n_eV):
    tag = "(continuum)" if En_mod > 0 else "(lié)"
    # On imprime En_J et En_eV en notation scientifique
    print(f"  n = {n:>2} -> E_{n}(modèle) = {En_mod:>8.4f}   |   "
          f"E_{n}(réel) = {En_J:.3e} J   ≃   {En_eV:.3e} eV {tag}")

# ---------------------------------------------------------------------
# Fonction pour résoudre r, B, C, D pour une énergie E_modèle
# ---------------------------------------------------------------------
def solve_coefficients(E_mod, V0, a):
    """
    Pour énergie E_mod (unité-modèle), on résout :
      ψ_I(x<-a)      = e^{ikx} + r e^{-ikx}
      ψ_II(-a<=x<=a) = B e^{iqx} + C e^{-iqx}
      ψ_III(x>+a)    = D e^{ikx}
    k = sqrt(E_mod), q = sqrt(E_mod + V0). Conditions de continuité 
    de ψ et dψ/dx en x = ±a. Retourne (r, B, C, D).
    """
    k = np.sqrt(E_mod + 0j)
    q = np.sqrt(E_mod + V0 + 0j)

    e_mika = np.exp(-1j * k * a)
    e_pika = np.exp(1j * k * a)
    e_miqa = np.exp(-1j * q * a)
    e_piqa = np.exp(1j * q * a)

    A    = np.zeros((4, 4), dtype=complex)
    Bvec = np.zeros(4, dtype=complex)

    # Continuité en x = -a
    A[0, 0] = e_pika         # r · e^{+i k a}
    A[0, 1] = -e_miqa        # B · e^{-i q a}
    A[0, 2] = -e_piqa        # C · e^{+i q a}
    A[0, 3] = 0.0            # D · 0
    Bvec[0] = -e_mika        # - e^{-i k a}

    A[1, 0] = -1j * k * e_pika
    A[1, 1] = -1j * q * e_miqa
    A[1, 2] = +1j * q * e_piqa
    A[1, 3] = 0.0
    Bvec[1] = -1j * k * e_mika

    # Continuité en x = +a
    A[2, 0] = 0.0
    A[2, 1] = e_piqa         # B · e^{+i q a}
    A[2, 2] = e_miqa         # C · e^{-i q a}
    A[2, 3] = -e_pika        # - D · e^{+i k a}
    Bvec[2] = 0.0

    A[3, 0] = 0.0
    A[3, 1] = 1j * q * e_piqa
    A[3, 2] = -1j * q * e_miqa
    A[3, 3] = -1j * k * e_pika
    Bvec[3] = 0.0

    sol    = np.linalg.solve(A, Bvec)
    r, Bc, Cc, D = sol
    return r, Bc, Cc, D

# ---------------------------------------------------------------------
# 2) Tracé des fonctions d’onde pour n=1..5
#    (étiquette d’énergie en J & eV, notation scientifique)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(len(n_vals), 1, figsize=(8, 2.8 * len(n_vals)), sharex=True)

x_left   = -3 * a_model
x_right  = +3 * a_model
n_points = 1000
x_grid   = np.linspace(x_left, x_right, n_points)

for idx, n in enumerate(n_vals):
    E_test_mod = E_n_model[idx]
    r_test, B_test, C_test, D_test = solve_coefficients(E_test_mod, V0_model, a_model)

    psi_x = np.zeros_like(x_grid, dtype=complex)
    V_x   = np.zeros_like(x_grid)

    for j, xj in enumerate(x_grid):
        # Potentiel V(x) en unité-modèle
        V_x[j] = -V0_model if abs(xj) <= a_model else 0.0

        # Fonction d’onde ψ(x)
        if xj < -a_model:
            psi_x[j] = (np.exp(1j * np.sqrt(E_test_mod + 0j) * xj) 
                        + r_test * np.exp(-1j * np.sqrt(E_test_mod + 0j) * xj))
        elif xj > +a_model:
            psi_x[j] = D_test * np.exp(1j * np.sqrt(E_test_mod + 0j) * xj)
        else:
            psi_x[j] = (B_test * np.exp(1j * np.sqrt(E_test_mod + V0_model + 0j) * xj)
                        + C_test * np.exp(-1j * np.sqrt(E_test_mod + V0_model + 0j) * xj))

    ax = axes[idx]
    ax.plot(x_grid, np.real(psi_x), label="Re[ψ(x)]", color="navy")
    ax.plot(x_grid, np.imag(psi_x), label="Im[ψ(x)]", color="teal", linestyle="--")
    ax.plot(x_grid, np.abs(psi_x)**2, label="|ψ(x)|²", color="crimson")

    # Tracé du potentiel (V(x)/V0) en pointillés gris
    V_scaled = V_x / V0_model
    ax.plot(x_grid, V_scaled, "--", color="gray", label="V(x)/V₀")

    # Frontières du puits
    ax.axvline(-a_model, color="black", linestyle=":", linewidth=1.0)
    ax.axvline(+a_model, color="black", linestyle=":", linewidth=1.0)

    ax.set_ylim(-1.2, 1.2)
    ax.set_ylabel(r"$\psi$, $|\psi|^2$, $V/V_0$")

    # Énergie en J & eV (notation scientifique) dans le titre
    E_J      = E_test_mod * E_unit
    E_eV     = E_J / e_charge
    state_tag = "continuum" if E_test_mod > 0 else "lié"
    ax.set_title(
        f"n={n} (E = {E_J:.3e} J  ≃  {E_eV:.3e} eV, {state_tag})"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("x (unité-modèle)")
plt.tight_layout()
plt.show()


# =====================================================================
# DEUXIÈME PARTIE : Tracé de T(E), R(E) et repérage des pics T≈1
# Conversion de l’axe d’énergie en J & eV
# =====================================================================

# ---------------------------------------------------------------------
# 3) Définition de compute_TR(E_mod, V0, a) : identique
# ---------------------------------------------------------------------
def compute_TR(E_vals_mod, V0, a):
    T_list = np.zeros_like(E_vals_mod, dtype=float)
    R_list = np.zeros_like(E_vals_mod, dtype=float)
    for idx, E_mod in enumerate(E_vals_mod):
        r, Bc, Cc, D = solve_coefficients(E_mod, V0, a)
        T_list[idx] = np.real_if_close(abs(D)**2)
        R_list[idx] = np.real_if_close(abs(r)**2)
    return T_list, R_list

# ---------------------------------------------------------------------
# 4) Balayage de E_mod sur [E_min, E_max]
# ---------------------------------------------------------------------
E_min         = 1e-3
E_max         = 50.0
nE            = 2000

E_vals_model  = np.linspace(E_min, E_max, nE)
T_vals, R_vals= compute_TR(E_vals_model, V0_model, a_model)

# Conversion couleur de la grille d’énergie en J & eV
E_vals_real = E_vals_model * E_unit       # en J
E_vals_eV   = E_vals_real / e_charge      # en eV

# ---------------------------------------------------------------------
# 5) Repérer numériquement les pics T≈1 (Ramsauer)
# ---------------------------------------------------------------------
eps = 1e-3
indices_peak = np.where(T_vals > 1.0 - eps)[0]
energies_RT_model = []

if indices_peak.size > 0:
    blocs = np.split(indices_peak, np.where(np.diff(indices_peak) != 1)[0] + 1)
    for bloc in blocs:
        i_max = np.argmax(T_vals[bloc])
        energies_RT_model.append(E_vals_model[bloc][i_max])

energies_RT_real = np.array(energies_RT_model) * E_unit  # en J
energies_RT_eV   = energies_RT_real / e_charge         # en eV

# ---------------------------------------------------------------------
# 6) Tracé de T(E), R(E) en fonction de l’énergie réelle (en J),
#    annotations en J & eV (notation scientifique)
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(E_vals_real, T_vals, label=r"$T(E)$", color="blue")
plt.plot(E_vals_real, R_vals, "--", label=r"$R(E)$", color="red")
plt.ylim(-0.05, 1.05)
plt.xlim(E_vals_real.min(), E_vals_real.max())

# Axe horizontal : on précise “Énergie réelle E (J) — équivalent eV”
plt.xlabel(r"Énergie réelle $E$ (Joules) — équivalent en eV", fontsize=12)
plt.ylabel(r"Coefficients $T(E),\,R(E)$", fontsize=12)
plt.title("Effet Ramsauer–Townsend (échelle réelle) : $T(E)$ et $R(E)$", fontsize=13)
plt.grid(alpha=0.3)
plt.legend(loc="upper right")

# Lignes verticales aux énergies analytiques, en J & eV (notation scientifique)
for E_J, E_e in zip(E_n_real, E_n_eV):
    plt.axvline(E_J, color="orange", linestyle="--", alpha=0.6)
    plt.text(
        E_J, 0.70,
        f"{E_J:.3e} J\n{E_e:.3e} eV",
        rotation=90, color="orange", fontsize=8, verticalalignment="center"
    )

# Lignes verticales aux pics T≈1, en J & eV
for E_J, E_e in zip(energies_RT_real, energies_RT_eV):
    plt.axvline(E_J, color="magenta", linestyle=":", alpha=0.7)
    plt.text(
        E_J, 0.90,
        f"{E_J:.3e} J\n{E_e:.3e} eV",
        rotation=90, color="magenta", fontsize=8, verticalalignment="center"
    )

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 7) Impression des listes des énergies, en unités-modèle, en J & eV
# ---------------------------------------------------------------------
print("\nListe des énergies analytiques E_n :")
for n, En_mod, En_J, En_eV in zip(n_vals, E_n_model, E_n_real, E_n_eV):
    tag = "continuum" if En_mod > 0 else "lié"
    print(f"  n={n:>2} -> E_{n} (modèle) = {En_mod:.4f},   "
          f"E_{n} (réel) = {En_J:.3e} J ≃ {En_eV:.3e} eV ({tag})")

print("\nListe des énergies repérées numériquement (T≈1) :")
for i, (Ev_mod, Ev_J, Ev_eV) in enumerate(zip(energies_RT_model, energies_RT_real, energies_RT_eV), start=1):
    print(f"  Pic {i:>2} : E (modèle) ≃ {Ev_mod:.4f},   "
          f"E (réel) ≃ {Ev_J:.3e} J ≃ {Ev_eV:.3e} eV")
