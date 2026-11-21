"""Check the simplified <D>, <D^2>, K_t (k_max = inf; slab + 1/s pair counts; plateau closed form)
against the exact Abel projection. See manuscript_notes/geometric_averages/01_turbulent_D.md."""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gridfree_kolmogorov import structure_function, I_INF
from test_gridfree_kolmogorov_model import C_shell
kmin = 2*2*np.pi/2.4
D = lambda s: structure_function(np.asarray(s, float), kmin)          # k_max = inf
A = (4/3)*I_INF*kmin**(2/3)
print(f"A = (4/3) I kmin^(2/3) = {A:.3f}")
# 1) C(s) exact pieces
for d in [0.02, 0.09, 0.3]:
    s = np.array([d*1.01, 0.5, 1.0, 2*(1-d)*0.99])
    print(f"Delta={d}: C/[pi D^2 (2-D)^2/(2s)] =", np.round(C_shell(s, d)/(np.pi*d**2*(2-d)**2/(2*s)), 6),
          " C(s<Delta)/[4pi(D - s/2)] at s=D/2,D/10:", np.round(C_shell(np.array([d/2, d/10]), d)/(4*np.pi*(d-np.array([d/2,d/10])/2)), 3))
# 2) constants J, J2
s = np.geomspace(1e-9, 2, 200001)
J, J2 = np.trapezoid(D(s)/s, s), np.trapezoid(D(s)**2/s, s)
print(f"J = int_0^2 D/s ds = {J:.4f}   J2 = int_0^2 D^2/s ds = {J2:.4f}   3 J2/J^2 = {3*J2/J**2:.4f}")
# 3) exact Abel averages
X = np.concatenate(([0.0], np.geomspace(1e-9, 2.0, 40000)))
def exact(ell, d):
    ss = np.sqrt(ell**2 + X**2); c = C_shell(ss, d); Ds = D(ss); W = np.trapezoid(c, X)
    return np.trapezoid(c*Ds, X)/W, np.trapezoid(c*Ds**2, X)/W, W
# simplified, thin shell: C = 4pi(D - s/2) for s<D, 2pi D^2/s for s>D (Abel kernel s/sqrt(s^2-l^2))
def simple(ell, d):
    lo = max(ell, d)
    u = np.linspace(0, 1, 20001)
    out = []
    for f in (lambda q: 1.0 + 0*q, D, lambda q: D(q)**2):
        tot = 0.0
        if ell < d:   # same-wall slab part, s in [ell, d]: s = sqrt(ell^2 + t^2), t in [0, sqrt(d^2-ell^2)]
            t = np.geomspace(1e-12, np.sqrt(d*d-ell*ell), 20001); q = np.sqrt(ell**2+t**2)
            tot += np.trapezoid(4*np.pi*(d-q/2)*f(q), t)
        umax, umin = np.arccosh(2/ell), np.arccosh(lo/ell)
        uu = np.linspace(umin, umax, 20001); q = ell*np.cosh(uu)
        tot += 2*np.pi*d**2*np.trapezoid(f(q), uu)
        out.append(tot)
    W, N1, N2 = out
    return N1/W, N2/W, W
def plateau_closed(d):
    L = 3 + 2*np.log(2/d)
    D1 = (2*J - 27/20*A*d**(2/3))/L; D2 = (2*J2 - 27/70*A**2*d**(4/3))/L
    return D1, D2
print("\n Delta   ell   | <D> exact  simple | K_t exact simple | plateau closed form <D>, K_t | K_b(0) thin = (3+2ln(2/D))/2")
for d in [0.005, 0.02, 0.09, 0.3, 1.0]:
    for ell in [1e-4, 0.01, 0.03, 0.1, 0.3]:
        e1, e2, _ = exact(ell, d); s1, s2, _ = simple(ell, d)
        line = f"{d:6} {ell:6} | {e1:.3f}  {s1:.3f} | {3*e2/e1**2:.3f} {3*s2/s1**2:.3f}"
        if ell == 1e-4:
            p1, p2 = plateau_closed(d)
            line += f" | {p1:.3f} {3*p2/p1**2:.3f} | {(3+2*np.log(2/d))/2:.3f}"
        print(line)
