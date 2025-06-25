import numpy as np

def rv2coe_vec(r, v, mu):
    """
    Vectorized Vallado routine: given arrays r,v of shape (N,3), and mu,
    returns arrays (p,a,ecc,incl,omega,argp,nu,m,arglat,truelon,lonper)
    each of shape (N,).
    """

    small    = 1e-8
    infinite = 9.999999e5
    undefined= np.nan
    twopi    = 2.0 * np.pi
    halfpi   = 0.5  * np.pi

    # ensure r,v are (N,3)
    r = np.atleast_2d(r)
    v = np.atleast_2d(v)
    N = r.shape[0]

    # preallocate outputs
    p       = np.full(N, undefined)
    a       = np.full(N, undefined)
    ecc     = np.full(N, undefined)
    incl    = np.full(N, undefined)
    omega   = np.full(N, undefined)
    argp    = np.full(N, undefined)
    nu      = np.full(N, undefined)
    m       = np.full(N, undefined)
    arglat  = np.full(N, undefined)
    truelon = np.full(N, undefined)
    lonper  = np.full(N, undefined)

    # magnitudes
    magr = np.linalg.norm(r, axis=1)
    magv = np.linalg.norm(v, axis=1)

    # --- find h, n and e vectors ---
    # hbar: (N,3)
    hbar = np.cross(r, v)
    magh = np.linalg.norm(hbar, axis=1)

    # only keep indices where h > small
    idx_small = magh > small
    idx       = np.nonzero(idx_small)[0]

    # nbar: line-of-nodes, only for idx
    nbar = np.zeros((idx.size,3))
    hbar_s = hbar[idx]
    nbar[:,0] = -hbar_s[:,1]
    nbar[:,1] =  hbar_s[:,0]
    magn = np.linalg.norm(nbar, axis=1)

    # extract checked r,v,h magnitudes
    magr_c = magr[idx]
    magv_c = magv[idx]
    magh_c = magh[idx]
    r_c    = r[idx]
    v_c    = v[idx]

    # c1 and r·v
    c1    = magv_c**2 - mu/magr_c
    rdotv = np.einsum('ij,ij->i', r_c, v_c)

    # ebar (N_c,3)
    ebar = np.column_stack([
        (c1 * r_c[:,0] - rdotv * v_c[:,0]),
        (c1 * r_c[:,1] - rdotv * v_c[:,1]),
        (c1 * r_c[:,2] - rdotv * v_c[:,2])
    ]) / mu
    ecci = np.linalg.norm(ebar, axis=1)
    ecc[idx] = ecci

    # --- find a, p ---
    sme      = 0.5 * magv_c**2 - mu/magr_c
    check_sme= np.abs(sme) > small
    a[idx[check_sme]]  = -mu / (2.0 * sme[check_sme])
    a[idx[~check_sme]] = infinite
    p[idx]            = magh_c**2 / mu

    # --- inclination ---
    hk    = hbar_s[:,2] / magh_c
    incli = np.arccos(hk)
    incl[idx] = incli

    # --- orbit type flags ---
    typeorbit = np.zeros(idx.size, dtype=int)
    check_ecc  = ecci < small
    check_ne  = ~check_ecc
    idx_ecc   = np.nonzero(check_ecc)[0]
    idx_ne    = np.nonzero(check_ne)[0]

    # circular equatorial vs circular inclined
    circ_eq = check_ecc & ((incli < small) | (np.abs(incli - np.pi) < small))
    typeorbit[idx_ecc[circ_eq[idx_ecc]]] = 2
    circ_in = check_ecc & ~circ_eq
    typeorbit[idx_ecc[circ_in[idx_ecc]]] = 3
    # elliptical equatorial
    ell_eq = check_ne & (((incli) < small) | (np.abs(incli - np.pi) < small))
    typeorbit[idx_ne[ell_eq[idx_ne]]] = 1
    # remaining (0) are elliptical inclined

    # --- RAAN (omega) ---
    ok_n = magn > small
    temp = nbar[ok_n,0] / magn[ok_n]
    temp = np.clip(temp, -1.0, 1.0)
    omega_temp = np.arccos(temp)
    neg = nbar[ok_n,1] < 0
    omega_temp[neg] = twopi - omega_temp[neg]
    omega[idx[ok_n]] = omega_temp

    # --- argument of perigee (argp) ---
    ei = typeorbit == 0
    argp_temp = angl_vec(nbar[ei], ebar[ei])
    negz = ebar[ei,2] < 0
    argp_temp[negz] = twopi - argp_temp[negz]
    argp[idx[ei]] = argp_temp

    # --- true anomaly (nu) ---
    ce = typeorbit < 1.5
    nu_temp = angl_vec(ebar[ce], r_c[ce])
    rv_neg = rdotv[ce] < 0
    nu_temp[rv_neg] = twopi - nu_temp[rv_neg]
    nu[idx[ce]] = nu_temp

    # --- argument of latitude (arglat) for circ. inclined ---
    ci = typeorbit == 3
    arglat_temp = angl_vec(nbar[ci], r_c[ci])
    negz = r_c[ci,2] < 0
    arglat_temp[negz] = twopi - arglat_temp[negz]
    arglat[idx[ci]] = arglat_temp
    m[idx[ci]] = arglat_temp

    # --- longitude of periapsis (lonper) for ell. equatorial ---
    ee = (typeorbit[idx_ne] == 1)
    ee_idx = idx_ne[ee]
    temp = ebar[ee,0] / ecci[ee]
    temp = np.clip(temp, -1.0, 1.0)
    lonper_temp = np.arccos(temp)
    negy = ebar[ee,1] < 0
    lonper_temp[negy] = twopi - lonper_temp[negy]
    highi = incli[ee] > halfpi
    lonper_temp[highi] = twopi - lonper_temp[highi]
    lonper[idx[ee_idx]] = lonper_temp

    # --- true longitude (truelon) for circ. equatorial ---
    ce2 = (typeorbit == 2)
    ok2 = magn > small
    mask2 = ce2 & ok2
    temp = r_c[mask2,0] / magr_c[mask2]
    temp = np.clip(temp, -1.0, 1.0)
    truelon_temp = np.arccos(temp)
    negy = r_c[mask2,1] < 0
    truelon_temp[negy] = twopi - truelon_temp[negy]
    highi = incli[mask2] > halfpi
    truelon_temp[highi] = twopi - truelon_temp[highi]
    truelon[idx[mask2]] = truelon_temp
    m[idx[mask2]] = truelon_temp

    # --- mean anomaly via Newton’s method ---
    tol = 1e-8
    mask_elliptical = (ecc > tol) & (ecc < 1 - tol)

    # now compute mean anomaly only for those
    e0_new, m_new = newtonnu_vec(ecc[mask_elliptical], nu[mask_elliptical])
    m[mask_elliptical] = m_new.ravel()

    # everything else stays NaN
    return p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper

import numpy as np

def angl_vec(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Calculate the angle between two sets of vectors row-wise.

    Parameters
    ----------
    vec1 : (N,3) array
        First set of vectors.
    vec2 : (N,3) array
        Second set of vectors.

    Returns
    -------
    theta : (N,) array
        Angle between each pair in radians, or 999999.1 if undefined.
    """
    undefined = 999999.1

    # magnitudes of each vector
    magv1 = np.sqrt(np.sum(vec1**2, axis=1))
    magv2 = np.sqrt(np.sum(vec2**2, axis=1))

    theta = np.zeros_like(magv1)
    mag_prod = magv1 * magv2

    # only compute where both magnitudes are non-zero
    check_mag = mag_prod > 1e-16

    # dot product row-wise, divided by product of magnitudes
    temp = np.sum(vec1[check_mag] * vec2[check_mag], axis=1) / mag_prod[check_mag]

    # clamp any tiny numerical overshoot to ±1
    temp[temp > 1.0] = 1.0
    temp[temp < -1.0] = -1.0

    # compute angles
    theta[check_mag] = np.arccos(temp)
    theta[~check_mag] = undefined

    return theta

def newtonnu_vec(ecc, nu):
    """
    Solve Kepler's equation when the true anomaly is known.

    Parameters
    ----------
    ecc : array_like
        Eccentricities (can be vector).
    nu : array_like
        True anomalies in radians (same shape as ecc).

    Returns
    -------
    e0 : ndarray
        Eccentric (or hyperbolic/parabolic) anomaly.
    m : ndarray
        Mean anomalies.

    Based on Vallado's newtonnu_vec (2002, 2007).
    """
    ecc = np.asarray(ecc, dtype=float)
    nu  = np.asarray(nu,  dtype=float)

    # constants and error codes
    e0_error = 999999.9
    m_error  = 999999.9
    small    = 1e-8

    # initialize outputs
    m  = np.zeros_like(nu)
    e0 = np.zeros_like(nu)

    # identify indices
    check_ecc    = np.abs(ecc) > small
    idx_ecc      = np.where(check_ecc)[0]
    idx_notecc   = np.where(~check_ecc)[0]

    # circular case: ecc ~ 0
    m[~check_ecc]  = nu[~check_ecc]
    e0[~check_ecc] = nu[~check_ecc]

    # elliptical case: 0 < ecc < 1
    idx_ell = idx_ecc[ecc[idx_ecc] < (1.0 - small)]
    if idx_ell.size > 0:
        ecci = ecc[idx_ell]
        nui  = nu[idx_ell]
        c_nui = np.cos(nui)
        one_p = 1.0 + ecci * c_nui
        sine  = (np.sqrt(1.0 - ecci**2) * np.sin(nui)) / one_p
        cose  = (ecci + c_nui) / one_p
        e0_temp = np.arctan2(sine, cose)
        e0[idx_ell] = e0_temp
        m[idx_ell]  = e0_temp - ecci * np.sin(e0_temp)

    # hyperbolic case: ecc > 1
    idx_hyp_all = idx_notecc[ecc[idx_notecc] > (1.0 + small)]
    # filter for valid hyperbolic anomaly region
    valid_h = []
    for i in idx_hyp_all:
        if (ecc[i] > 1.0) and (abs(nu[i]) + 1e-5 < np.pi - np.arccos(1.0 / ecc[i])):
            valid_h.append(i)
    idx_hyp = np.array(valid_h, dtype=int)
    if idx_hyp.size > 0:
        ecch = ecc[idx_hyp]
        nuh = nu[idx_hyp]
        sine = (np.sqrt(ecch**2 - 1.0) * np.sin(nuh)) / (1.0 + ecch * np.cos(nuh))
        e0_temp = np.arcsinh(sine)
        e0[idx_hyp] = e0_temp
        m[idx_hyp]  = ecch * np.sinh(e0_temp) - e0_temp
    # mark invalid hyperbolic
    bad_h = np.setdiff1d(idx_notecc, idx_hyp)
    if bad_h.size > 0:
        e0[bad_h] = e0_error
        m[bad_h]  = m_error

    # parabolic case: ecc==1 (within tolerance)
    idx_par = idx_notecc[np.abs(ecc[idx_notecc] - 1.0) <= small]
    if idx_par.size > 0:
        # restrict to |nu| < 168 deg
        thresh = 168.0 * np.pi/180.0
        ok_par = idx_par[np.abs(nu[idx_par]) < thresh]
        e0_temp = np.tan(nu[ok_par] * 0.5)
        e0[ok_par] = e0_temp
        m[ok_par]  = e0_temp + e0_temp**3 / 3.0
        bad_par = np.setdiff1d(idx_par, ok_par)
        if bad_par.size > 0:
            e0[bad_par] = e0_error
            m[bad_par]  = m_error

    # wrap e0 and m into [0,2pi) for ecc<1
    mask = ecc < 1.0
    if np.any(mask):
        m_temp = np.mod(m[mask], 2.0*np.pi)
        m_temp[m_temp < 0] = m[mask][m_temp < 0] + 2.0*np.pi
        m[mask] = m_temp
        e0[mask] = np.mod(e0[mask], 2.0*np.pi)

    e0 = np.array(e0).ravel()
    m  = np.array(m).ravel()
    return e0, m