"""Tract profiles: scalar values along a bundle's core, from its MCAT density map.

For one bundle and its core (``mcat.core``, n nodes in world mm):

1. Support.  The voxels ``bundle_core`` builds its graph on: density above ``floor`` x peak, in
   the connected component that holds the peak.  Disconnected islands are left out.
2. Assignment.  Every support voxel goes to the core node nearest to it along the bundle, in the
   metric of MCAT's own cross-sections (``bundle_core`` step 3): -log(density / peak) per mm on the
   support's 26-neighbour graph, plus its 1e-9-per-step tie-break -- one multi-source Dijkstra.
   Step lengths are |A[:3, :3] @ offset| mm (the same as bundle_core's on isotropic grids, correct
   on anisotropic or sheared ones).  A node enters the graph through the support voxels at the
   eight corners of the voxel cell it lies in, at (distance in mm) x (that voxel's cost).  Nodes are
   never moved, merged or renumbered: a node whose cell has no support corner (detached), one that
   coincides with another (an exact tie goes to whichever SciPy's Dijkstra settles first), or one
   closer to its neighbours than a voxel, can own no voxels -- its profile is NaN.
3. Weights (issue 6).  For node n's voxels, with density d_i at voxel centre x_i (mm): the
   density-weighted mean mu and population covariance S, the Mahalanobis distance
   D_i = sqrt((x_i - mu)' pinvh(S) (x_i - mu)) (SciPy's default eigenvalue cutoff, so thin rank-1
   and rank-2 cross-sections are kept), and w_i = d_i / D_i.  As in pyAFQ's ``gaussian_weights``, a
   voxel exactly at mu (an infinite inverse distance) gets zero weight.  A node whose weights are
   all zero (one voxel: no covariance) is weighted by density alone.
4. Profile.  P_n = sum w_i s_i / sum w_i over node n's voxels with a finite scalar s_i; NaN if
   there are none.  Scalars are sampled at the voxel centres by linear interpolation on their own
   grid: a sample outside that grid, or mixing in a non-finite voxel, is missing (not zero).
"""
from __future__ import annotations

import csv
import json
import os.path as op
import time

import nibabel as nib
import numpy as np

from .core import _graph, _support, bundle_files

ZERO_DISTANCE = "pyAFQ: w = d / D, and 0 where D == 0; a node with no positive weight is weighted by d"
CELL = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def _assign(dn, keep, affine, xyz):
    """(ijk (m, 3) support voxels, owner (m,) node index or -1) -- see step 2."""
    import scipy.sparse as sps
    from scipy.sparse.csgraph import dijkstra

    A = affine[:3, :3]
    cost = np.where(keep, -np.log(np.where(keep, dn, 1.0)), 0.0)   # as bundle_core's cost1
    G, idx, lin = _graph(cost, keep, None, A)
    m, n = len(idx), len(xyz)
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), xyz)
    corner = np.floor(ijk).astype(np.int64)[:, None] + CELL            # (n, 8, 3)
    inside = np.all((corner >= 0) & (corner < keep.shape), -1)
    vox = np.full(inside.shape, -1, np.int64)
    vox[inside] = lin[tuple(corner[inside].T)]
    on = vox >= 0
    mm = np.linalg.norm((corner - ijk[:, None]) @ A.T, axis=-1)
    c = np.zeros(vox.shape)
    c[on] = cost[tuple(idx[vox[on]].T)]
    E = sps.csr_matrix((mm[on] * c[on] + 1e-9, (m + np.nonzero(on)[0], vox[on])), shape=(m + n, m + n))
    G = sps.block_diag((G, sps.csr_matrix((n, n)))).tocsr() + E
    dist, _, src = dijkstra(G, indices=m + np.arange(n), min_only=True, return_predecessors=True)
    return idx, np.where(np.isfinite(dist[:m]), src[:m] - m, -1)


def _mahalanobis(x, d):
    """Mahalanobis distance of each x_i from the d-weighted mean, in the d-weighted population
    covariance (pseudo-inverted)."""
    from scipy.linalg import pinvh

    r = x - d @ x / d.sum()
    S = (d[:, None] * r).T @ r / d.sum()
    D2 = np.einsum("ij,jk,ik->i", r, pinvh(0.5 * (S + S.T)), r)
    return np.sqrt(np.clip(D2, 0, None))


def _weights(x, d, owner, n):
    """Unnormalised profile weight of every voxel (step 3), and the nodes weighted by density."""
    w = np.zeros(len(x))
    fallback = []
    for k in range(n):
        i = np.flatnonzero(owner == k)
        if not len(i):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            wk = d[i] / _mahalanobis(x[i], d[i])
        wk[~np.isfinite(wk)] = 0
        if not wk.sum() > 0:
            wk, _ = d[i], fallback.append(k)
        w[i] = wk
    return w, fallback


def _sample(data, affine, xyz):
    """data at world points xyz by linear interpolation; NaN outside the grid or where a voxel
    with nonzero interpolation weight is not finite."""
    from scipy.ndimage import map_coordinates

    ijk = nib.affines.apply_affine(np.linalg.inv(affine), xyz).T
    tol = 1e-6
    inside = np.all((ijk >= -tol) & (ijk <= np.array(data.shape[:3])[:, None] - 1 + tol), 0)
    fin = np.isfinite(data)
    v = map_coordinates(np.where(fin, data, 0.0), ijk, order=1, mode="nearest")
    ok = map_coordinates(fin.astype(float), ijk, order=1, mode="nearest") > 1 - tol
    return np.where(inside & ok, v, np.nan)


def tract_profiles(density, affine, core, scalars, n_nodes=100, floor=1e-6):
    """Profiles of one bundle: ({scalar: (n,) values}, info).

    density   3-D bundle density map (finite, >= 0), affine its voxel-to-world matrix
    core      (n, 3) core nodes in world mm (``Core.xyz``), or None for an empty bundle
    scalars   {name: nibabel image or (data, affine)}
    n_nodes   the node count when there is no core; otherwise the core's own count is kept
    """
    d = np.asarray(density, float)
    affine = np.asarray(affine, float)
    if d.ndim != 3 or affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("density must be 3-D with a finite 4x4 affine")
    if not np.all(np.isfinite(d)) or d.min() < 0:
        raise ValueError("density must be finite and non-negative")
    scalars = {k: (np.asarray(v.dataobj, float), v.affine) if hasattr(v, "affine")
               else (np.asarray(v[0], float), np.asarray(v[1], float)) for k, v in scalars.items()}
    info = dict(floor=floor)
    if d.max() <= 0:
        return {k: np.full(n_nodes, np.nan) for k in scalars}, dict(info, status="empty")
    core = np.asarray(core, float)
    if core.ndim != 2 or core.shape[1] != 3 or not np.all(np.isfinite(core)):
        raise ValueError("core must be (n_nodes, 3) finite world coordinates")
    n = len(core)

    t = [time.perf_counter()]
    dn = d / d.max()
    ijk, owner = _assign(dn, _support(dn, floor), affine, core)
    t.append(time.perf_counter())
    x = nib.affines.apply_affine(affine, ijk)
    w, fallback = _weights(x, d[tuple(ijk.T)], owner, n)
    t.append(time.perf_counter())
    out = {}
    for name, (data, aff) in scalars.items():
        s = _sample(data, aff, x)
        ok = np.isfinite(s) & (w > 0)
        num = np.bincount(owner[ok], w[ok] * s[ok], n)
        den = np.bincount(owner[ok], w[ok], n)
        out[name] = np.divide(num, den, out=np.full(n, np.nan), where=den > 0)
    t.append(time.perf_counter())
    count = np.bincount(owner[owner >= 0], minlength=n)
    info.update(status="ok", n_nodes=n, n_support=len(owner), n_unassigned=int((owner < 0).sum()),
                empty_nodes=np.flatnonzero(count == 0).tolist(), density_weighted_nodes=fallback,
                seconds=dict(zip(("assignment", "weights", "sampling"), np.round(np.diff(t), 3).tolist())))
    return out, info


def write_profiles(folder, scalars, out=None, n_nodes=100, floor=1e-6):
    """Profiles of every MCAT bundle under ``folder`` from its saved core (``write_cores``), as
    pyAFQ-style ``<stem>_desc-profiles_tractography.csv`` (tractID, nodeID, one column per scalar;
    missing values empty) and a ``.json`` with the method and each bundle's status.

    scalars   {column name: path of a scalar image}, e.g. pyAFQ's exported ``dti_fa``
    """
    files = bundle_files(str(folder))
    if not files:
        raise ValueError(f"no MCAT density maps under {folder}")
    if out is None:
        out = next(iter(files.values()))["prefix"].split("_bundle-")[0] + "_desc-profiles_tractography.csv"
    images = {k: nib.load(p) for k, p in scalars.items()}
    images = {k: (np.asarray(im.dataobj, float), im.affine) for k, im in images.items()}
    rows, meta = [], {}
    for key, f in files.items():
        summary = f["prefix"] + "_desc-summary.json"
        name = json.load(open(summary)).get("bundle", key) if op.exists(summary) else key
        img = nib.load(f["density"])
        d = np.asarray(img.dataobj, float)
        core_tsv = f["prefix"] + "_desc-core_path.tsv"
        core = np.loadtxt(core_tsv, skiprows=1, usecols=(1, 2, 3), ndmin=2) if op.exists(core_tsv) else None
        if core is None and d.max() > 0:
            prof, info = {k: np.full(n_nodes, np.nan) for k in images}, dict(status="no core")
        else:
            prof, info = tract_profiles(d, img.affine, core, images, n_nodes, floor)
        meta[name] = dict(info, density=f["density"], core=core_tsv if core is not None else None)
        n = len(next(iter(prof.values()))) if prof else (len(core) if core is not None else n_nodes)
        for i in range(n):
            rows.append([name, i] + ["" if np.isnan(prof[k][i]) else repr(float(prof[k][i])) for k in images])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tractID", "nodeID", *images])
        w.writerows(rows)
    with open(out[: -len(".csv")] + ".json", "w") as fh:
        json.dump(dict(method="density-weighted inverse Mahalanobis distance over geodesic "
                              "cross-sections of the MCAT core (mcat.profiles)",
                       assignment="nearest core node in bundle_core's cross-section metric: "
                                  "-log(density / peak) per mm (affine step lengths)",
                       zero_distance=ZERO_DISTANCE, pseudoinverse="scipy.linalg.pinvh default cutoff",
                       support=f"density > {floor} x peak, component holding the peak",
                       node_order="nodeID 0 is the core's first node (its start end: see its json 'ends')",
                       scalars={k: str(p) for k, p in scalars.items()}, bundles=meta), fh, indent=1)
    return out
