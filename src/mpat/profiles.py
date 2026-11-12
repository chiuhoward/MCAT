"""Tract profiles: scalar values along a bundle's core, from its MPAT density map.

For one bundle and its core (``mpat.core``, n nodes in world mm):

1. Support.  The voxels ``bundle_core`` builds its graph on: density above ``floor`` x peak, in
   the connected component that holds the peak.  Disconnected islands are left out.
2. Assignment.  Every support voxel goes to the core node nearest to it along the bundle, in the
   metric of MPAT's own cross-sections (``bundle_core`` step 3): -log(density / peak) per mm on the
   support's 26-neighbour graph, plus its 1e-9-per-step tie-break -- one multi-source Dijkstra.
   Step lengths are bundle_core's too: |offset in voxels| x vox_mm, vox_mm = |det A|^(1/3) (exact
   on isotropic grids, a mean voxel size on anisotropic ones).  A node enters the graph through the
   support voxels at the eight corners of the voxel cell it lies in, at (its distance from each, in
   voxels x vox_mm) x (that voxel's cost).  Nodes are
   never moved, merged or renumbered: a node whose cell has no support corner (detached), one that
   coincides with another (an exact tie goes to whichever SciPy's Dijkstra settles first), or one
   closer to its neighbours than a voxel, can own no voxels -- its profile is NaN, and info lists it
   in ``empty_nodes``.  With 100 nodes on 2 mm voxels that is about half the nodes of a 100 mm
   bundle (pyAFQ has no such case: it interpolates scalars along resampled streamlines).
3. Weights (issue 6).  For node n's voxels, with density d_i at voxel centre x_i (mm): the
   density-weighted mean mu and population covariance S, the Mahalanobis distance
   D_i = sqrt((x_i - mu)' pinvh(S) (x_i - mu)) (SciPy's default eigenvalue cutoff, so thin rank-1
   and rank-2 cross-sections are kept), and w_i = d_i / D_i.  As in pyAFQ's ``gaussian_weights``, a
   voxel exactly at mu (an infinite inverse distance) gets zero weight.  A node whose weights are
   all zero (one voxel: no covariance) is weighted by density alone.
4. Profile.  P_n = sum w_i s_i / sum w_i over node n's voxels with a finite scalar s_i; NaN if
   there are none.  Scalars are put on the density grid with MPAT's ``afq._on_grid`` (linear, as
   pyAFQ's tracker).  Edge case: bundle voxels outside a scalar's grid read 0 there, not missing.
"""
from __future__ import annotations

import csv
import json
import os.path as op

import nibabel as nib
import numpy as np

from .afq import _on_grid
from .core import _graph, _support, bundle_files

CELL = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def _assign(dn, keep, affine, xyz):
    """(ijk (m, 3) support voxels, owner (m,) node index or -1) -- see step 2."""
    import scipy.sparse as sps
    from scipy.sparse.csgraph import dijkstra

    vox_mm = float(np.abs(np.linalg.det(affine[:3, :3])) ** (1 / 3))  # as bundle_core
    cost = np.where(keep, -np.log(np.where(keep, dn, 1.0)), 0.0)   # as bundle_core's cost1
    G, idx, lin = _graph(cost, keep, vox_mm)
    m, n = len(idx), len(xyz)
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), xyz)
    corner = np.floor(ijk).astype(np.int64)[:, None] + CELL            # (n, 8, 3)
    inside = np.all((corner >= 0) & (corner < keep.shape), -1)
    vox = np.full(inside.shape, -1, np.int64)
    vox[inside] = lin[tuple(corner[inside].T)]
    on = vox >= 0
    mm = np.linalg.norm(corner - ijk[:, None], axis=-1) * vox_mm
    c = np.zeros(vox.shape)
    c[on] = cost[tuple(idx[vox[on]].T)]
    E = sps.csr_matrix((mm[on] * c[on] + 1e-9, (m + np.nonzero(on)[0], vox[on])), shape=(m + n, m + n))
    G = sps.block_diag((G, sps.csr_matrix((n, n)))).tocsr() + E
    dist, _, src = dijkstra(G, indices=m + np.arange(n), min_only=True, return_predecessors=True)
    return idx, np.where(np.isfinite(dist[:m]), src[:m] - m, -1)


def _mahalanobis(x, d):
    """Mahalanobis distance of each x_i from the d-weighted mean, in the d-weighted population
    covariance (pseudo-inverted).  pyAFQ's ``gaussian_weights`` (population covariance, pinvh),
    with every sum weighted by d:
    https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/_fixes.py#L335-L350"""
    from scipy.linalg import pinvh

    r = x - d @ x / d.sum()
    S = (d[:, None] * r).T @ r / d.sum()
    D2 = np.einsum("ij,jk,ik->i", r, pinvh(0.5 * (S + S.T)), r)
    return np.sqrt(np.clip(D2, 0, None))


def _weights(x, d, owner, n):
    """Unnormalised profile weight of every voxel (step 3)."""
    w = np.zeros(len(x))
    for k in range(n):
        i = np.flatnonzero(owner == k)
        if not len(i):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            wk = d[i] / _mahalanobis(x[i], d[i])
        # inverse distances that are not finite get no weight, as in pyAFQ:
        # https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/_fixes.py#L355-L357
        wk[~np.isfinite(wk)] = 0
        w[i] = wk if wk.sum() > 0 else d[i]
    return w


def tract_profiles(density, affine, core, scalars, n_nodes=100, floor=1e-6):
    """Profiles of one bundle: ({scalar: (n,) values}, info).

    density   3-D bundle density map (NaN / negative read as 0, as bundle_core does)
    core      (n, 3) core nodes in world mm (``Core.xyz``), or None when there is none
    scalars   {name: nibabel image}
    n_nodes   the node count when there is no core; otherwise the core's own count is kept
    """
    d = np.clip(np.nan_to_num(np.asarray(density, float)), 0, None)   # as bundle_core
    if d.max() <= 0 or core is None:
        status = "empty" if d.max() <= 0 else "no core"
        return {k: np.full(n_nodes, np.nan) for k in scalars}, dict(status=status)
    n = len(core)
    dn = d / d.max()
    ijk, owner = _assign(dn, _support(dn, floor), affine, core)
    x = nib.affines.apply_affine(affine, ijk)
    w = _weights(x, dn[tuple(ijk.T)], owner, n)   # density / peak: same estimator
    out = {}
    ref = nib.Nifti1Image(d, affine)
    for name, img in scalars.items():
        s = _on_grid(img, ref)[tuple(ijk.T)].astype(float)
        ok = np.isfinite(s) & (w > 0)
        num = np.bincount(owner[ok], w[ok] * s[ok], n)
        den = np.bincount(owner[ok], w[ok], n)
        out[name] = np.divide(num, den, out=np.full(n, np.nan), where=den > 0)
    count = np.bincount(owner[owner >= 0], minlength=n)
    return out, dict(status="ok", n_support=len(owner), empty_nodes=np.flatnonzero(count == 0).tolist())


def write_profiles(folder, scalars):
    """Profiles of every MPAT bundle under ``folder`` from its saved core (``write_cores``), as
    pyAFQ-style ``<stem>_desc-profiles_tractography.csv`` (tractID, nodeID, one column per scalar;
    missing values empty) and a ``.json`` with each bundle's status.

    scalars   {column name: path of a scalar image}, e.g. pyAFQ's exported ``dti_fa``
    """
    files = bundle_files(str(folder))
    out = next(iter(files.values()))["prefix"].split("_bundle-")[0] + "_desc-profiles_tractography.csv"
    images = {k: nib.load(p) for k, p in scalars.items()}
    rows, meta = [], {}
    for key, f in files.items():
        summary = f["prefix"] + "_desc-summary.json"
        name = json.load(open(summary)).get("bundle", key) if op.exists(summary) else key
        img = nib.load(f["density"])
        core_tsv = f["prefix"] + "_desc-core_path.tsv"
        core = np.loadtxt(core_tsv, skiprows=1, usecols=(1, 2, 3), ndmin=2) if op.exists(core_tsv) else None
        prof, meta[name] = tract_profiles(img.dataobj, img.affine, core, images)
        for i, values in enumerate(zip(*prof.values())):
            rows.append([name, i] + ["" if np.isnan(v) else repr(float(v)) for v in values])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tractID", "nodeID", *images])
        w.writerows(rows)
    with open(out[: -len(".csv")] + ".json", "w") as fh:
        json.dump(dict(method="mpat.profiles (see its docstring); nodeID 0 is the core's start end",
                       scalars={k: str(p) for k, p in scalars.items()}, bundles=meta), fh, indent=1)
    return out
