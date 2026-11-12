"""Bundle cores: the cross-section ridge path of a MPAT density map.

A core is a single curve through the bundle, from one end to the other, that stays on the
density ridge -- the most visited voxels of each cross-section -- including through bends.

Method (parameter-free apart from where the ends are; two shortest-path searches):

1. Anchors.  Each end of the bundle is anchored at the white-matter voxel nearest the
   centroid of the walkers' endpoints at that end, weighted by the endpoint density
   (``endpoints_map``) -- or, with ``anchor="mode"``, nearest the endpoint density's peak
   (smoothed over a voxel), where most walkers end, which a broad or curved ending ROI (an
   eye, a gyrus) pulls the centroid away from.  Which endpoints belong to which end, and so
   which end is node 0, follows pyAFQ's orientation of streamlines: the start ROI is the first
   end and the end ROI the last (the other endpoints are the other end when the bundle has
   only one of them); otherwise the endpoints nearer the first include ROI are the first end
   and those nearer the last the last; otherwise the two hemispheres; otherwise two clusters
   of the endpoints.
2. A first most-probable path: Dijkstra on the 26-neighbour voxel graph with cost
   -log(density / peak) per mm.  It finds the right corridor but cuts corners, because
   every step below the peak costs something.
3. Cross-sections.  Every voxel joins the cross-section of the first path's voxel nearest
   to it (nearest in the same density metric, so cross-sections follow the bundle).
4. The ridge path: Dijkstra again with cost -log(density / max density of its own
   cross-section).  That cost is zero on each cross-section's maximum, so the path is not
   penalised for length while it follows the ridge, and bends are no longer cut.
5. Gaussian smoothing (``smooth`` voxels along the path) and resampling to ``n_nodes``
   equally spaced nodes, oriented from the start end to the other end.
"""
from __future__ import annotations

import glob
import json
import logging
import os.path as op
import re
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np

log = logging.getLogger("mpat")

OFFS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                 if (i, j, k) != (0, 0, 0)])


@dataclass
class Core:
    xyz: np.ndarray            # (n_nodes, 3) world mm, from the start end to the other
    density: np.ndarray        # (n_nodes,) density / bundle peak at each node (trilinear)
    length_mm: float
    ends: str                  # how the two ends were defined
    info: dict = field(default_factory=dict)
    first_xyz: np.ndarray = None  # the plain most-probable path (step 2), same smoothing


def _resample(xyz, n):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    if s[-1] == 0:
        return np.repeat(xyz[:1], n, 0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, xyz[:, k]) for k in range(3)], 1)


def _support(dn, floor):
    """Voxels with density above floor x peak, in the connected component holding the peak
    (moved unchanged out of bundle_core, which mpat.profiles shares it with)."""
    from scipy.ndimage import label

    lab, _ = label(dn > floor, np.ones((3, 3, 3)))
    return lab == lab[np.unravel_index(np.argmax(dn), dn.shape)]


def _graph(cost_vox, keep, vox_mm):
    import scipy.sparse as sps

    idx = np.argwhere(keep)
    n = len(idx)
    lin = -np.ones(keep.shape, np.int64)
    lin[tuple(idx.T)] = np.arange(n)
    c = cost_vox[tuple(idx.T)]
    rows, cols, w = [], [], []
    for o in OFFS:
        nb = idx + o
        ok = np.all((nb >= 0) & (nb < keep.shape), 1)
        j = np.full(n, -1, np.int64)
        j[ok] = lin[tuple(nb[ok].T)]
        m = j >= 0
        rows.append(np.flatnonzero(m))
        cols.append(j[m])
        # +1e-9: a tiny length tie-break, so zero-cost ridges still give a shortest path
        w.append(np.linalg.norm(o) * vox_mm * 0.5 * (c[m] + c[j[m]]) + 1e-9)
    G = sps.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n, n))
    return G, idx, lin


def _path(G, src, dst):
    from scipy.sparse.csgraph import dijkstra

    dist, pred, _ = dijkstra(G, indices=src, min_only=True, return_predecessors=True)
    t = dst[np.argmin(dist[dst])]
    if not np.isfinite(dist[t]):
        raise ValueError("the two ends are not connected through the bundle")
    p = [t]
    while pred[p[-1]] >= 0:
        p.append(pred[p[-1]])
    return np.array(p[::-1])


def _anchor(ep, roi, body, how="centroid"):
    """Body voxel nearest the endpoint-weighted centroid (``how="centroid"``) or the smoothed
    endpoint-density peak (``"mode"``) of the endpoints in ``roi``."""
    w = np.where(roi, ep, 0.0)
    if w.sum() <= 0:
        w = roi.astype(float)
    if how == "mode":
        from scipy.ndimage import gaussian_filter

        sm = np.where(roi, gaussian_filter(w, 1.0), -np.inf)
        c = np.array(np.unravel_index(np.argmax(sm), sm.shape), float)
    elif how == "centroid":
        ijk = np.argwhere(w > 0)
        c = (ijk * w[tuple(ijk.T)][:, None]).sum(0) / w[tuple(ijk.T)].sum()
    else:
        raise ValueError(f"anchor must be 'centroid' or 'mode', not {how!r}")
    cand = np.argwhere(body)
    return cand[np.argmin(((cand - c) ** 2).sum(1))]


def split_ends(ep, affine, start=None, end=None, include=None):
    """Two boolean masks of endpoint voxels, first end then last end of the bundle, and a
    description.  ``include``: the include ROIs labelled 1, 2, ... in pyAFQ's order (MPAT
    ``include_mask``)."""
    from scipy.ndimage import distance_transform_edt

    has = ep > 0
    if start is not None and end is not None and (has & start).any() and (has & end).any():
        return has & start, has & end, "start and end ROIs"
    if start is not None and (has & start).any() and (has & ~start).any():
        return has & start, has & ~start, "start ROI vs all other endpoints"
    if end is not None and (has & end).any() and (has & ~end).any():
        return has & ~end, has & end, "all other endpoints vs end ROI"
    if include is not None and include.max() > 1:
        vox = np.sqrt((affine[:3, :3] ** 2).sum(0))
        first = has & (distance_transform_edt(include != 1, vox) <
                       distance_transform_edt(include != include.max(), vox))
        if first.any() and (has & ~first).any():
            return first, has & ~first, "first vs last include ROI"
    ijk = np.argwhere(has)
    if len(ijk) < 2:
        raise ValueError(f"{len(ijk)} endpoint voxel(s): no two ends to split")
    w = ep[tuple(ijk.T)]
    xyz = nib.affines.apply_affine(affine, ijk)
    left = xyz[:, 0] < 0
    if min(w[left].sum(), w[~left].sum()) > 0.1 * w.sum():
        a = np.zeros(ep.shape, bool)
        a[tuple(ijk[left].T)] = True
        return a, has & ~a, "left vs right hemisphere"
    # two weighted clusters, seeded at the extremes of the principal axis
    c = (xyz * w[:, None]).sum(0) / w.sum()
    u = np.linalg.svd((xyz - c) * np.sqrt(w)[:, None], full_matrices=False)[2][0]
    proj = (xyz - c) @ u
    cen = np.stack([xyz[np.argmin(proj)], xyz[np.argmax(proj)]])
    for _ in range(20):
        lab = ((xyz[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
        cen = np.stack([(xyz[lab == k] * w[lab == k, None]).sum(0) / max(w[lab == k].sum(), 1e-300)
                        for k in (0, 1)])
    a = np.zeros(ep.shape, bool)
    a[tuple(ijk[lab == 0].T)] = True
    return a, has & ~a, "two endpoint clusters"


def bundle_core(density, affine, endpoints, start=None, end=None, n_nodes=100, floor=1e-6,
                body=1e-2, smooth=1.5, anchor="centroid", include=None):
    """Cross-section ridge path of one bundle (see module docstring).

    density, endpoints  3-D arrays on the same grid (MPAT ``density_map``, ``endpoints_map``)
    start, end          optional boolean endpoint ROIs (MPAT ``start_mask``, ``end_mask``)
    include             optional include ROIs labelled 1, 2, ... in pyAFQ's order (MPAT ``include_mask``)
    floor               voxels with density below floor x peak are left out of the graph
    body                anchors are placed on voxels with density >= body x peak
    anchor              "centroid" (default) or "mode": where in each end's endpoints to anchor
    """
    from scipy.ndimage import gaussian_filter1d, map_coordinates
    from scipy.sparse.csgraph import dijkstra

    d = np.clip(np.nan_to_num(np.asarray(density, float)), 0, None)
    ep = np.clip(np.nan_to_num(np.asarray(endpoints, float)), 0, None)
    if d.max() <= 0:
        raise ValueError("empty density map")
    dn = d / d.max()
    vox_mm = float(np.abs(np.linalg.det(affine[:3, :3])) ** (1 / 3))
    S_roi, E_roi, how = split_ends(ep, affine, start, end, include)
    keep = _support(dn, floor)
    bod = keep & (dn >= body)
    s_vox, e_vox = _anchor(ep, S_roi, bod, anchor), _anchor(ep, E_roi, bod, anchor)

    cost1 = np.where(keep, -np.log(np.where(keep, dn, 1.0)), 0.0)
    G, idx, lin = _graph(cost1, keep, vox_mm)
    src, dst = np.array([lin[tuple(s_vox)]]), np.array([lin[tuple(e_vox)]])
    p0 = _path(G, src, dst)

    dist, _, owner = dijkstra(G, indices=p0, min_only=True, return_predecessors=True)
    dv = d[tuple(idx.T)]
    ok = np.isfinite(dist)
    mx = np.zeros(len(idx))
    np.maximum.at(mx, owner[ok], dv[ok])
    rel = np.zeros(len(idx))
    rel[ok] = dv[ok] / mx[owner[ok]]
    cost2 = np.zeros(d.shape)
    cost2[tuple(idx.T)] = np.where(ok, -np.log(np.maximum(rel, 1e-300)), 50.0)
    G2, idx2, lin2 = _graph(cost2, keep, vox_mm)
    p = _path(G2, src, dst)

    def curve(vox):
        c = nib.affines.apply_affine(affine, vox.astype(float))
        if smooth and len(c) > 2:
            c = gaussian_filter1d(c, smooth, axis=0, mode="nearest")
        return _resample(c, n_nodes)

    xyz, first = curve(idx2[p]), curve(idx[p0])
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), xyz)
    dens = map_coordinates(dn, ijk.T, order=1, mode="nearest")
    length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    L0 = float(np.linalg.norm(np.diff(first, axis=0), axis=1).sum())
    info = dict(anchor=anchor, n_graph_voxels=int(len(idx)), first_path_length_mm=round(L0, 1),
                start_anchor_vox=s_vox.tolist(), end_anchor_vox=e_vox.tolist(),
                min_node_density=float(dens.min()), median_node_density=float(np.median(dens)))
    return Core(xyz, dens, length, how, info, first)


def write_core(core, prefix, extra=None):
    """``<prefix>_desc-core_path.tsv`` (nodes) and ``<prefix>_desc-core.tck`` (one streamline)."""
    tsv = f"{prefix}_desc-core_path.tsv"
    with open(tsv, "w") as fh:
        fh.write("node\tx_mm\ty_mm\tz_mm\tdensity_rel_peak\n")
        for i, (p, v) in enumerate(zip(core.xyz, core.density)):
            fh.write(f"{i}\t{p[0]:.3f}\t{p[1]:.3f}\t{p[2]:.3f}\t{v:.6g}\n")
    tck = f"{prefix}_desc-core.tck"
    nib.streamlines.save(nib.streamlines.Tractogram([core.xyz.astype(np.float32)],
                                                    affine_to_rasmm=np.eye(4)), tck)
    with open(f"{prefix}_desc-core.json", "w") as fh:
        json.dump(dict(method="cross-section ridge path (mpat.core)", ends=core.ends,
                       n_nodes=int(len(core.xyz)), length_mm=round(core.length_mm, 1),
                       **core.info, **(extra or {})), fh, indent=1)
    return tsv, tck


def bundle_files(folder):
    """{bundle: {prefix, density, endpoints, start, end, include, exclude, odf}} for the MPAT outputs under
    ``folder`` (None where a file is missing)."""
    out = {}
    for f in sorted(glob.glob(op.join(folder, "**", "*_desc-density_map.nii.gz"), recursive=True)):
        m = re.search(r"_bundle-(.+?)_desc-density_map", op.basename(f))
        if not m:
            continue
        pre = f[: -len("_desc-density_map.nii.gz")]
        get = lambda k: (pre + f"_desc-{k}.nii.gz") if op.exists(pre + f"_desc-{k}.nii.gz") else None  # noqa: E731
        out[m.group(1)] = dict(prefix=pre, density=f, endpoints=get("endpoints_map"),
                               start=get("start_mask"), end=get("end_mask"), include=get("include_mask"),
                               exclude=get("exclude_mask"), odf=get("visitodf_dwimap"))
    return out


def write_cores(folder, n_nodes=100, anchor="centroid"):
    """Compute and save the core of every bundle under ``folder`` (MPAT outputs)."""
    for name, f in bundle_files(folder).items():
        if f["endpoints"] is None:
            continue
        img = nib.load(f["density"])
        load = lambda k: (np.asarray(nib.load(f[k]).dataobj) > 0) if f[k] else None  # noqa: E731
        try:
            c = bundle_core(np.asarray(img.dataobj), img.affine, np.asarray(nib.load(f["endpoints"]).dataobj),
                            load("start"), load("end"), n_nodes, anchor=anchor,
                            include=np.asarray(nib.load(f["include"]).dataobj) if f["include"] else None)
        except ValueError as e:
            log.warning("%s: no core (%s)", name, e)
            continue
        write_core(c, f["prefix"])
        log.info("core %-24s %4.0f mm, ends: %s", name, c.length_mm, c.ends)
