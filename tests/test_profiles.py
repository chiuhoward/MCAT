"""Tract profiles from MCAT density maps (mcat.profiles; rokemlab/MCAT issue 6).

Expected values come from outside mcat.profiles: pyAFQ's own Mahalanobis weighting (and the
numbers pinned in pyAFQ's ``test_mahal_fix``), hand calculation, and geometry simple enough to
know each voxel's node without a shortest-path search.
"""
import csv
import json

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from mcat.profiles import _assign, _mahalanobis, _weights, tract_profiles, write_profiles

# --- weighted geometry and estimator --------------------------------------------------------

# pyAFQ AFQ/tests/test_fixes.py::test_mahal_fix (tractometry/pyAFQ@65a43a88): 20 streamlines,
# two nodes each, and the Mahalanobis distances pyAFQ pins for the first three streamlines.
MAHAL_SLS = np.array([
    [[30, 41, 61], [28, 61, 38]], [[30, 41, 62], [20, 44, 34]], [[50, 67, 88], [10, 10, 20]],
    [[35, 43, 65], [25, 55, 35]], [[40, 50, 70], [15, 15, 25]], [[45, 54, 75], [12, 22, 32]],
    [[32, 48, 68], [28, 58, 40]], [[38, 52, 72], [18, 38, 28]], [[34, 44, 64], [21, 41, 31]],
    [[36, 46, 66], [23, 53, 33]], [[37, 47, 67], [24, 54, 34]], [[39, 49, 69], [19, 39, 29]],
    [[33, 53, 73], [22, 42, 32]], [[31, 51, 71], [26, 56, 36]], [[29, 59, 79], [27, 57, 37]],
    [[28, 58, 78], [17, 47, 27]], [[27, 57, 77], [16, 36, 26]], [[26, 56, 76], [14, 24, 34]],
    [[25, 55, 75], [13, 23, 33]], [[24, 54, 74], [11, 21, 31]]], float)
MAHAL_PINNED = np.array([[1.718654, 1.550252], [2.202227, 0.7881], [3.415999, 2.689814]])


def test_unit_density_reproduces_pyafq_mahalanobis():
    """With equal densities the estimator is pyAFQ's (population covariance, pinvh)."""
    for node in range(2):
        D = _mahalanobis(MAHAL_SLS[:, node], np.ones(20))
        npt.assert_allclose(D[:3], MAHAL_PINNED[:, node], atol=1e-6)


def _pyafq_replicated(x, d):
    """pyAFQ's Mahalanobis distances and weights for x with each point repeated d_i times."""
    from AFQ._fixes import gaussian_weights

    rep = np.repeat(x, d, axis=0)[:, None]               # (n_sls, 1 node, 3)
    groups = np.zeros(rep.shape[:2], int)
    D = gaussian_weights(rep, assignment_idxs=groups, n_points=None, return_mahalanobis=True)[:, 0]
    w = gaussian_weights(rep, assignment_idxs=groups, n_points=None)[:, 0]
    first = np.r_[0, np.cumsum(d)[:-1]]
    return D[first], w, np.repeat(np.arange(len(x)), d)


# a small, asymmetric, non-planar cross-section with unequal integer densities (mm)
XSEC = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 1], [4, 1, 0], [1, 3, 2], [3, 3, 0]], float)
XSEC_D = np.array([1, 5, 2, 3, 1, 4, 2])                 # sum 18 >= pyAFQ's 15-point minimum
XSEC_S = np.array([0.30, 0.55, 0.41, 0.62, 0.20, 0.48, 0.71])


def test_density_weights_equal_replicated_points():
    """Density d_i is exactly a voxel repeated d_i times: weighted mean, population covariance,
    distances and the inverse-distance profile all match pyAFQ on the replicated points."""
    D_ref, w_ref, which = _pyafq_replicated(XSEC, XSEC_D)
    npt.assert_allclose(_mahalanobis(XSEC, XSEC_D.astype(float)), D_ref, rtol=1e-10)

    owner = np.zeros(len(XSEC), int)
    w = _weights(XSEC, XSEC_D.astype(float), owner, 1)
    profile = (w * XSEC_S).sum() / w.sum()
    expected = (w_ref * XSEC_S[which]).sum()              # pyAFQ weights already sum to one
    npt.assert_allclose(profile, expected, rtol=1e-10)

    # ...which neither unweighted geometry nor density-only weighting (QSIRecon's bundle mean) gives
    D_unweighted = _mahalanobis(XSEC, np.ones(len(XSEC)))
    unweighted = (XSEC_D / D_unweighted * XSEC_S).sum() / (XSEC_D / D_unweighted).sum()
    density_only = (XSEC_D * XSEC_S).sum() / XSEC_D.sum()
    assert abs(unweighted - expected) > 1e-3
    assert abs(density_only - expected) > 1e-3


@pytest.mark.parametrize("flat", [(2,), (1, 2)], ids=["planar", "linear"])
def test_thin_cross_sections_use_the_pseudoinverse(flat):
    """Rank-2 and rank-1 cross-sections are valid: no fallback, and pinvh agrees with pyAFQ."""
    x = XSEC.copy()
    x[:, list(flat)] = 7.0
    x[:, 0] = XSEC[:, 0] + 0.1 * np.arange(len(x))      # keep the points distinct
    D_ref, _, _ = _pyafq_replicated(x, XSEC_D)
    npt.assert_allclose(_mahalanobis(x, XSEC_D.astype(float)), D_ref, rtol=1e-8)
    w = _weights(x, XSEC_D.astype(float), np.zeros(len(x), int), 1)
    assert np.all(np.isfinite(w) & (w > 0))


def test_collapsed_node_falls_back_locally():
    """A one-voxel node is weighted by density alone; the other nodes are untouched."""
    x = np.vstack([XSEC, [[50, 50, 50]]])
    d = np.r_[XSEC_D, 3.0]
    owner = np.r_[np.zeros(len(XSEC), int), 1]
    w = _weights(x, d, owner, 2)
    assert w[-1] == 3.0
    alone = _weights(XSEC, XSEC_D.astype(float), np.zeros(len(XSEC), int), 1)
    npt.assert_allclose(w[:-1], alone, rtol=1e-12)


def test_zero_and_near_zero_distance():
    """pyAFQ's rule: an infinite inverse distance becomes zero weight; a near-centre voxel keeps
    its (large) finite d / D.  Symmetric square + its exact centre, then the centre nudged."""
    sq = np.array([[-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0], [0, 0, 0]], float)
    d = np.ones(5)
    w = _weights(sq, d, np.zeros(5, int), 1)
    assert w[4] == 0 and np.all(w[:4] > 0)                 # the exact centre is discarded
    nudged = sq.copy()
    nudged[4, 0] = 1e-6
    w2 = _weights(nudged, d, np.zeros(5, int), 1)
    D = _mahalanobis(nudged, d)
    npt.assert_allclose(w2, d / D)
    assert w2[4] / w2.sum() > 0.99                        # ...and dominates the node


# --- geodesic assignment --------------------------------------------------------------------

def _u_bundle():
    """Two one-voxel-wide arms (y=1 and y=3) joined at x=20, with an empty row between them."""
    keep = np.zeros((23, 5, 3), bool)
    keep[1:21, 1, 1] = keep[1:21, 3, 1] = True
    keep[20, 1:4, 1] = True
    nodes_vox = np.array([[1, 1, 1], [8, 1, 1], [15, 1, 1], [20, 2, 1],
                          [18, 3, 1], [11, 3, 1], [4, 3, 1]], float)
    return keep, nodes_vox


def test_assignment_follows_the_bundle_not_straight_lines():
    keep, nodes_vox = _u_bundle()
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    # uniform density: MCAT's cost is zero, so its 1e-9-per-step tie-break counts steps
    ijk, owner = _assign(keep.astype(float), keep, aff, nodes_vox @ aff[:3, :3].T)
    assert np.all(owner >= 0)
    arm_of_node = np.array([1, 1, 1, -1, 3, 3, 3])
    checked = wrong_by_euclid = 0
    for (i, j, _), o in zip(ijk, owner):
        if i > 16:
            continue                                      # near the bend node: no straight-arm oracle
        # in a straight one-voxel-wide arm the geodesic is |dx|: nearest node in the same arm
        same = np.flatnonzero(arm_of_node == j)
        dx = np.abs(nodes_vox[same, 0] - i)
        if np.sort(dx)[1] - dx.min() < 1:
            continue                                      # (near) tie
        assert o == same[np.argmin(dx)], (i, j)
        checked += 1
        euclid = np.argmin(np.linalg.norm(nodes_vox - [i, j, 1], axis=1))
        wrong_by_euclid += arm_of_node[euclid] != j
    assert checked > 20 and wrong_by_euclid > 0           # the case discriminates


def test_assignment_uses_mcats_density_cost():
    """MCAT's cross-section metric (core.py step 3): -log(density / peak) per mm.  A line of
    voxels x = 0..10, density 0.01 for x <= 3 and 1 beyond, nodes at x = 2 and x = 8.  By hand,
    with c = -ln(0.01) = 4.61: voxel 3 costs c x 1 mm = 4.61 from node 0, but only
    0 (x = 8..4) + (0 + c) / 2 = 2.30 from node 1, so node 1 owns it -- physical length says node 0."""
    dn = np.where(np.arange(11) <= 3, 0.01, 1.0)[:, None, None]
    keep = np.ones(dn.shape, bool)
    ijk, owner = _assign(dn, keep, np.eye(4), np.array([[2.0, 0, 0], [8.0, 0, 0]]))
    npt.assert_array_equal(owner[np.argsort(ijk[:, 0])], [0, 0, 0] + [1] * 8)


def test_assignment_steps_match_mcat():
    """Step lengths are bundle_core's: |offset| x vox_mm, vox_mm = |det A|^(1/3) (one mean voxel
    size), including the step from a node to its cell's corners.  On a (1, 1, 4) mm grid vox_mm =
    4^(1/3) = 1.587, and with cost c = ln 2 off the peak, by hand: voxel (2,0,0) is one diagonal
    step from node B at voxel (3,0,1), sqrt(2) x 1.587 x c = 1.56, while from node A at voxel
    (0,0,0) the cheapest route (via the zero-cost peak at (0,0,1)) is
    sqrt(2) x 1.587 x c / 2 + 1.587 x c = 1.88.  So B owns it (physical mm would say A: 2 vs 4.1)."""
    dn = np.full((4, 1, 2), 0.5)
    dn[0, 0, 1] = 1.0                                     # the peak: elsewhere cost is ln 2
    aff = np.diag([1.0, 1.0, 4.0, 1.0])
    ijk, owner = _assign(dn, dn > 0, aff, np.array([[0, 0, 0], [3, 0, 4.0]]))
    assert owner[np.flatnonzero((ijk == [2, 0, 0]).all(1))[0]] == 1


def test_duplicate_and_detached_nodes_keep_their_numbers():
    keep, nodes_vox = _u_bundle()
    nodes_vox = np.vstack([nodes_vox[:2], nodes_vox[1:2], nodes_vox[2:], [[10, 4.8, 2.5]]])
    aff = np.eye(4)
    _, owner = _assign(keep.astype(float), keep, aff, nodes_vox)
    counts = np.bincount(owner[owner >= 0], minlength=len(nodes_vox))
    # nodes 1 and 2 coincide: one owns their voxels, the other is empty; the last node has no
    # support voxel at any corner of its cell (detached) and is empty too
    assert sorted(counts[1:3] > 0) == [False, True]
    assert counts[-1] == 0
    assert np.all(counts[[0, 3, 4, 5, 6, 7]] > 0)


# --- scalars on another grid ----------------------------------------------------------------

def _scalar_img():
    """A linear field on an anisotropic, shifted, flipped grid; linear interpolation is exact."""
    aff = np.array([[-1.5, 0, 0, 20], [0, 1.0, 0, -3], [0, 0, 2.5, 1], [0, 0, 0, 1]])
    ijk = np.moveaxis(np.indices((12, 9, 6), dtype=float), 0, -1)
    xyz = nib.affines.apply_affine(aff, ijk)
    field = lambda p: 0.1 * p[..., 0] - 0.05 * p[..., 1] + 0.02 * p[..., 2] - 1.0  # noqa: E731
    return aff, field(xyz), field


# --- integration and output -----------------------------------------------------------------

def _straight_bundle():
    """A 3x3-voxel tube along x (2 mm voxels), density peaking on its axis, plus a far island."""
    d = np.zeros((24, 7, 7))
    prof = np.array([0.4, 1.0, 0.4])
    d[2:22, 2:5, 2:5] = prof[None, :, None] * prof[None, None, :] * np.linspace(1, 2, 20)[:, None, None]
    d[2:22, 3, 3] += 0.05 * np.arange(20) % 3           # break the cross-sections' symmetry
    d[23, 0, 0] = 1.0                                     # disconnected island: never in support
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = [-10, 4, 7]
    n = 10
    x_vox = np.linspace(2, 21, n)
    core = nib.affines.apply_affine(aff, np.stack([x_vox, np.full(n, 3.0), np.full(n, 3.0)], 1))
    return d, aff, core


def test_profile_end_to_end():
    d, aff, core = _straight_bundle()
    s_aff, s_data, field = _scalar_img()
    s_aff = s_aff.copy()
    s_aff[:3, :] = [[-4, 0, 0, 36], [0, 1, 0, 5], [0, 0, 2.5, 3]]  # a grid holding the bundle
    s_data = field(nib.affines.apply_affine(s_aff, np.moveaxis(np.indices(s_data.shape, float), 0, -1)))
    holes = s_data.copy()
    holes[...] = np.nan                                   # an all-invalid scalar
    prof, info = tract_profiles(d, aff, core, {"lin": nib.Nifti1Image(s_data, s_aff), "none": nib.Nifti1Image(holes, s_aff)})
    assert prof["lin"].shape == (10,) and np.all(np.isnan(prof["none"]))
    # each node's profile is a convex combination of a linear field over its own slab of the tube
    npt.assert_allclose(prof["lin"], field(core), atol=0.12)
    assert np.all(np.diff(np.sign(np.diff(prof["lin"]))) == 0)  # monotone along x, in node order
    assert info["n_support"] == np.count_nonzero(d[:22])  # the island is not support
    assert info["empty_nodes"] == []

    rev, _ = tract_profiles(d, aff, core[::-1], {"lin": nib.Nifti1Image(s_data, s_aff)})
    npt.assert_allclose(rev["lin"], prof["lin"][::-1], rtol=1e-12)


def test_empty_bundle_returns_nan_nodes():
    d, aff, core = _straight_bundle()
    prof, info = tract_profiles(np.zeros_like(d), aff, None, {"FA": nib.Nifti1Image(d, aff)}, n_nodes=7)
    assert prof["FA"].shape == (7,) and np.all(np.isnan(prof["FA"]))
    assert info["status"] == "empty"


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_invalid_density_is_zero_as_in_bundle_core(bad):
    """bundle_core reads density as clip(nan_to_num(d), 0): NaN and negative voxels are empty,
    +Inf the largest float.  The profiles do the same rather than raising."""
    d, aff, core = _straight_bundle()
    lin = {"x": nib.Nifti1Image(d, aff)}
    d_bad = d.copy()
    d_bad[5, 3, 3] = bad
    expected = np.clip(np.nan_to_num(d_bad), 0, None)
    prof, info = tract_profiles(d_bad, aff, core, lin)
    ref, _ = tract_profiles(expected, aff, core, lin)
    npt.assert_array_equal(prof["x"], ref["x"])


def test_write_profiles(tmp_path):
    from mcat.core import Core, write_core

    d, aff, core = _straight_bundle()
    stem = tmp_path / "sub-01_dwi_space-RASMM_model-probCSD"
    for name, dens in (("LeftArc", d), ("Empty", np.zeros_like(d))):
        pre = f"{stem}_bundle-{name}"
        nib.save(nib.Nifti1Image(dens.astype(np.float32), aff), f"{pre}_desc-density_map.nii.gz")
        with open(f"{pre}_desc-summary.json", "w") as fh:
            json.dump(dict(bundle="Left Arcuate" if name == "LeftArc" else "Empty one"), fh)
    write_core(Core(core, np.ones(len(core)), 1.0, "test"), f"{stem}_bundle-LeftArc")
    s_aff, s_data, _ = _scalar_img()
    s_aff[:3, :] = [[-4, 0, 0, 36], [0, 1, 0, 5], [0, 0, 2.5, 3]]
    fa = tmp_path / "fa.nii.gz"
    nib.save(nib.Nifti1Image(np.abs(s_data).astype(np.float32), s_aff), fa)

    out = write_profiles(tmp_path, {"dti_fa": fa})
    assert out.endswith("_desc-profiles_tractography.csv")
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == ["tractID", "nodeID", "dti_fa"]
    by = {}
    for r in rows:
        by.setdefault(r["tractID"], []).append(r)
    # bundles in file order; the empty bundle still has a row per node (of the core's count)
    assert list(by) == ["Empty one", "Left Arcuate"]
    assert [int(r["nodeID"]) for r in by["Left Arcuate"]] == list(range(10))
    assert all(r["dti_fa"] for r in by["Left Arcuate"])
    assert len(by["Empty one"]) == 100 and not any(r["dti_fa"] for r in by["Empty one"])
    meta = json.load(open(out[: -len(".csv")] + ".json"))
    assert meta["bundles"]["Empty one"]["status"] == "empty"
    assert meta["bundles"]["Left Arcuate"]["status"] == "ok"
    assert meta["scalars"] == {"dti_fa": str(fa)}
