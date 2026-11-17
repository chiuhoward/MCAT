"""SS3T fODFs from cs_dmri (mpat.ss3t): shell selection, the SH conversion to MPAT's basis, and
the fit's direction frame and placement on an oblique, flipped, anisotropic grid."""
import json

import nibabel as nib
import numpy as np
import pytest

SHAPE = (24, 20, 12)
FIBRE_A = np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5])    # image axes
FIBRE_B = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)


def _oblique_affine():
    """Anisotropic voxels, rotated about two axes, one axis flipped (negative determinant)."""
    a, b = np.radians(25), np.radians(-15)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    aff = np.eye(4)
    aff[:3, :3] = rz @ rx @ np.diag([-2.0, 2.0, 2.5])
    aff[:3, 3] = [30.0, -20.0, -15.0]
    return aff


def _phantom(bvals, seed=0):
    """Two single-fibre regions (fibres given in image axes, as bvecs are), grey matter and CSF."""
    from dipy.core.gradients import gradient_table
    from dipy.data import get_sphere
    from dipy.sims.voxel import multi_tensor

    rng = np.random.default_rng(seed)
    ndw = int(np.sum(np.asarray(bvals) > 50))
    v = get_sphere(name="repulsion724").vertices
    dirs = v[rng.choice(len(v), ndw, replace=False)]
    bvecs = np.zeros((len(bvals), 3))
    bvecs[np.asarray(bvals) > 50] = dirs
    gtab = gradient_table(np.asarray(bvals, float), bvecs=bvecs)
    i, j, _ = np.indices(SHAPE)
    regions = {"A": i < 8, "B": (i >= 8) & (i < 16), "gm": (i >= 16) & (j < 10), "csf": (i >= 16) & (j >= 10)}
    ang = lambda d: (np.degrees(np.arccos(d[2])), np.degrees(np.arctan2(d[1], d[0])))  # noqa: E731
    wm_ev = np.array([[1.7e-3, 0.2e-3, 0.2e-3]])
    sig = {"A": multi_tensor(gtab, wm_ev, S0=100, angles=[ang(FIBRE_A)], fractions=[100], snr=None)[0],
           "B": multi_tensor(gtab, wm_ev, S0=100, angles=[ang(FIBRE_B)], fractions=[100], snr=None)[0],
           "gm": multi_tensor(gtab, np.array([[0.8e-3] * 3]), S0=80, angles=[(0, 0)], fractions=[100], snr=None)[0],
           "csf": multi_tensor(gtab, np.array([[3e-3] * 3]), S0=150, angles=[(0, 0)], fractions=[100], snr=None)[0]}
    dwi = np.zeros(SHAPE + (len(bvals),), np.float32)
    for k, m in regions.items():
        dwi[m] = sig[k]
    dwi += rng.normal(0, 1.0, dwi.shape).astype(np.float32)
    return dwi, bvecs, regions


def _write(tmp_path, dwi, bvals, bvecs, aff):
    p = {k: str(tmp_path / f"sub-01_dwi.{k}") for k in ("bval", "bvec")}
    p["dwi"] = str(tmp_path / "sub-01_dwi.nii.gz")
    p["mask"] = str(tmp_path / "sub-01_mask.nii.gz")
    nib.save(nib.Nifti1Image(dwi, aff), p["dwi"])
    nib.save(nib.Nifti1Image(np.ones(SHAPE, np.uint8), aff), p["mask"])
    np.savetxt(p["bval"], np.asarray(bvals)[None], fmt="%g")
    np.savetxt(p["bvec"], bvecs.T, fmt="%.8f")
    return p


def _peak(coef):
    """Peak direction of MPAT-basis coefficients, on MPAT's own sphere and basis."""
    from mpat.geometry import _sh_basis, build_geometry

    V = build_geometry((2.0, 2.0, 2.0), 35.0).vertices
    return V[np.argmax(coef @ _sh_basis(V, coef.size))]


def test_tournier_to_descoteaux_matches_cs_dmri_export(tmp_path):
    """The conversion reproduces, on the sphere, cs_dmri's own PAM5 (descoteaux07, legacy)
    export of the same fit.  Adapted from cs_dmri's test_pam5_export_loads_in_dipy
    (python/tests/test_models.py): an SS3TFit built directly on a RAS+ grid, where the export's
    world frame is the image frame; here its WM coefficients are not zonal, so a sign or
    axis slip in the conversion shows."""
    cs = pytest.importorskip("cs_dmri")
    from dipy.io.peaks import load_pam

    from mpat.geometry import _sh_basis
    from mpat.ss3t import tournier_to_descoteaux

    shape, lmax = (5, 5, 4), 8
    rng = np.random.default_rng(0)
    wm = rng.normal(size=shape + (45,)).astype(np.float32)
    responses = cs.ResponseSet(
        wm=cs.TissueResponse(np.array([[3.5, 0, 0, 0, 0], [1.2, -0.6, 0.2, -0.05, 0.01]]), lmax),
        gm=cs.TissueResponse(np.array([[3.0], [1.0]]), 0),
        csf=cs.TissueResponse(np.array([[4.0], [0.3]]), 0))
    fit = cs.SS3TFit(wm=wm, gm=np.full(shape, 0.05, np.float32), csf=np.full(shape, 0.02, np.float32),
                     lmax_wm=lmax, mask=np.ones(shape, bool), affine=np.diag([2.0, 2.0, 2.0, 1.0]),
                     frame="world", responses=responses)
    fit.export(tmp_path / "fit.pam5")
    ref = load_pam(str(tmp_path / "fit.pam5")).shm_coeff
    v = rng.normal(size=(200, 3))
    B = _sh_basis(v / np.linalg.norm(v, axis=1, keepdims=True), 45)
    np.testing.assert_allclose(tournier_to_descoteaux(wm) @ B, ref @ B, atol=1e-4)


def test_fit_ss3t_frame_and_placement(tmp_path):
    """On an oblique, flipped grid: each region's fODF peaks along its fibre in image axes (the
    frame MPAT and DIPY take bvecs in), the output keeps the DWI's grid, and only the b=0
    volumes and the chosen shell are fitted (multishell input without a shell is refused)."""
    pytest.importorskip("cs_dmri")
    from mpat.ss3t import fit_ss3t

    bvals = np.r_[0, np.full(32, 1000.0), 0, np.full(32, 2000.0)]
    dwi, bvecs, regions = _phantom(bvals)
    aff = _oblique_affine()
    p = _write(tmp_path, dwi, bvals, bvecs, aff)
    fit = lambda **kw: fit_ss3t(p["dwi"], p["bval"], p["bvec"], p["mask"], tmp_path / "out" / "sub-01", **kw)  # noqa: E731
    with pytest.raises(ValueError, match="one DWI shell"):
        fit()
    with pytest.raises(ValueError, match="no volumes"):
        fit(shell=3000)
    out = fit(shell=2000)
    img = nib.load(out["wm"])
    assert img.shape == SHAPE + (45,)
    np.testing.assert_allclose(img.affine, aff, atol=1e-5)
    sh = np.asarray(img.dataobj, np.float32)
    assert np.isfinite(sh).all()
    rot = aff[:3, :3] / np.linalg.norm(aff[:3, :3], axis=0)
    for fibre, ijk in ((FIBRE_A, (3, 5, 6)), (FIBRE_A, (6, 15, 3)), (FIBRE_B, (11, 14, 2)), (FIBRE_B, (13, 3, 8))):
        pk = _peak(sh[ijk])
        assert abs(pk @ fibre) > 0.99, (ijk, pk, fibre)
        w = rot @ fibre                                         # the same fibre in world axes
        assert abs(pk @ (w / np.linalg.norm(w))) < 0.5           # a world-frame fit would peak here
    meta = json.loads(open(out["json"]).read())
    assert meta["volumes"] == [0, *range(33, 66)]
    assert meta["SphericalHarmonicBasis"] == "descoteaux07_legacy" and meta["DirectionFrame"] == "image"
    with pytest.raises(FileExistsError):
        fit(shell=2000)


def test_missing_cs_dmri(monkeypatch):
    import sys

    from mpat.ss3t import fit_ss3t

    monkeypatch.setitem(sys.modules, "cs_dmri", None)
    with pytest.raises(ImportError, match=r"mpat-tractography\[ss3t\]"):
        fit_ss3t("dwi.nii.gz", "dwi.bval", "dwi.bvec", "mask.nii.gz", "out")


def test_run_from_images(tmp_path, monkeypatch):
    """SS3T fODF -> MPAT bundles from explicit files only (no pyAFQ: importing it fails here):
    the synthetic subject's DWI, its CSF / GM / WM probabilities as separate files on the finer
    T1 grid, a seed mask and the bundles' ROI files.  Each bundle stays in its own band, and the
    seeds given are the ones used."""
    import sys

    pytest.importorskip("cs_dmri")
    from nibabel.processing import resample_from_to
    from synthetic_subject import make_subject

    import mpat
    from mpat.ss3t import fit_ss3t

    f = make_subject(tmp_path / "raw")
    dwi = nib.load(f["dwi"])
    mask = resample_from_to(nib.load(f["mask"]), (dwi.shape[:3], dwi.affine), order=0)
    nib.save(mask, tmp_path / "mask_dwi.nii.gz")
    fodf = fit_ss3t(f["dwi"], f["bval"], f["bvec"], tmp_path / "mask_dwi.nii.gz", tmp_path / "fit" / "sub-01")["wm"]
    bundles = [dict(name=k, **v) for k, v in f["rois"].items()]
    monkeypatch.setitem(sys.modules, "AFQ", None)

    def run(seeds, out):
        seed_file = tmp_path / f"{out}_seeds.nii.gz"
        nib.save(nib.Nifti1Image(seeds.astype(np.uint8), f["affine"]), seed_file)
        res = mpat.run_from_images(fodf, f["pves"], f["mask"], seed_file, bundles, tmp_path / out, n_proc=2,
                                   max_mem="2GB", viz=False)
        return res, tmp_path / out

    res, out = run(f["wm"], "both")
    dseg = nib.load(out / "sub-01_desc-mpat_dseg.nii.gz")
    np.testing.assert_allclose(dseg.affine, f["affine"])
    assert (np.asarray(dseg.dataobj)[f["A"] & ~f["B"]][:, None] == 1).mean() > 0.5    # WM is WM: channel order
    for b, own in (("A", f["A"]), ("B", f["B"])):
        assert res[b].info["expected_bundle_streamlines"] > 10
        assert res[b].info["endpoint_mass_check"] == pytest.approx(1.0, abs=1e-6)
        d = nib.load(out / f"sub-01_bundle-{b}_desc-density_map.nii.gz")
        np.testing.assert_allclose(d.affine, f["affine"])
        assert d.get_fdata()[own].sum() > 0.99 * d.get_fdata().sum()
        assert (out / f"sub-01_bundle-{b}_desc-endpoints_map.nii.gz").exists()
    assert (out / "sub-01_desc-mpat_table.tsv").exists()
    res_a, _ = run(f["A"] & ~f["B"], "seedsA")                  # seeds only in band A
    assert res_a["B"].info["expected_bundle_streamlines"] < 1e-3 * res["B"].info["expected_bundle_streamlines"]
    assert res_a["A"].info["expected_bundle_streamlines"] > 10
