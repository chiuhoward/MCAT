"""Single-shell three-tissue CSD fODFs (cs_dmri's SS3T: Dhollander responses, SS3T,
mtnormalise) as MPAT reads fODFs: DIPY descoteaux07 (legacy) SH, directions in the image's
voxel axes -- the frame DIPY's CSD, and so pyAFQ, fits bvecs in (as given, never flipped).
cs_dmri is optional: ``pip install "mpat-tractography[ss3t]"``.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np


def tournier_to_descoteaux(sh):
    """MRtrix3 SH (DIPY tournier07, the basis cs_dmri fits in) -> DIPY descoteaux07, legacy (MPAT's),
    same order.  Exact: both span the same functions, so the fODF is sampled on a sphere with
    more points than coefficients and projected back."""
    from dipy.data import get_sphere
    from dipy.reconst.shm import order_from_ncoef, sh_to_sf_matrix

    order = order_from_ncoef(np.shape(sh)[-1])
    S = get_sphere(name="repulsion724")
    B = sh_to_sf_matrix(S, sh_order_max=order, basis_type="tournier07", legacy=False, return_inv=False)
    _, inv = sh_to_sf_matrix(S, sh_order_max=order, basis_type="descoteaux07", legacy=True, return_inv=True)
    return (np.asarray(sh, np.float64) @ (B @ inv)).astype(np.float32)


def fit_ss3t(dwi, bval, bvec, mask, out_prefix, shell=None, shell_tol=100.0, b0_threshold=50.0, lmax=8,
             niter=3, normalize=True, n_threads=None, overwrite=False):
    """SS3T with cs_dmri's public API on the b=0 volumes and one shell (b within ``shell_tol`` of
    ``shell``; None: all volumes, which cs_dmri requires to be a single shell) of a preprocessed
    DWI series; ``mask``: brain mask on the DWI grid.  Writes
    ``<out_prefix>_model-ss3t_param-{wm,gm,csf}_dwimap.nii.gz`` (WM: the fODF in MPAT's basis;
    GM, CSF: signal fractions, not tissue probabilities) and a JSON sidecar next to the WM file
    (volumes, responses, settings, diagnostics, cs_dmri's version).  Returns their paths."""
    try:
        import cs_dmri
    except ImportError as err:
        raise ImportError('SS3T fODFs need cs_dmri: pip install "mpat-tractography[ss3t]"') from err
    from dipy.core.gradients import get_bval_indices

    out = {k: f"{out_prefix}_model-ss3t_param-{k}_dwimap.nii.gz" for k in ("wm", "gm", "csf")}
    out["json"] = out["wm"].replace(".nii.gz", ".json")
    if not overwrite and any(Path(p).exists() for p in out.values()):
        raise FileExistsError(f"{out_prefix}_model-ss3t_* exist; pass overwrite=True")
    d = cs_dmri.DWI.from_files(dwi, bval, bvec, mask=mask, b0_threshold=b0_threshold)   # checks the mask's grid
    g = d.gtab
    idx = np.arange(len(g))
    if shell is not None:
        sel = get_bval_indices(g.bvals, shell, tol=shell_tol)
        if not len(sel):
            raise ValueError(f"no volumes within {shell_tol:g} of b = {shell:g}")
        idx = np.union1d(np.flatnonzero(g.b0s_mask), sel)
        d = cs_dmri.DWI(d.data[..., idx], cs_dmri.GradientTable(g.bvals[idx], g.bvecs[idx], b0_threshold=b0_threshold),
                        affine=d.affine, mask=d.mask)
    resp = d.estimate_responses(lmax_wm=lmax, n_threads=n_threads)
    fit = cs_dmri.SS3TModel(d.gtab, resp, lmax_wm=lmax, niter=niter, bvec_frame="image").fit(
        d, diagnostics=True, n_threads=n_threads)
    if normalize:
        fit = fit.mtnormalise()
    Path(out["wm"]).parent.mkdir(parents=True, exist_ok=True)
    for k, v in (("wm", tournier_to_descoteaux(fit.wm)), ("gm", fit.gm), ("csf", fit.csf)):
        nib.save(nib.Nifti1Image(np.asarray(v, np.float32), d.affine), out[k])
    conv = np.asarray(fit.diagnostics["converged"]) > 0
    Path(out["json"]).write_text(json.dumps(dict(
        Model="SS3T-CSD (cs_dmri)", source=str(dwi), volumes=idx.tolist(), bvals=d.gtab.bvals.tolist(),
        SphericalHarmonicBasis="descoteaux07_legacy", SphericalHarmonicDegree=int(fit.lmax_wm), DirectionFrame="image",
        lmax=lmax, niter=niter, mtnormalise=normalize,
        responses={k: getattr(resp, k).to_text() for k in ("wm", "gm", "csf")},
        normalization={k: v for k, v in (fit.normalization or {}).items() if np.ndim(v) == 0},
        unconverged_voxels=int((~conv & d.effective_mask).sum()), warnings=list(fit.warnings),
        cs_dmri_version=cs_dmri.__version__), indent=1, default=float))
    return out
