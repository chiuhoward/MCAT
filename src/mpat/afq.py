"""MPAT on top of pyAFQ.  Everything except the Markov chain comes from a pyAFQ
``ParticipantAFQ`` (or each participant of a ``GroupAFQ``): the fODF pyAFQ tracks on, the
tissue PVE and brain mask, the white/grey-matter interface (seeds), the bundle
definitions, and their ROIs warped to the subject.  Outputs go to ``<output_dir>/mpat``
(``mpat-moments`` with spatial moments), next to pyAFQ's own, or to ``out_dir`` (pyAFQ's
derivatives are then only read).  ``run_from_images`` runs the same chain and solves from
image files alone (an fODF, tissue probabilities, masks and subject-space ROIs), without pyAFQ.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from .chain import ABSORB_CLASSES, Chain, ChainConfig
from .geometry import visitation_odf
from .solve import Clock, overlength_states, prepare_launches, segment_bundles

log = logging.getLogger("mpat")
WM_THRESHOLD = 0.4   # pyAFQ's tracker stops where WM PVE <= gm_threshold (0.4)
ODF_FLOOR = 1e-3     # visitation ODFs are written where the density reaches this x its peak


def _on_grid(img, ref, order=1):
    """img on ref's grid: linear as pyAFQ's tracker does (dipy.align.resample), or nearest
    neighbour (order=0) for masks."""
    from dipy.align import resample
    from nibabel.processing import resample_from_to

    data = np.asarray(img.dataobj, np.float32)
    if img.shape[:3] == ref.shape[:3] and np.allclose(img.affine, ref.affine, atol=1e-3):
        return data
    if order == 0:
        return np.asarray(resample_from_to(nib.Nifti1Image(data, img.affine), (ref.shape[:3], ref.affine),
                                           order=0).dataobj, np.float32)
    vols = [data] if data.ndim == 3 else [data[..., i] for i in range(data.shape[-1])]
    out = [np.asarray(resample(v, np.zeros(ref.shape[:3], np.float32), moving_affine=img.affine,
                               static_affine=ref.affine).get_fdata(), np.float32) for v in vols]
    return out[0] if data.ndim == 3 else np.stack(out, -1)


HOLE_WARN = 0.01   # warn when this fraction of the fODF region is all-zero holes inside it


def tissue_labels(sh_img, pve_img, brain_mask_img=None):
    """As pyAFQ's tracker: walkers move where WM PVE > 0.4 inside the fODF mask; the mask's
    inner boundary counts as grey matter ("edge", so fibres leaving the field end validly);
    other in-mask voxels are GM (GM PVE >= CSF PVE), which absorbs walkers, or CSF, which
    walkers never enter (zero transition probability).  White matter that cannot reach any
    ending -- a pocket walled in by CSF and the edge of the fODF field (in a tumour, say) --
    would trap walkers seeded in it forever and stall the solves: it is taken out of the walk
    (treated as CSF, never entered), labelled 5 in dseg, and logged.
    Returns transient, absorbing class (0 outside, 1 GM, 2 CSF, 3 edge), dseg (1 WM, 2 GM,
    3 CSF, 4 edge, 5 isolated WM)."""
    from skimage.segmentation import find_boundaries

    mask = np.any(np.asarray(sh_img.dataobj) != 0, axis=-1)
    if brain_mask_img is not None:
        mask &= _on_grid(brain_mask_img, sh_img) > 0.5
    csf, gm, wm = np.moveaxis(_on_grid(pve_img, sh_img), -1, 0)
    # holes inside the fODF region (voxels whose fit is all zero) each get a rim of "edge",
    # where walkers end validly: a few are harmless, many cut every bundle short
    from scipy import ndimage as ndi

    holes = int((ndi.binary_fill_holes(mask) & ~mask).sum())
    if holes > HOLE_WARN * mask.sum():
        log.warning("%d voxels (%.1f%% of the fODF region) inside it have an all-zero fODF; each hole is "
                    "ringed by edge voxels where walkers end, so bundles may come out near zero.  Check "
                    "the fODF fit (a stale or partial model file, a failed fit, NaN / zero signal)",
                    holes, 100 * holes / mask.sum())
    elif holes:
        log.info("%d voxels inside the fODF region have an all-zero fODF", holes)
    edge = find_boundaries(mask, mode="inner") & mask
    transient = mask & ~edge & (wm > WM_THRESHOLD)
    absorb = np.zeros(mask.shape, np.int8)
    absorb[mask & ~transient & (gm >= csf)] = 1
    absorb[mask & ~transient & (gm < csf)] = 2
    absorb[edge] = 3
    # white matter with no way to an ending (walkers step to any of the 26 neighbours)
    cube = np.ones((3, 3, 3), bool)
    lab, n = ndi.label(transient, cube)
    trapped = np.zeros(mask.shape, bool)
    if n:
        ends = ndi.binary_dilation((absorb == 1) | (absorb == 3), cube)
        reach = np.zeros(n + 1, bool)
        reach[np.unique(lab[ends & transient])] = True
        trapped = transient & ~reach[lab]
        if trapped.any():
            sizes = np.bincount(lab[trapped])
            sizes = np.sort(sizes[sizes > 0])[::-1]
            log.info("isolated white matter: %d voxels in %d pockets with no way to grey matter or the "
                     "edge (largest %d voxels) are left out of the walk (dseg 5)", trapped.sum(),
                     len(sizes), sizes[0])
            transient = transient & ~trapped
            absorb[trapped] = 2
    dseg = np.where(transient, 1, np.where(absorb > 0, absorb + 1, 0)).astype(np.uint8)
    dseg[trapped] = 5
    return transient, absorb, dseg


def seal_26(mask):
    """Make a thin ROI impassable for 26-connected walks.  Streamlines cannot slip through a
    one-voxel sheet; walkers stepping diagonally can.  Adding the sheet's 6-neighbours closes
    those gaps."""
    from scipy import ndimage

    return ndimage.binary_dilation(mask, ndimage.generate_binary_structure(3, 1))


def _subject_thalamus(name, k, roi, thalamus, x, tol):
    """``roi``, or the subject's thalamus on its side if ``roi`` is the template's thalamus."""
    from scipy.ndimage import distance_transform_edt

    either = thalamus["left"] | thalamus["right"]
    if not either.any():
        log.warning("%s %s: SynthSeg found no thalamus on this grid; keeping the template's ROI", name, k)
        return roi
    near = distance_transform_edt(~either) <= max(tol, 1)
    centre = tuple(np.round(np.argwhere(roi).mean(0)).astype(int))
    frac = float(near[roi].mean())
    if frac < 0.5 and not either[centre]:
        if frac >= 0.1:   # thalamus-ish but not taken: say so, so it can be checked
            log.info("%s %s: %.0f%% on the subject's thalamus, centre outside it: kept the template's ROI",
                     name, k, 100 * frac)
        return roi
    side = "left" if x[roi].mean() < 0 else "right"
    log.info("%s %s: the template thalamus (%.0f%% on the subject's) -> SynthSeg's %s thalamus "
             "(%d voxels, was %d)", name, k, 100 * frac, side, thalamus[side].sum(), roi.sum())
    return thalamus[side].copy()


def bundle_specs(afq, ref_img, transient, names=None, thalamus=None, no_midline=(), midline_tol=None,
                 fornix=True):
    """bundle_specs_iter as a list (every bundle's ROIs at once)."""
    return list(bundle_specs_iter(afq, ref_img, transient, names, thalamus, no_midline, midline_tol, fornix))


def mni_box(box, bd, mapping, ref_img):
    """An MNI box {x: (lo, hi), y: ..., z: ...} (mm) on ``ref_img``'s grid, warped as pyAFQ warps
    its template ROIs."""
    import AFQ.utils.volume as auv

    tpl = bd.resample_to
    tpl = nib.load(tpl) if isinstance(tpl, (str, Path)) else tpl
    xyz = nib.affines.apply_affine(tpl.affine, np.indices(tpl.shape[:3]).reshape(3, -1).T)
    m = np.ones(len(xyz), bool)
    for i, k in enumerate("xyz"):
        lo, hi = box[k]
        m &= (xyz[:, i] >= lo) & (xyz[:, i] <= hi)
    warped = auv.transform_roi(m.reshape(tpl.shape[:3]).astype(np.float64), mapping, True)
    return _on_grid(nib.Nifti1Image(np.asarray(warped, np.float32), ref_img.affine), ref_img, order=0) > 0


def bundle_specs_iter(afq, ref_img, transient, names=None, thalamus=None, no_midline=(), midline_tol=None,
                      fornix=True, hypo_file=None, parc_file=None):
    """MPAT bundle specs from pyAFQ's bundle dictionary and MPAT's rules (mpat.bundles): include /
    exclude ROIs (sealed), start / end ROIs grown by pyAFQ's ``dist_to_atlas``, and the other
    hemisphere excluded for bundles that may not cross the midline.  Other pyAFQ criteria
    (curvature, length, cleaning) are streamline-based and do not apply.  Yields one bundle at
    a time, each warped only when it is asked for, so a run holds the ROIs of the bundles it is
    solving and no others.

    no_midline  bundle names whose cross_midline=False is ignored (no other-hemisphere exclusion)
    midline_tol overrides the midline_tol rule of the bundles that have one (the corticospinal
                tracts); None: the rules' own
    fornix      False: drop the fornix box (mpat.bundles.FORNIX_BOX_MNI) from the exclusions
                (the anterior thalamic radiations as pyAFQ defines them)
    parc_file   where SynthSeg with its cortical parcels and hypothalamus is kept (written on
                first need), when a bundle's SynthSeg ROIs use cortical parcels (or the
                hypothalamus and no hypo_file has it)
    hypo_file   where SynthSeg with the hypothalamus is kept, when a bundle's SynthSeg ROIs use
                the hypothalamus (the fornix) and pyAFQ's SynthSeg lacks it (_synthseg_on_grid)
    thalamus    {"left": mask, "right": mask} on ``ref_img``'s grid (SynthSeg's): a start / end
                ROI that is the template's thalamus -- at least half of it within ``dist_to_atlas``
                of a thalamus, or its centre inside one -- is replaced by the subject's thalamus
                on its side (unless the bundle's rule says thalamus="template").  The thalamus
                is grey matter, where walkers end, so a template thalamus warped a few mm off
                would make most real endings invalid."""
    from scipy.ndimage import distance_transform_edt

    from .bundles import FORNIX_BOX_MNI, rules, synthseg_values

    bd, mapping = afq.export("bundle_dict"), afq.export("mapping")
    dist = afq.export("segmentation_params").get("dist_to_atlas", 4)
    R = ref_img.affine[:3, :3]
    tol = int(dist / np.mean(np.diag(np.linalg.cholesky(R.T @ R))))   # pyAFQ's tolerance_mm_to_vox
    x = nib.affines.apply_affine(ref_img.affine, np.indices(transient.shape).reshape(3, -1).T)[:, 0]
    x = x.reshape(transient.shape)
    boxes, seg, seg_has = {}, None, set()

    def box(b):
        key = tuple(sorted((k, tuple(v)) for k, v in b.items()))
        if key not in boxes:
            boxes[key] = seal_26(mni_box(b, bd, mapping, ref_img))
            log.info("MNI box %s: %d voxels on this grid", dict(b), int(boxes[key].sum()))
        return boxes[key]

    for name in (names or list(bd)):
        r = rules(bd, name)
        rois = bd.transform_rois(name, mapping, ref_img)
        arr = lambda img: _on_grid(img, ref_img, order=0) > 0  # noqa: E731
        spec = dict(name=name, include=[seal_26(arr(m)) for m in rois.get("include", [])])
        spec["include"] += [box(b) for b in r["include_mni_boxes"]]
        exc = [seal_26(arr(m)) for m in rois.get("exclude", [])]
        exc += [box(b) for b in r["exclude_mni_boxes"] if fornix or dict(b) != FORNIX_BOX_MNI]
        spec["exclude_rois"] = np.any(exc, axis=0) if exc else None      # without the hemisphere (for display)
        for k in ("start", "end"):
            labels = r[f"{k}_synthseg"]
            if labels:
                values = synthseg_values(labels)
                if seg is None or not _extras(values) <= seg_has:
                    seg = _synthseg_on_grid(afq, ref_img, values, hypo_file=hypo_file, parc_file=parc_file)
                    seg_has = _extras(np.unique(seg))
                roi = np.isin(seg, values)
                if not roi.any():
                    log.warning("%s %s: SynthSeg labels %s are empty on this grid", name, k, labels)
                log.info("%s %s: SynthSeg %s (%d voxels)", name, k, ", ".join(map(str, labels)), int(roi.sum()))
            elif k in rois:
                roi = arr(rois[k])
                if thalamus is not None and r["thalamus"] == "synthseg" and roi.any():
                    roi = _subject_thalamus(name, k, roi, thalamus, x, tol)
            else:
                continue
            k_tol = tol
            if labels and r["synthseg_tol_mm"] is not None:            # SynthSeg ends: their own growth
                k_tol = int(r["synthseg_tol_mm"] / np.mean(np.diag(np.linalg.cholesky(R.T @ R))))
            spec[k] = roi if k_tol <= 0 else distance_transform_edt(~roi) - 0.5 <= k_tol   # pyAFQ's endpoint test
        if bd[name].get("cross_midline") is False and name in no_midline:
            log.info("%s: no midline exclusion (no_midline)", name)
        elif bd[name].get("cross_midline") is False:
            t = r["midline_tol"] if midline_tol is None or not r["midline_tol"] else midline_tol
            left = _is_left(name, spec, x)
            exc.append(x > t if left else x < -t)
            log.info("%s: kept to the %s hemisphere (other side excluded past %g mm)", name,
                     "left" if left else "right", t)
        spec["exclude"] = np.any(exc, axis=0) if exc else None
        yield spec


HYPOTHALAMUS = (102, 103)     # SynthSegLabels.LEFT_ / RIGHT_HYPOTHALAMUS


def _is_left(name, spec, x):
    """Which hemisphere a one-sided bundle belongs to: its name ("Left ..." / "Right ..."), else
    its include ROIs, else its start / end ROIs, each ROI counted once (its centre).  Not
    pooled voxels: a big midline end ROI shared by both sides (the corticospinal tracts'
    brainstem) would outvote the waypoints and put a bundle on the wrong side."""
    n = name.lower()
    if n.startswith("left") or n.endswith("_l") or n.endswith(" l"):
        return True
    if n.startswith("right") or n.endswith("_r") or n.endswith(" r"):
        return False
    for rois in (spec["include"], [spec[k] for k in ("start", "end") if spec.get(k) is not None]):
        c = [x[m].mean() for m in rois if m.any()]
        if c:
            return float(np.mean(c)) < 0
    return True


CORTEX_PARCELS = range(34, 102)  # SynthSegLabels.CTX_LH_BANKSSTS .. CTX_RH_INSULA (Desikan-Killiany)


def _extras(values):
    """SynthSeg's extra models that labels ``values`` need: subset of {"hypothalamus", "cortex"}."""
    v = {int(x) for x in values}
    return ({"hypothalamus"} if v & set(HYPOTHALAMUS) else set()) | \
        ({"cortex"} if v & set(CORTEX_PARCELS) else set())


def _has(img, extras):
    """Does the label image hold the labels of these extra models?"""
    return _extras(np.unique(np.asarray(img.dataobj))) >= set(extras)


def parc_file_of(stem):
    return f"{stem}_model-synthseg2parc_dseg.nii.gz"


def _synthseg_on_grid(afq, ref_img, values=(), hypo_file=None, parc_file=None):
    """pyAFQ's SynthSeg labels on ``ref_img``'s grid, holding the labels ``values`` need.

    pyAFQ's own SynthSeg run has neither the hypothalamus (102 / 103) nor the cortical parcels
    (34-101).  When ``values`` need one that pyAFQ's segmentation lacks, an earlier rerun is
    reused (``hypo_file``: hypothalamus only; ``parc_file``: both), or else SynthSeg is rerun
    on pyAFQ's T1 with both extra models and written to ``parc_file`` (synthseg_parcellated).
    pyAFQ's own segmentation and tissue are left as they are."""
    need = _extras(values)
    seg = afq.export("synthseg_model")
    seg = nib.load(seg) if isinstance(seg, str) else seg
    if need and not _has(seg, need):
        seg = None
        if need == {"hypothalamus"} and hypo_file and Path(hypo_file).exists() and _has(nib.load(hypo_file), need):
            seg = nib.load(hypo_file)
            log.info("SynthSeg with hypothalamus: %s", hypo_file)
        if seg is None:
            seg = synthseg_parcellated(afq, parc_file)
    lab = np.rint(_on_grid(seg, ref_img, order=0)).astype(np.int32)
    return lab[..., 0] if lab.ndim == 4 else lab


def synthseg_parcellated(afq, path=None):
    """SynthSeg 2 with its cortical parcellation and hypothalamus models on pyAFQ's T1
    (AFQ.nn.synthseg.run_synthseg, parc_cortex and parc_hypothalamus): read from ``path`` if
    it is there and complete, else run (once; several minutes on a CPU) and written there."""
    full = {"hypothalamus", "cortex"}
    if path is not None and Path(path).exists():
        img = nib.load(path)
        if _has(img, full):
            log.info("SynthSeg with cortical parcels and hypothalamus: %s", path)
            return img
        log.warning("%s lacks cortical parcels or the hypothalamus; rerunning SynthSeg", path)
    from AFQ.nn.synthseg import run_synthseg
    from AFQ.tasks.utils import check_onnxruntime

    ort = check_onnxruntime("SynthSeg 2.0 (parcellation / hypothalamus)",
                            "Or give these bundles template ROIs instead of SynthSeg labels.")
    try:
        onnx_kwargs = afq.export("onnx_kwargs")
    except Exception:  # noqa: BLE001  (older pyAFQ)
        onnx_kwargs = {}
    t1 = afq.kwargs["t1_file"]              # an input, not an export
    log.info("running SynthSeg with its cortical parcellation and hypothalamus models on %s "
             "(pyAFQ's SynthSeg has neither; once per subject)", t1)
    t0 = time.time()
    img = run_synthseg(ort, nib.load(t1), "synthseg2", onnx_kwargs, parc_cortex=True, parc_hypothalamus=True)
    img = nib.Nifti1Image(np.asarray(img.dataobj).astype(np.int16), img.affine)
    for e in sorted(full):
        if not _has(img, {e}):
            log.warning("SynthSeg's %s model found no %s labels", e, e)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, path)
        log.info("SynthSeg with cortical parcels and hypothalamus written to %s (%.0fs)", path, time.time() - t0)
    return img


def synthseg_cst_for_pyafq(afq, bd, moments=False, out_dir=None):
    """``bd`` (already pyafq_bd'd) with pyAFQ's own versions of the SynthSeg corticospinal tracts
    (mpat.bundles.synthseg_cst_bd): their start / end masks from SynthSeg's parcellation
    (synthseg_parcellated: run if it is not there yet) written on the DWI grid in MPAT's
    folder, as subject-space ROIs pyAFQ's recognition reads (endpoints only, exact_endpoints: not
    grown by dist_to_atlas, as MPAT's synthseg_tol_mm 0)."""
    from .bundles import synthseg_cst_labels, synthseg_values

    out = mpat_dir(afq, moments, out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = str(out / Path(afq.kwargs["base_fname"]).name)
    seg = synthseg_parcellated(afq, parc_file_of(stem))
    ref_path = afq.export("b0")
    ref = nib.load(ref_path)
    ref = nib.Nifti1Image(np.zeros(ref.shape[:3], np.uint8), ref.affine)
    lab = np.rint(_on_grid(seg, ref, order=0)).astype(np.int32)
    lab = lab[..., 0] if lab.ndim == 4 else lab
    bd.resample_subject_to = ref
    for side in ("Left", "Right"):
        name = f"{side} Corticospinal"
        entry = dict(cross_midline=False, space="subject", exact_endpoints=True)   # no 4 mm growth, as in MPAT
        for k, labels in zip(("start", "end"), synthseg_cst_labels(side)):
            f = f"{stem}_bundle-{side}Corticospinal_desc-synthseg{k}pyafq_mask.nii.gz"
            m = np.isin(lab, synthseg_values(labels)).astype(np.uint8)
            nib.save(nib.Nifti1Image(m, ref.affine), f)
            log.info("%s %s for pyAFQ: %s (%d voxels)", name, k, f, int(m.sum()))
            entry[k] = f
        bd[name] = entry
    return bd


def _subject_thalami(afq, sh_img):
    """{"left", "right"} SynthSeg thalami on the fODF grid."""
    from AFQ.nn.synthseg import SynthSegLabels

    lab = _synthseg_on_grid(afq, sh_img)
    return {"left": lab == int(SynthSegLabels.LEFT_THALAMUS), "right": lab == int(SynthSegLabels.RIGHT_THALAMUS)}


def _thalami_if_synthseg(afq, sh_img, synthseg_thalamus):
    """SynthSeg's thalami when pyAFQ's tissue comes from SynthSeg (``pve`` unset, the default
    "synthseg", or a built-in method that includes it), else None; logged."""
    pve = afq.kwargs.get("pve")
    from_synthseg = pve is None or (isinstance(pve, str) and "synthseg" in pve.lower())
    use = synthseg_thalamus and from_synthseg
    why = "--mni-thalamus" if not synthseg_thalamus else f"pve={pve if isinstance(pve, str) else type(pve).__name__}"
    log.info("thalamic start / end ROIs: %s", "SynthSeg's thalami wherever the template's thalamus is used"
             if use else f"the template's ({why})")
    return _subject_thalami(afq, sh_img) if use else None


def roi_specs(afq, sh_img, transient, stem, bundles=None, thalamus=None, no_midline=(), midline_tol=None,
              fornix=True):
    """bundle_specs_iter, writing each bundle's ROI masks as it goes: start / end, the include
    ROIs (labelled 1, 2, ... in pyAFQ's order) and the exclude ROIs (without the other
    hemisphere, which cross_midline=False bundles exclude too)."""
    for spec in bundle_specs_iter(afq, sh_img, transient, bundles, thalamus=thalamus, no_midline=no_midline,
                                  midline_tol=midline_tol, fornix=fornix,
                                  hypo_file=f"{stem}_model-synthseg2hypo_dseg.nii.gz",
                                  parc_file=parc_file_of(stem)):
        b = f"{stem}_bundle-{re.sub(r'[^A-Za-z0-9]', '', spec['name'])}"
        save = lambda m, desc: nib.save(nib.Nifti1Image(np.asarray(m, np.uint8), sh_img.affine),  # noqa: E731
                                        f"{b}_desc-{desc}_mask.nii.gz")
        for k in ("start", "end"):
            if spec.get(k) is not None:
                save(spec[k], k)
        if spec["include"]:
            lab = np.zeros(transient.shape, np.uint8)
            for i, m in enumerate(spec["include"], 1):
                lab[m & (lab == 0)] = i
            save(lab, "include")
        exc = spec.pop("exclude_rois", None)
        if exc is not None:
            save(exc, "exclude")
        yield spec


def write_rois(afq, bundles=None, moments=False, synthseg_thalamus=True, no_midline=(), page=True,
               midline_tol=None, out_dir=None, fornix=True):
    """Write the ROI masks of a finished run's bundles (start / end / include / exclude) without
    solving anything, and a viewer page showing them (bundles with no streamlines included).
    Returns the page's path (or None)."""
    from .core import bundle_files
    from .viewer import save_html

    sh_img = nib.load(afq.export(f"{afq.export('tracking_params')['odf_model'].lower()}_params"))
    transient, _, _ = tissue_labels(sh_img, nib.load(afq.export("pve_internal")), nib.load(afq.export("brain_mask")))
    out = mpat_dir(afq, moments, out_dir)
    stem = str(out / Path(afq.kwargs["base_fname"]).name)
    done = bundle_files(out)
    names = list(bundles or afq.export("bundle_dict"))
    run_names = [n for n in names if re.sub(r"[^A-Za-z0-9]", "", n) in done]
    if len(run_names) < len(names):
        log.warning("no run outputs for %s: their ROIs are not written",
                    [n for n in names if n not in run_names])
    thal = _thalami_if_synthseg(afq, sh_img, synthseg_thalamus)
    for spec in roi_specs(afq, sh_img, transient, stem, run_names, thal, no_midline, midline_tol, fornix):
        log.info("%s: ROI masks written", spec["name"])
    if not page or not run_names:
        return None
    return save_html(f"{stem}_desc-mpat_rois.html", str(out),
                     bundles=[re.sub(r"[^A-Za-z0-9]", "", n) for n in run_names], rois=True)


def visualize(out, stem, bundles, fodf=None, t1=None, rois=False):
    """The run's figures: a MIP collage of the bundles, a viewer page with all of them
    (densities and cores), and a page per bundle with its glyphs (and the fODF, if given);
    ``rois``: the pages also show each bundle's include / exclude ROIs.
    Failures are logged, not raised: the bundles are already on disk."""
    from .collage import write_collage
    from .core import bundle_files
    from .viewer import save_html

    jobs = [("collage", lambda: write_collage(out, f"{stem}_desc-mpat_collage.png", bundles, t1=t1)),
            ("viewer", lambda: save_html(f"{stem}_desc-mpat_view.html", out, bundles=bundles, rois=rois))]
    files = bundle_files(out)
    for b in bundles:
        if b in files:   # glyphs from 1e-2 of the peak (the slider can raise it); keeps pages small
            jobs.append((f"{b} viewer", lambda b=b: save_html(
                f"{files[b]['prefix']}_desc-view.html", out, bundles=[b], glyphs=True, fodf=fodf,
                glyph_range=1e-2, rois=rois)))
    for what, job in jobs:
        try:
            log.info("%s: %s", what, job())
        except Exception as err:  # noqa: BLE001
            log.warning("%s not written: %s", what, err)


def mpat_dir(afq, moments=False, out_dir=None):
    """Where MPAT's outputs go: ``out_dir`` if given, else <pyAFQ output_dir>/mpat (mpat-moments)."""
    if out_dir:
        return Path(out_dir)
    return Path(afq.kwargs["output_dir"]) / ("mpat-moments" if moments else "mpat")


GiB = 1024 ** 3
AUTO_FRACTION, AUTO_HEADROOM = 0.8, 10 * GiB    # max_mem="auto": min(80% of the limit, limit - 10 GB)


def memory_limit():
    """(bytes this process may use, where that came from): the smallest of its cgroup's
    memory limits (what SLURM and containers enforce), SLURM's request, and physical memory."""
    found = []
    try:
        for line in open("/proc/self/cgroup"):
            hid, ctrl, path = line.rstrip("\n").split(":", 2)
            if hid == "0" and ctrl == "":                       # cgroup v2
                files = ["memory.max"]
                root = "/sys/fs/cgroup"
            elif "memory" in ctrl.split(","):                   # cgroup v1
                files = ["memory.limit_in_bytes"]
                root = "/sys/fs/cgroup/memory"
            else:
                continue
            p = Path(path)
            for d in [p, *p.parents]:                          # the job's cgroup and its parents
                for f in files:
                    try:
                        v = (Path(root) / d.relative_to("/") / f).read_text().strip()
                    except OSError:
                        continue
                    if v.isdigit() and int(v) < 1 << 60:
                        found.append((int(v), f"cgroup {d}"))
    except OSError:
        pass
    env = os.environ
    if env.get("SLURM_MEM_PER_NODE", "").isdigit():
        found.append((int(env["SLURM_MEM_PER_NODE"]) * 1024 ** 2, "SLURM_MEM_PER_NODE"))
    elif env.get("SLURM_MEM_PER_CPU", "").isdigit():
        cpus = env.get("SLURM_CPUS_ON_NODE") or env.get("SLURM_CPUS_PER_TASK") or "1"
        if cpus.isdigit():
            found.append((int(env["SLURM_MEM_PER_CPU"]) * 1024 ** 2 * int(cpus), "SLURM_MEM_PER_CPU"))
    try:
        found.append((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "physical memory"))
    except (ValueError, OSError, AttributeError):
        pass
    return min(found) if found else (None, "unknown")


def memory_in_use():
    """Resident memory of this process (bytes; 0 if unknown)."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _budget(max_mem, chain, launches):
    """max_mem as segment_bundles takes it (bytes for the chain, the launches and the solve
    columns).  "auto": min(80% of the memory limit, limit - 10 GB), at least half the limit,
    less what the process already holds besides the chain and the launches (pyAFQ's data,
    the fODF, ...).  An explicit size is used as given, with a warning above the limit."""
    limit, src = memory_limit()
    held = memory_in_use()
    other = max(held - chain.nbytes - launches.nbytes, 0)
    if isinstance(max_mem, str) and max_mem.strip().lower() == "auto":
        if limit is None:
            raise ValueError("max_mem='auto': cannot tell this machine's memory limit; give a size")
        total = max(min(AUTO_FRACTION * limit, limit - AUTO_HEADROOM), 0.5 * limit)
        budget = total - other
        log.info("memory: limit %.1f GB (%s), MPAT may use %.1f GB; %.1f GB already held besides the chain "
                 "(%.1f GB) and launches (%.1f GB) -> %.1f GB budget", limit / GiB, src, total / GiB,
                 other / GiB, chain.nbytes / GiB, launches.nbytes / GiB, budget / GiB)
        return budget
    b = _bytes(max_mem)
    if limit is not None and b + other > limit:
        log.warning("max_mem %.1f GB plus the %.1f GB already held exceeds this job's memory limit "
                    "(%.1f GB, %s): it may be killed; try max_mem='auto'", b / GiB, other / GiB, limit / GiB, src)
    return b


def write_fixels(chain, sh, results, launches, path, affine, tol, max_mem, on_trapped=None,
                 lobes=None, whole=None):
    """The subject's fixels and each bundle's expected streamlines through them (mpat.fixels),
    as an MRtrix fixel directory at ``path``: ``<bundle>_streamlines``, ``all_streamlines``
    (the unconditioned chain: every walk from every seed, no ROIs), ``<bundle>_share`` (of
    all walks through the fixel) and ``<bundle>_share_of_bundles`` (of the solved bundles')."""
    from .fixels import fixel_streamlines, segment_lobes, write_fixel_directory

    t0 = time.time()
    ijk = np.stack(np.unravel_index(chain.t_lin, chain.pshape), 1) - 1
    rows = sh[tuple(ijk.T)]
    if lobes is None:
        lobes = segment_lobes(rows, chain.geo)
    if whole is None:
        whole = segment_bundles(chain, [dict(name="whole brain")], launches, tol=tol, max_mem=max_mem,
                                on_trapped=on_trapped)["whole brain"]
    n_all, _ = fixel_streamlines(whole.density_class, rows, chain.geo, lobes)
    data = {"all_streamlines": n_all}
    per = {re.sub(r"[^A-Za-z0-9]", "", name): fixel_streamlines(res.density_class, rows, chain.geo, lobes)[0]
           for name, res in results.items()}
    n_bundles = sum(per.values())
    frac = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)  # noqa: E731
    for key, n_b in per.items():
        data[f"{key}_streamlines"] = n_b
        data[f"{key}_share"] = frac(n_b, n_all)                    # of all walks through the fixel
        data[f"{key}_share_of_bundles"] = frac(n_b, n_bundles)     # of the solved bundles' walks
    write_fixel_directory(path, lobes, ijk, affine, chain.shape, data)
    log.info("fixels: %d in %d voxels, %d bundles -> %s (%.0fs)", lobes.n_fixels, len(ijk), len(results), path,
             time.time() - t0)
    return lobes, data


def _bytes(x):
    if isinstance(x, str):
        m = re.fullmatch(r"\s*([\d.]+)\s*([kmgt]?)b?\s*", x.lower())
        if not m:
            raise ValueError(f"cannot read memory size {x!r} (e.g. '16GB')")
        return float(m.group(1)) * 1024 ** "kmgt".find(m.group(2)) * (1024 if m.group(2) else 1)
    return float(x)


def run(afq, n_proc, max_mem="auto", bundles=None, moments=False, tol=1e-6, angle=35.0, double_odf=True,
        cores=True, viz=True, synthseg_thalamus=True, max_length=250.0, viz_rois=False,
        no_midline=(), midline_tol=None, out_dir=None, verbose=False, fornix=True, flux=False,
        fixels=False):
    """Segment pyAFQ's bundles with MPAT.

    afq         ``ParticipantAFQ`` or ``GroupAFQ`` (every participant in turn)
    n_proc      threads
    max_mem     memory budget, bytes, e.g. "16GB", or "auto" (default): sets how many bundles
                share each solve.  "auto" reads the job's limit (cgroup / SLURM / physical),
                keeps min(80%, limit - 10 GB) of it and subtracts what is already held
    bundles     pyAFQ bundle names (default: all in the bundle dictionary)
    moments     spatial first moments (narrower, more accurate bundles; ~5x slower)
    tol         remaining walker mass at which the solves stop (relative)
    angle       maximum turn per one-voxel hop (degrees)
    double_odf  weight each hop by the fODF where it lands (False: singleODF, like pyAFQ's
                first-order probabilistic tracker)
    cores       also write each bundle's ridge-path core (mpat.core)
    viz         also write a MIP collage of all bundles (PNG), a viewer page of all bundles
                (densities and cores), and one page per bundle with its visitation-ODF glyphs
                and the fODF (mpat.viewer.save_html).  Pages load NiiVue from a CDN.
    viz_rois    the viewer pages also show each bundle's include / exclude ROIs
    no_midline  bundle names whose cross_midline=False is ignored (no other-hemisphere exclusion)
    midline_tol mm past the midline the corticospinal tracts may go before the other hemisphere
                is excluded (default: their rule, 10; 0: a strict cut); every other one-sided bundle
                follows its own rule (mpat.bundles; default a strict cut at x = 0)
    max_length  soft maximum streamline length (mm; 0: none).  Before any bundle is solved,
                direction states from which most walkers would still be walking after this
                long -- traps where they circle, unable to turn out under the angle limit --
                become invalid ends for every bundle (mpat.solve.overlength_states); the
                survival per voxel is written to <stem>_desc-overlength_map.nii.gz
    out_dir     write MPAT's outputs here instead of <pyAFQ output_dir>/mpat (mpat-moments);
                pyAFQ's derivatives are only read.  For a GroupAFQ: <out_dir>/sub-<s>/ses-<t>
    fornix      the anterior thalamic radiations exclude the fornix / septum on the midline
                (FORNIX_BOX_MNI; False: as pyAFQ defines them)
    verbose     log timing inside the solves (ms per power step: matrix product / mask and
                sum); the time per stage is logged at the end anyway
    synthseg_thalamus  with pyAFQ's SynthSeg tissue (pve unset or "synthseg"), a template start / end
                ROI that is the thalamus is replaced by SynthSeg's thalamus (see bundle_specs)
    flux        also write each bundle's edge flux (mpat.fields): the expected streamline steps
                from every voxel into each of its 26 neighbours, and the flux through each
                voxel.  Not with moments.  The fields themselves are not kept here; use
                ``segment_bundles(keep_fields=True)`` to work with them.
    fixels      also write an MRtrix fixel directory, <stem>_desc-mpat_fixels/ (mpat.fixels):
                the subject's fixels from the fODF, and per fixel each bundle's expected
                streamlines through it (<bundle>_streamlines), the streamlines of the
                unconditioned chain (all_streamlines: every walk from every seed, no ROIs, one
                extra solve), the bundle's share of them (<bundle>_share) and its share of the
                solved bundles' streamlines (<bundle>_share_of_bundles).  For fixel-based
                analysis without tractography.
    Returns {bundle: BundleResult} (a dict per participant for a GroupAFQ)."""
    import numba

    if hasattr(afq, "pAFQ_list"):
        kw = dict(cores=cores, viz=viz, synthseg_thalamus=synthseg_thalamus, max_length=max_length,
                  viz_rois=viz_rois, no_midline=no_midline, midline_tol=midline_tol,
                  verbose=verbose, fornix=fornix, flux=flux, fixels=fixels)
        return {f"sub-{s}_ses-{t}": run(p, n_proc, max_mem, bundles, moments, tol, angle, double_odf, **kw,
                                        out_dir=out_dir and str(Path(out_dir) / f"sub-{s}" / f"ses-{t}"))
                for s, t, p in zip(afq.valid_sub_list, afq.valid_ses_list, afq.pAFQ_list)}
    numba.set_num_threads(max(1, min(int(n_proc), numba.config.NUMBA_NUM_THREADS)))
    t0 = time.time()
    clock = Clock()
    with clock("pyAFQ inputs"):
        sh_path = afq.export(f"{afq.export('tracking_params')['odf_model'].lower()}_params")
        sh_img = nib.load(sh_path)
        pve_img, bm_img = nib.load(afq.export("pve_internal")), nib.load(afq.export("brain_mask"))
        seed_img = nib.load(afq.export("wm_gm_interface"))
        # pyAFQ's tissue from SynthSeg: its thalami stand in for the template's thalamic ROIs
        thal = _thalami_if_synthseg(afq, sh_img, synthseg_thalamus)
    out = mpat_dir(afq, moments, out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = str(out / Path(afq.kwargs["base_fname"]).name)

    def specs(transient):
        # each bundle's ROIs warped just before its group is solved, its ROI masks written then
        # (the solve keeps what it needs, so the ROIs can go)
        yield from roi_specs(afq, sh_img, transient, stem, bundles, thal, no_midline, midline_tol, fornix)

    results = _segment(sh_img, pve_img, bm_img, seed_img, specs, stem, clock, max_mem=max_mem, moments=moments,
                       tol=tol, angle=angle, double_odf=double_odf, cores=cores, max_length=max_length,
                       verbose=verbose, flux=flux, fixels=fixels)
    t_viz = time.perf_counter()
    if viz:
        try:
            t1 = afq.export("t1_masked")
        except Exception:  # noqa: BLE001 -- the collage falls back to the tissue labels
            t1 = None
        visualize(out, stem, [re.sub(r"[^A-Za-z0-9]", "", n) for n in results], sh_path, t1=t1, rois=viz_rois)
        clock.add("viz", time.perf_counter() - t_viz)
    clock.report("MPAT time per stage", time.time() - t0)
    log.info("MPAT: %d bundles in %.0fs -> %s", len(results), time.time() - t0, out)
    return results


def _segment(sh_img, pve_img, bm_img, seed_img, specs, stem, clock, max_mem="auto", moments=False, tol=1e-6,
             angle=35.0, double_odf=True, cores=True, max_length=250.0, verbose=False, flux=False, fixels=False):
    """run's chain and solves, from images: tissue labels, the chain, seeds (``seed_img`` within
    the walkable white matter), the bundles of ``specs(transient)``, and every per-bundle
    output, the table and the cores written at ``stem``.  Returns {bundle: BundleResult}."""
    with clock("tissue labels"):
        transient, absorb, dseg = tissue_labels(sh_img, pve_img, bm_img)
    nib.save(nib.Nifti1Image(dseg, sh_img.affine), f"{stem}_desc-mpat_dseg.nii.gz")
    out = Path(stem).parent
    cfg = ChainConfig(angle=angle, double_odf=double_odf, moments=moments)
    with clock("fODF load"):
        sh = np.asarray(sh_img.dataobj, np.float32)
    with clock("chain build / cache"):
        chain = Chain.build(sh, sh_img.header.get_zooms()[:3], transient, absorb, cfg,
                            cache=f"{stem}_desc-chain_cache.npz")
    seeds = np.argwhere((_on_grid(seed_img, sh_img, 0) > 0) & transient)
    t_ol = time.perf_counter()
    if max_length and max_length > 0:
        t1_ = time.time()
        kill, surv, T = overlength_states(chain, max_length, sh_img.header.get_zooms()[:3])
        vox = surv.reshape(-1, chain.nb)
        nib.save(nib.Nifti1Image(chain.to_volume(vox.max(1)), sh_img.affine), f"{stem}_desc-overlength_map.nii.gz")
        chain.kill = kill if kill.any() else None
        log.info("soft maximum streamline length %.0f mm (%d steps): %d of %d direction states, in %d voxels, "
                 "keep most walkers longer than that and end them invalidly (%.0fs)", max_length, T,
                 kill.sum(), kill.size, int(kill.reshape(-1, chain.nb).any(1).sum()), time.time() - t1_)
    clock.add("soft max length", time.perf_counter() - t_ol)
    with clock("seed launches (prepare)"):
        launches = prepare_launches(chain, seeds)
    chain.release_pmfs()
    rows = []

    n_trapped = [0]

    def trapped(what, names, vol):
        # where the mass a stalled solve could not finish sits: open it over the T1
        n_trapped[0] += 1
        f = f"{stem}_desc-trapped{what}{n_trapped[0]}_map.nii.gz"
        nib.save(nib.Nifti1Image(vol, sh_img.affine), f)
        peak = nib.affines.apply_affine(sh_img.affine, np.unravel_index(np.argmax(vol), vol.shape))
        log.warning("%s solve of %s did not finish: the mass left (%.1e of the start) is mapped in %s; "
                    "most of it at (%.0f, %.0f, %.0f) mm", what, ", ".join(names), vol.sum(), f, *peak)

    budget = _budget(max_mem, chain, launches)
    from .fixels import segment_lobes
    from .quantify import quantify_bundle

    t_ijk = np.stack(np.unravel_index(chain.t_lin, chain.pshape), 1) - 1
    sh_rows = sh[tuple(t_ijk.T)]
    with clock("whole-brain reference & lobes"):
        lobes = segment_lobes(sh_rows, chain.geo)
        whole = segment_bundles(chain, [dict(name="whole brain")], launches, tol=tol, max_mem=budget,
                                on_trapped=trapped)["whole brain"]

    def write(res):
        name = re.sub(r"[^A-Za-z0-9]", "", res.name)
        b = f"{stem}_bundle-{name}"
        sp = np.zeros(chain.shape, np.float32)
        np.add.at(sp, tuple(res.seed_vox.T), res.seed_prob)
        maps = dict(seedprob=(sp, "probmap"), density=(chain.to_volume(res.density_t), "map"),
                    endpoints=(chain.to_volume(res.endpoints_a, "absorbing"), "map"))
        d = res.density_t
        keep = np.flatnonzero(d >= ODF_FLOOR * d.max()) if d.max() > 0 else np.zeros(0, np.int64)
        ijk = tuple((np.stack(np.unravel_index(chain.t_lin[keep], chain.pshape), 1) - 1).T)
        coef = visitation_odf(res.density_class[keep], sh[ijk], chain.geo)
        odf = np.zeros(chain.shape + coef.shape[1:], np.float32)
        odf[ijk] = coef
        maps["visitodf"] = (odf, "dwimap")
        qb = None
        if getattr(res, "flux26", None) is not None or res.fields is not None:
            qb = quantify_bundle(chain, res, whole.density_class, sh_rows, sh_img.affine, lobes=lobes,
                                 compute_boundary_conductance=False)
            maps["afd"] = (chain.to_volume(qb.voxel_afd_t), "map")
            maps["tdimm"] = (chain.to_volume(qb.tdi_mm_t), "map")
            maps["qharmonic"] = (chain.to_volume(qb.q_harmonic_t), "map")
            maps["coherence"] = (chain.to_volume(qb.flux_coherence_t), "map")
            maps["keff"] = (chain.to_volume(qb.branching_k_eff_t), "map")
            res.info["quantification"] = qb.summary
        res.start_mask = None
        res.end_mask = None
        if res.fields is not None:
            f26 = res.fields.edge_flux()
            maps["throughflux"] = (chain.to_volume(res.fields.through_flux(f26)), "map")
            f26vol = np.zeros(chain.shape + (26,), np.float32)
            f26vol[tuple((np.stack(np.unravel_index(chain.t_lin, chain.pshape), 1) - 1).T)] = f26
            maps["edgeflux"] = (f26vol, "map")
            res.info["flux_conservation"] = {k: v for k, v in res.fields.conservation(f26).items()
                                             if not isinstance(v, np.ndarray)}
            res.fields = None
        for desc, (vol, suffix) in maps.items():
            nib.save(nib.Nifti1Image(vol, sh_img.affine), f"{b}_desc-{desc}_{suffix}.nii.gz")
        with open(f"{b}_desc-summary.json", "w") as fh:
            json.dump(dict(bundle=res.name, config=cfg.key(), result=res.info,
                           valid_ends=[ABSORB_CLASSES[c] for c in (1, 3)],
                           density="expected steps of bundle streamlines in each voxel, all seeds",
                           afd="AFD-calibrated intra-axonal bundle density per voxel (steradians, analytical SIFT2)",
                           tdimm="expected physical chord length traversed per voxel (mm)",
                           qharmonic="harmonic committor coordinate in [0, 1] along the reactive flux graph",
                           coherence="reactive flux directional coherence in [0, 1] per voxel",
                           keff="effective outgoing neighbour branching exp(H) in [1, 26] per voxel",
                           endpoints="expected bundle endpoints per voxel (two per streamline)",
                           visitodf="the walkers' directions per voxel: SH (descoteaux07, legacy, "
                                    "lmax 8), expected visits per steradian",
                           seedprob="P(the streamline seeded here belongs to the bundle)",
                           **({"throughflux": "expected bundle-streamline steps leaving each voxel",
                               "edgeflux": "the same per neighbour: 26 volumes, mpat.geometry.OFFSETS order"}
                              if flux else {})), fh, indent=1)
        i = res.info
        row = dict(bundle=res.name, expected_streamlines=round(i["expected_bundle_streamlines"], 4))
        if qb is not None:
            row.update(
                mean_length_mm=qb.summary["mean_streamline_length_mm"],
                afd_volume_mm3=qb.summary["tract_afd_volume_mm3"],
                afd_area_mm2=qb.summary["effective_afd_area_mm2"],
                median_cut_flux=qb.summary["median_cut_flux"],
            )
        row.update(columns=i["n_columns"], backward_steps=i["backward"]["iterations"],
                   forward_steps=i["forward"]["iterations"], seconds=i["seconds"])
        rows.append(row)
        log.info("%s: %.2f expected streamlines", res.name, i["expected_bundle_streamlines"])

    results = segment_bundles(chain, specs(transient), launches, tol=tol, max_mem=budget, on_result=write,
                              on_trapped=trapped, verbose=verbose, clock=clock, keep_fields=flux)
    if fixels:
        with clock("fixels"):
            write_fixels(chain, sh, results, launches, f"{stem}_desc-mpat_fixels", sh_img.affine, tol, budget,
                         on_trapped=trapped, lobes=lobes, whole=whole)
    with open(f"{stem}_desc-mpat_table.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    if cores:
        from .core import write_cores
        with clock("cores"):
            write_cores(out)
    return results


def run_from_images(fodf, tissue, brain_mask, seeds, bundles, out_dir, n_proc, max_mem="auto", tol=1e-6,
                    angle=35.0, double_odf=True, max_length=250.0, viz=True, t1=None):
    """Segment bundles with MPAT from image files alone (tissue and masks may also be nibabel
    images), without pyAFQ.

    fodf        WM fODF file: SH, DIPY descoteaux07 (legacy), directions in the image's voxel axes
                (mpat.ss3t.fit_ss3t writes one; so does pyAFQ's CSD).  MPAT works on its grid
    tissue      anatomical tissue probabilities: a 4-D image of (CSF, GM, WM), or three images
                in that order
    brain_mask, seeds  masks, e.g. the white / grey-matter interface as seeds
    bundles     [{"name", "include": [masks], "exclude": mask, "start": mask, "end": mask}],
                masks used as given (nothing grown, sealed or added)
    t1          optional background for the collage
    The other arguments are run's.  Outputs: <out_dir>/<fODF file name up to "_model-">_*, as
    run's.  Returns {bundle: BundleResult}."""
    import numba

    numba.set_num_threads(max(1, min(int(n_proc), numba.config.NUMBA_NUM_THREADS)))
    t0 = time.time()
    clock = Clock()
    load = lambda x: nib.load(x) if isinstance(x, (str, Path)) else x  # noqa: E731
    sh_img = load(fodf)
    if isinstance(tissue, (list, tuple)):
        tissue = nib.Nifti1Image(np.stack([_on_grid(load(t), sh_img) for t in tissue], -1), sh_img.affine)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = str(out / re.split(r"_model-|\.nii", Path(sh_img.get_filename()).name)[0])

    def specs(transient):
        mask = lambda m: _on_grid(load(m), sh_img, order=0) > 0  # noqa: E731
        for b in bundles:
            yield dict(name=b["name"], include=[mask(m) for m in b.get("include", [])],
                       **{k: mask(b[k]) for k in ("exclude", "start", "end") if b.get(k) is not None})

    results = _segment(sh_img, load(tissue), load(brain_mask), load(seeds), specs, stem, clock, max_mem=max_mem,
                       tol=tol, angle=angle, double_odf=double_odf, max_length=max_length)
    if viz:
        t_viz = time.perf_counter()
        visualize(str(out), stem, [re.sub(r"[^A-Za-z0-9]", "", n) for n in results], sh_img.get_filename(), t1=t1)
        clock.add("viz", time.perf_counter() - t_viz)
    clock.report("MPAT time per stage", time.time() - t0)
    log.info("MPAT: %d bundles in %.0fs -> %s", len(results), time.time() - t0, out)
    return results
