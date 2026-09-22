# Unified single radiance cache — design

**Status:** design approved in conversation 2026-09-22; implementation plan not yet written.

## Goal

Replace the BDRC mode's three-network factorized system (`eyeNrc` colour cache,
`lightNrc` colour cache, `maskNrc` density cache) with **one** neural field
trained by a single weighted density-ratio objective over two supervision
streams. The field stores incident flux per unit world area on diffuse
receivers; outgoing radiance is formed by applying the receiver BSDF at query.

This is the "unified transport cache" architecture from the sibling Python
project (`bidirectional-radiance-caching`), whose live, validated
implementation is the WebGPU demo at `third_party/neural-cache-demo/`. The
Mitsuba/Python unified presets were retired in that project's revision
`ef4cc61c` because its inverse-rendering recoveries run the decomposed
Eye/Light pair; the retirement did not withdraw the architecture, and the demo
plan (`docs/plans/neural-cache-demo.md` there) explicitly selects "one unified
primal cache with cached-tail supervision" for forward rendering, which is what
this renderer does.

## Why one network

The two streams are both positive streams of one density-ratio problem against
a common surface reference `p_r`. For any Bregman generator the fitted field
satisfies `U* = sum_s p_s * w_bar_s` — the sum over streams of endpoint density
times mean weight. Disjoint path families induce measures that add, so one
network supervised by both memorises their sum. Neither stream needs the
other's density, and the mask/density network is unnecessary because the
reference anchors fix the absolute scale directly.

Decomposition, with `S` any vertex without a diffuse lobe:

```
U(x) = C(x) + T(x)

C(x)  caustic term: flux per area deposited at x by emitter walks whose
      vertices before x are all non-diffuse and number at least one (L S+ D)

T(x)  everything else, as one gather segment:
      T(x) = E_dir(x) + integral f_s(y) U(y) |cos| dw
```

`E_dir` is direct illumination by emitter sampling; the integral samples one
direction at `x`, continues through any non-diffuse chain to the first diffuse
vertex `y`, and reads this same network's EMA there. The three families
`L -> D`, `L -> S+ -> D` and `L -> ... -> D -> S* -> D` are disjoint and
complete, and the recursion through `U(y)` carries every longer path.

Source of truth for the derivation: the sibling project's
`docs/concepts/unified-transport-cache.md` and
`docs/concepts/path-stream-sums.md`. Source of truth for the implementation:
`third_party/neural-cache-demo/src/renderer/shaders/transport.wgsl` and
`src/cache/shaders/dre_loss.wgsl` there.

## Decisions taken

| Decision | Choice | Reason |
|---|---|---|
| Integration | Replace BDRC in place | Smallest end state, no dead code; the old design stays reachable in git |
| Coordinates | Keep AABB-normalized world position | The surface reference table, photon deposits and debug views already live in this space; contracted-NDC is a separate later change requiring a fresh fit |
| Generators | Full family, log-space | logistic / brier / lsif / bregman at arbitrary power; ~30 lines in one kernel, and the demo's scene recipes disagree on which is best |
| Labels | Both, 5-dim input | Required once gathers start on unlit backfaces and closed-solid interiors |

## Architecture

### The network

One `unified_ngp_cache`, replacing `mask_ngp_cache`, derived from
`base_ngp_cache` (which keeps its optimizer, EMA, gradient-scaler and
weight-clip roles unchanged).

- **Output:** 3 channels, softplus activation, interpreted as incident flux per
  unit world area. Not radiance, not outgoing, no BSDF baked in.
- **Input:** 5 dims. Dims 0–2 are the AABB-normalized position; dims 3–4 are
  two scalar labels. The encoding is a tcnn `Composite` with `Concatenation`
  reduction and two nested entries:

  ```json
  {"otype": "Composite", "nested": [
    {"otype": "HashGrid", "n_dims_to_encode": 3, ...},
    {"otype": "Identity", "n_dims_to_encode": 2}
  ]}
  ```

  Confirmed supported by the vendored tiny-cuda-nn
  (`3rdparty/tiny-cuda-nn/include/tiny-cuda-nn/encodings/composite.h`, which
  accepts `n_dims_to_encode` and `dims_to_encode_begin` per nested encoding).
  Labels bypass the hash grid and enter the MLP directly, matching the demo,
  where `field.wgsl` appends them after the grid features.
- **Labels:** `sign_cam` (geometric face orientation relative to a **fixed
  reference origin**) and arrival side.

  The reference origin is captured once when the cache is constructed and
  stamped with it; it is **not** the live viewing camera. This renderer's
  camera is interactive, and a label taken against the live camera would flip
  for every point the user orbits past, silently relabelling the whole learned
  function mid-fit. The demo avoids this by construction because its authored
  camera owns the mapping and reference distribution while the viewing camera
  only changes rays and resets film accumulation; this port has no authored
  camera, so the fixed origin has to be explicit. Moving the viewing camera
  must change rays and film accumulation only — never training coordinates,
  labels, reference densities or learned weights.

  The field holds a value per (point, side): anchors
  take either side with probability 1/2 at half the area density, deposits and
  gathers use the same halved density, and the interior of a closed solid is a
  side of its own that fits to zero. Layer and side stay separate inputs; their
  product would give a thin shell's dark inner back layer the label of its lit
  outer front.

### Supervision streams

One records array per update. Lanes `[0, photonCount)` are photons; lanes
`[photonCount, photonCount + gatherCount)` are gathers. Each lane carries an
RGB positive weight, a reference density, and the two per-stream normalizers.

**Photon stream** — already implemented as `bdrcLightIntegrator`'s ungated
`L S+ D` grammar (one deposit per path, no shadow-ray gating, no dependence on
the camera). Changes:

- Store **raw clamped photon power**. The uncommitted `bsdfEvalTowardCamera`
  work that bakes the receiver BSDF toward the camera into the target is
  dropped: under the incident convention the receiver BSDF belongs at the
  query, not in the stored field.
- Reference density is the surface table's `invTotalArea`, already written.
- Positive coefficient `beta * w * (1/photonCount)`; reference coefficient `0`.
- The photon stream has no anchor pairing available and works as a measure or
  not at all.

**Gather stream** — new. A dedicated raygen program and SBT, launched over
`gatherCount` lanes, mirroring the existing light-path launch pattern
(`__raygen__light_path_tracer` with `m_sbtLightPath`). Per lane:

1. Draw an anchor from the existing global diffuse-surface CDF table
   (`sampleSurfaceUniform`), then pick a side with probability 1/2.
2. Estimate direct illumination there with a **unit BSDF**, so the estimate is
   incident flux rather than outgoing radiance. The demo achieves this by
   setting the receiver's `kd = pi` before calling `estimate_direct`.
3. Sample a cosine hemisphere, trace to the next diffuse vertex `y`, and store
   `tailPoint`, `tailNormal` and `factor = throughput * kd/pi` at `y`.
4. The anchor is both the reference anchor and the gather start: one query of
   the field serves the lane's reference term and its positive term, which
   makes the gathered half of the fit a regression at known points.
   Independent draws reach the same stationary point with more variance.

Positive coefficient `beta * w * (1/gatherCount)` with
`w = (E_dir + factor * U_EMA(y)) / p_r`; reference coefficient
`alpha * (1/gatherCount)`.

**Camera stream** — the existing eye path, structurally unchanged: trace the
prefix, take ordinary NEE with MIS at non-diffuse vertices, stop at the first
diffuse vertex, and make one query there. The pixel reads
`prefix + throughput * f_s(x) * U(x)`.

### Cached tail

The tail read is a host-side two-pass, the pattern the existing
virtual-emission fold-in already established in `bdrc_host.cu`:

```
raygen writes tailPoint/tailNormal
  -> pack tail queries (position + labels)
  -> EMA inference over tails
  -> assembleTargets kernel builds per-lane records
  -> DRE loss + backward
```

The tail read uses **EMA (teacher) weights**; the fit updates the live
(student) weights. A bootstrapped estimator that ends by reading the field's
own EMA is unbiased for the measure induced by the current field, and the true
solution is the fixed point of the iteration.

### Loss

One kernel replacing `k_dre_logistic_grad`, keeping its shape: analytic
`dL/dOutput` written into tcnn's buffer, then `m_network->backward` with a
`GradientMode`. Differences: 3 channels instead of 1, log-space evaluation, and
per-lane `(positive, reference)` coefficients instead of a `label` switch.

Per lane, per channel, with `u` the softplus density and `q` the lane's
reference density:

| Generator | positive term | reference term |
|---|---|---|
| logistic | `softplus(-logit)` | `softplus(logit)` |
| brier | `sigmoid(-logit)^2` | `sigmoid(logit)^2` |
| lsif | `-r` | `r^2 / 2` |
| bregman(a) | `-(r^(a-1) - 1)/(a-1)` | `(r^a - 1)/a` |

with `logit = log u + log beta - log q - log alpha` and `r = u/q`. Limits at
`a = 0` and `a = 1` are handled separately, as in the demo's `terms()`.
Gradients are taken with respect to `log u` and chained through softplus.

Both streams accumulate into **one** optimizer step through the existing
`accumGrad` (Overwrite on the first, Accumulate on the rest) then
`optimizerStep` exactly once, as `mask_ngp_cache` already does.

### Per-frame dispatch

```
optixLaunch  eye paths        -> camera records (prefix, throughput, query point, labels)
optixLaunch  photon paths     -> photon records
optixLaunch  gather paths     -> gather records (anchor, direct, tail, factor)
pack queries (starts)         -> positions + labels for every lane
pack queries (tails)          -> positions + labels for gather tails
EMA inference over tails
assembleTargets               -> per-lane (RGB weight, p_r, 1/N_pos, 1/N_ref)
accumGrad over the records    -> one optimizer step -> EMA update
inference over camera queries -> composite: prefix + throughput * f_s * U
```

There is no two-phase warmup: the single fit runs from frame 0.

### Removed

`eyeNrc`, `lightNrc`, `mask_ngp_cache`'s scalar-density role, `m_bdrcPhase` and
`m_bdrcWarmup` with the two-phase schedule, the virtual-emission buffers and
fold-in (`virtEmitPos`/`virtEmitThroughput`/`virtEmitValid` and their
compaction/scatter-add kernels), the 14-dim NRC feature packing on the BDRC
path, and the suffix-training machinery on the BDRC path
(`trainNextFeatures`, `trainSuffixScaled`, `trainThroughputAtK`,
`trainNextValid`). The NRC mode (`m_nrc == 1`) keeps all of its own copies and
is not modified.

## Numerical contracts

These are the properties the implementation must not break; each is a
consequence of the derivation, not a tuning preference.

1. **The objective stays plain.** No per-stream reweighting: no
   self-normalized weights, no learned global scale, no proxy mixture, no
   detached reweighting. Any of these rescales one stream's mean weight against
   the other's and moves the fixed point off the sum.
2. **Raw-ratio members require `alpha == beta`.** The posterior forms
   (logistic, brier) have a fixed point independent of `alpha`; the raw-ratio
   forms (lsif, bregman) read `u* = (beta/alpha) * sum_s p_s w_bar_s`. The demo
   rejects unequal constants for those members and so must this port.
3. **`alpha` is the noise ratio and matters.** The field is 10^2–10^4 times the
   reference density, so at `alpha = 1` the posterior saturates and the shared
   head weighs the brightest regions least. Expose it per recipe.
4. **Each stream is normalized by its own draw count**, not by the live count
   of valid samples in the frame. A frame with fewer valid samples must not
   implicitly renormalize its gradient up. This is what `fixedNormWidth`
   already does for the colour caches.
5. **Reference density is per unit world surface area** at both the anchor and
   the deposit, evaluated from the same table. Anchors and deposits on a
   two-sided field carry half the area density.
6. **EMA teacher, live student.** The tail read, and any reference or
   diagnostic query, uses EMA weights; only the fit itself touches live
   weights.
7. **The stored quantity is incident flux per area.** The receiver BSDF is
   applied at the query, in the renderer, on every path that reads the field —
   camera composite, gather tail and any debug view alike.
8. **The cache lives on diffuse lobes only.** Glossy-only and delta surfaces
   are transport vertices: every walk samples through them, and they take no
   deposit, gather start, reference mass or query.
9. **Training is camera-independent.** Moving the viewing camera changes rays
   and film accumulation only. Training coordinates, labels, reference
   densities and learned weights do not depend on the live camera; the
   `sign_cam` reference origin is fixed at construction and stamped.

## Verification

This repository has no unit-test framework, and the plan must not add one.
Every gate is a build plus an observable at runtime.

- **Build:** `export CUDACXX=/usr/local/cuda/bin/nvcc && cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc)`
- **Smoke scenes:** `data/cornell_caustic_glass_sphere.glb` (simple caustic)
  and `data/cornell_caustic_glass_sphere_sds.glb` (specular-diffuse-specular).
- **Stream-count gate:** console counts for photon and gather lanes are both
  nonzero and stable frame to frame, and the gather count is independent of
  camera framing.
- **Reference-density gate:** the per-lane `p_r` printed for anchors and
  deposits is the same positive constant, consistent with
  `1 / totalSceneSurfaceArea` (halved for the two-sided field).
- **Fit gate:** the loss decreases and stabilises rather than diverging or
  never moving, and the cache-only debug view shows a plausible spatial
  pattern, not a uniformly zero or saturated field.
- **Transport gate:** the caustic under the glass sphere appears in the
  cache-only view, and the SDS scene does not go darker than the simple scene's
  equivalent region.
- **Camera-independence gate:** orbit the camera during a fit and confirm the
  cache-only view stays consistent with what was already learned. A fit that
  visibly degrades or re-converges on camera motion means a label, coordinate
  or reference density is reading the live camera.
- **Regression expectation:** image quality is expected to be *worse* than the
  current three-cache BDRC until the gather stream is correct. A worse image at
  an intermediate task is not by itself evidence of a bug; the stream-count,
  reference-density and fit gates localise where to look.

## Risks

- **Shading state at a non-hit point.** The anchor is a triangle plus
  barycentrics, not a ray hit. Building a surface interaction there, and
  evaluating direct lighting with a unit BSDF, is the largest single
  implementation risk in the port. The demo constructs its `Surface` directly
  from the triangle; this renderer's shading state is assembled in `hit.cu`
  from the SBT hit record and is not currently reachable from a raygen program
  without a hit.
- **Half-precision fragility.** `mask_ngp_cache` documents reproducible NaN by
  iteration ~400 without `gradScaler` plus `weightClip = 1.0`. The new loss is
  log-space and better behaved for the posterior members, but the raw-ratio
  members can be heavy-tailed when `u/q` spans orders of magnitude. Carry the
  existing guards over unchanged and treat learning rate as per-generator.
- **Loss of a working baseline.** Replacing in place means the render path
  depends on the new fit from the first commit that removes the colour caches.
  The plan should sequence so the build stays green and the old path keeps
  rendering until the unified path produces an image.
- **Stale cross-project references.** The completed parity plan cites Python
  paths (`cpm/losses/density_ratio.py`, `cpm/scene/steady/...`) that no longer
  exist; that project reorganised into `src/core/` and `src/apps/`. Resolve
  every cross-project reference against the demo or the current tree, not
  against that plan.

## Out of scope

Contracted-NDC-v2 coordinates; the decomposed Eye/Light pair; any adjoint or
inverse-rendering work; the NRC mode; checkpoint format changes beyond what the
new contract requires; shape-balanced or NDC-projected-area reference mixtures
(the pure area-uniform reference is the documented default and what the surface
table already provides).
