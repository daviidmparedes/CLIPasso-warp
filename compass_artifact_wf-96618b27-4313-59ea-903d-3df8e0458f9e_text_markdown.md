# CLIPasso: A Deep Technical Explanation of Semantically-Aware Object Sketching

## TL;DR

- **CLIPasso is test-time optimization, not a trained model:** it represents a sketch as *n* cubic Bézier curves (each defined by 4 control points), renders them with the differentiable rasterizer **diffvg**, and directly optimizes the control-point coordinates with **Adam (lr = 1.0)** against a frozen CLIP-based perceptual loss until convergence (~2,000 iterations, ~6 min on a Tesla V100). There is no network to train and no sketch dataset — the abstraction level is a knob set by the number of strokes *n* (defaults: 16; the paper studies 4/8/16/32).
- **The loss is the core idea:** a **geometric** term (L2 between intermediate CLIP **RN101** ResNet activations, layers 3 and 4, weights `0,0,1.0,1.0,0`) grounds the sketch spatially, while a **semantic** term (cosine distance between the final CLIP image embeddings of sketch and target, weight `w_s = 0.1`) enforces conceptual fidelity. Both are computed over augmented views (RandomPerspective + RandomResizedCrop applied to both sketch and target) to prevent adversarial/degenerate solutions.
- **Initialization matters because the problem is highly non-convex:** initial stroke locations are sampled from a saliency distribution built by taking CLIP **ViT-B/32** self-attention, aggregating it via the Chefer et al. relevancy method (no text supervision), multiplying by an **XDoG** edge map, and softmax-normalizing. Three seeds run in parallel and the lowest-loss result is selected. Practically, the code is brittle: it targets Python 3.7 / PyTorch 1.7.1+cu101 and depends on diffvg, which is notorious for build failures; the Replicate deployment is currently unrunnable due to an outdated Cog/Python.

---

## Key Findings

1. **Problem formulation.** A sketch is a set of *n* black strokes on a white 224×224 canvas. Each stroke is one cubic Bézier segment with 4 control points $\{p_i^j\}_{i=1}^4$, $p_i^j = (x_i,y_i)\in[0,224]^2$. The free parameters are exactly the control-point coordinates — the code freezes color to black `[0,0,0,1]`, keeps stroke width fixed (default 1.5), and by default does **not** optimize width or opacity. Abstraction is controlled by two integers: number of strokes `--num_paths`/`--num_strokes` (default 16) and control points per segment (`--control_points_per_seg`, default 4) × number of segments (`--num_segments`, default 1). This is per-image (test-time) optimization: `loss.backward()` flows gradients into the raw control-point tensors.

2. **diffvg makes rasterization differentiable.** Naïve rasterization is piecewise-constant in the control points (a pixel is either covered by a stroke or not), so $\partial(\text{pixel})/\partial(\text{control point})$ is 0 almost everywhere and undefined at edges. diffvg (Li et al., SIGGRAPH Asia 2020) makes the rendering differentiable via **pixel prefiltering**: the pixel color is the convolution of an antialiasing kernel with the scene function, which is continuous in the geometry, so its derivative exists. diffvg offers analytic prefiltering (fast, but can suffer conflation artifacts) and **multisampling antialiasing** (the CLIPasso path). It supplies a PyTorch `autograd.Function` (`pydiffvg.RenderFunction`) whose forward rasterizes and whose backward returns gradients w.r.t. control points, stroke width, and color.

3. **The CLIP-based perceptual loss is the mechanism that removes the need for sketch data.** $L = L_{geometric} + w_s\,L_{semantic}$, with $w_s = 0.1$. $L_{semantic} = D_{cos}(\text{CLIP}(S), \text{CLIP}(I)) = 1 - \cos(\text{CLIP}(S),\text{CLIP}(I))$ on the final embeddings; $L_{geometric} = \sum_{\ell\in\{3,4\}} \|\text{CLIP}_\ell(S) - \text{CLIP}_\ell(I)\|_2^2$ on intermediate ResNet-101 activations. Intermediate CNN activations retain spatial layout (so they enforce where the object's parts are), while the final projected embedding is a global, spatially-collapsed semantic vector (so it enforces what the object is). This is why omitting $L_{semantic}$ yields geometrically faithful but semantically weak contours, and omitting $L_{geometric}$ yields recognizable-class but poorly-localized, instance-unfaithful sketches.

4. **Saliency-guided initialization + multi-seed selection** converts a hard non-convex search into a reliable pipeline. Chefer et al.-style relevancy over ViT-B/32 attention, multiplied by an XDoG edge map and softmax-normalized (temperature `softmax_temp = 0.3`), gives a probability map from which the first control point of each stroke is sampled; the other three are sampled uniformly within radius 0.05 (normalized) around it. Three seeds (0, 1000, 2000) run and the sketch with the lowest evaluation loss is copied to `*_best.svg`.

5. **Evaluation confirms graceful degradation with a "breaking point" at 4 strokes.** A user study with **121 participants** and automatic classifier studies (ResNet34, CLIP ViT-B/32 zero-shot) show recognizability holding up well down to 8 strokes and degrading sharply at 4.

---

## Details

### 1. Repository / pipeline structure

The runnable entry point is **`run_object_sketching.py`**. It:
- Downloads the U²-Net weight file `U2Net_/saved_models/u2net.pth` via `gdown` if missing.
- Builds `seeds = list(range(0, num_sketches*1000, 1000))` — so with the default `num_sketches = 3`, seeds are **0, 1000, 2000**.
- For each seed, spawns `python painterly_rendering.py <target> --num_paths <n> --num_iter <2001> --seed <seed> ...` either sequentially or with a multiprocessing `Pool(ncpus=10)` when `--multiprocess` is set (Colab uses sequential).
- After all seeds finish, loads each run's saved `config.npy`, reads the `loss_eval` array, takes its minimum, sorts runs by that minimum, and `copyfile`s the best run's `best_iter.svg` to `<name>_best.svg`. This is the **best-seed selection**.

Default CLI args (from `run_object_sketching.py`): `--num_strokes 16`, `--num_iter 2001`, `--num_sketches 3`, `--fix_scale 0`, `--mask_object 0`, `save_interval = 10`.

**`painterly_rendering.py`** contains the optimization loop `main(args)`:
- `loss_func = Loss(args)`; `inputs, mask = get_target(args)`; `renderer = load_renderer(...)`; `optimizer = PainterOptimizer(args, renderer)`.
- `renderer.set_random_noise(0)`, `img = renderer.init_image(stage=0)`, `optimizer.init_optimizers()`.
- Loop over `args.num_iter`: `optimizer.zero_grad_()` → `sketches = renderer.get_image()` → `losses_dict = loss_func(sketches, inputs.detach(), ...)` → `loss = sum(losses_dict.values())` → `loss.backward()` → `optimizer.step_()`.
- Every `save_interval` (10) iterations it dumps a JPG and SVG log; every `eval_interval` (10) it recomputes the loss in a `torch.no_grad()` "eval" mode (augmentations off) to get a clean `loss_eval` and tracks `best_loss`/`best_iter`, saving `best_iter.svg`.
- **Convergence criterion:** `min_delta = 1e-5`. When `abs(loss_eval - best_loss) <= min_delta` on two consecutive evaluations (the `terminate` flag), the loop breaks early. This is the "loss delta < 0.00001, evaluated every 10 iterations" rule.

`get_target(args)`: opens the image, composites RGBA onto white, optionally masks the object via `utils.get_mask_u2net` (if `--mask_object`), optionally fixes non-square scale, resizes to `image_scale = 224` (BICUBIC), center-crops, and `ToTensor`s to a `[1,3,224,224]` tensor.

**`config.py`** holds all defaults (verbatim): `--lr 1.0`, `--color_lr 0.01`, `--image_scale 224`, `--num_paths 16`, `--width 1.5`, `--control_points_per_seg 4`, `--num_segments 1`, `--attention_init 1`, `--saliency_model clip`, `--saliency_clip_model ViT-B/32`, `--xdog_intersec 1`, `--softmax_temp 0.3`, `--num_aug_clip 4`, `--augment_both 1`, `--augemntations affine`, `--aug_scale_min 0.7`, `--clip_conv_loss 1`, `--clip_conv_loss_type L2`, `--clip_conv_layer_weights 0,0,1.0,1.0,0`, `--clip_model_name RN101`, `--clip_fc_loss_weight 0.1`, `--clip_text_guide 0`, `--num_iter 500` (overridden to 2001 by the runner). `set_seed()` seeds `random`, `numpy`, `PYTHONHASHSEED`, and all torch RNGs. It also calls `pydiffvg.set_use_gpu(...)` and `pydiffvg.set_device(...)`.

### 2. Stroke parameterization and rendering (`models/painter_params.py`)

The `Painter(torch.nn.Module)` class holds the vector scene. Its key methods (confirmed from the closely-derived SwiftSketch/ControlSketch fork, which reuses CLIPasso's code near-verbatim):

- **`get_path()`** builds one stroke. It sets `num_control_points = zeros(num_segments) + (control_points_per_seg - 2)` (= `[2]` for a single cubic segment, since a cubic Bézier has 2 interior control points plus shared endpoints). The first point `p0` is taken from the saliency-sampled, normalized index list `self.inds_normalised[strokes_counter]` (or uniform random if attention init is off). Then for each of the remaining `control_points_per_seg - 1` points it does `p1 = (p0[0] + radius*(rand-0.5), p0[1] + radius*(rand-0.5))` with **`radius = 0.05`**, chaining `p0 = p1`. Points are scaled by `canvas_width/height` (224) and wrapped in a `pydiffvg.Path(num_control_points, points, stroke_width=torch.tensor(width), is_closed=False)`.
- **`init_image()`** creates `num_paths` such `Path`s, each in a `pydiffvg.ShapeGroup(shape_ids=[i], fill_color=None, stroke_color=torch.tensor([0,0,0,1]))` — i.e. opaque black, no fill (open strokes). It sets `optimize_flag = [True]*len(shapes)` and renders once.
- **`render_warp()` / `get_image()`** call `pydiffvg.RenderFunction.apply` after `serialize_scene(canvas_width, canvas_height, shapes, shape_groups)`, with `num_samples_x = num_samples_y = 2`, `seed = 0`. The RGBA output is alpha-composited onto white: `img = opacity*rgb + 1*(1-opacity)`, then permuted HWC→NCHW to feed CLIP.
- **`parameters()`** sets `path.points.requires_grad = True` for every optimizable path and returns the list `points_vars` — these tensors *are* the optimization variables.
- **`save_svg()`** calls `pydiffvg.save_svg(...)`. The final artifacts are `best_iter.svg` and `final_svg.svg` in the output dir.

`PainterOptimizer` wraps a single `torch.optim.Adam(renderer.parameters(), lr=args.lr, betas=(0.9,0.9), eps=1e-6)` over the point tensors (color/width optimizers exist in the code but are inactive with defaults). Note the unusual **β₂ = 0.9** (not the usual 0.999) and the aggressive **lr = 1.0** — appropriate here because the parameters are pixel coordinates on a 224-px canvas, not neural weights.

### 3. How diffvg computes gradients w.r.t. control points

The scene function $f(x,y;\Theta)$ maps a location to a color given curve parameters $\Theta$ (control points, width). Standard rasterization performs an inside/outside test and alpha-composites overlapping primitives in a user order (Porter–Duff), which is **discontinuous** at curve boundaries — hence non-differentiable in $\Theta$. diffvg's key observation is that after **prefiltering** with an antialiasing kernel $k$,

$$ I(x,y) = \iint_A k(u,v)\,f(x-u, y-v;\Theta)\,du\,dv $$

the average color inside a pixel changes *continuously* as a curve moves, so $\partial I/\partial\Theta$ is well-defined. diffvg evaluates this integral (no closed form in general) by either **analytic prefiltering** or **Monte Carlo multisampling**; CLIPasso uses the multisampling path with `num_samples_x = num_samples_y = 2`. The rasterizer detects boundary/edge samples and produces unbiased gradients at those samples that attribute intensity changes to the underlying control-point positions — conceptually analogous to the edge-sampling / boundary-integral treatment of visibility discontinuities used in differentiable rendering. In code this is entirely encapsulated: `pydiffvg.RenderFunction` is a PyTorch `autograd.Function`, so `loss.backward()` transparently populates `path.points.grad`, and Adam updates the coordinates. (For open strokes, diffvg optimizes stroke width and color; for closed shapes it optimizes fill — CLIPasso uses open, fixed-width, fixed-black strokes.)

### 4. The loss (`models/loss.py`)

The top-level `Loss(nn.Module)` reads the config and instantiates sub-losses into a `loss_mapper` dict; with CLIPasso defaults the active loss is `CLIPConvLoss` (the `clip_conv_loss` family plus the `fc` cosine term). LPIPS and plain L2 are available (`--percep_loss lpips/l2`) but off by default. `forward` returns a dict of named per-layer losses; `painterly_rendering.py` sums them.

**`CLIPConvLoss`** loads `clip.load("RN101")`, freezes it (`.eval()`), and stores the OpenAI CLIP image normalization constants **mean `(0.48145466, 0.4578275, 0.40821073)`, std `(0.26862954, 0.26130258, 0.27577711)`**. It defines `augment_trans = Compose([RandomPerspective(fill=0, p=1.0, distortion_scale=0.5), RandomResizedCrop(224, scale=(0.7,0.7) [per aug_scale_min]/(0.8,0.8) in some variants, ratio=(1.0,1.0)), Normalize(...)])`. On each `forward(sketch, target, mode="train")` it builds a batch consisting of the un-augmented normalized pair plus **`num_aug_clip = 4`** augmented pairs, applying the *same* sampled affine transform to both sketch and target (`torch.cat([x,y])` then split) so the comparison is fair. Both batches go through `forward_inspection_clip_resnet`, which runs the CLIP RN101 stem then `layer1..layer4`, collecting the four intermediate feature maps and the final attention-pooled embedding.

- **Geometric term:** per-layer distance via `distance_metrics["L2"]` (squared/L2 differences of feature maps), each multiplied by the parsed weight vector `[0,0,1.0,1.0,0]` — so only the layer-3 and layer-4 outputs contribute, and are returned as `clip_conv_loss_layer2`/`clip_conv_loss_layer3` entries.
- **Semantic term:** `fc = (1 - cosine_similarity(sketch_embedding, target_embedding)).mean() * clip_fc_loss_weight` with `clip_fc_loss_weight = 0.1`. This is the `"fc"` entry that `painterly_rendering.py` separately tracks for `best_fc_loss`.

*(Note on sourcing: exact method bodies of `loss.py` could not be fetched verbatim through the tools; the hyperparameters `clip_model_name=RN101`, `clip_conv_loss_type=L2`, `clip_conv_layer_weights=0,0,1.0,1.0,0`, `clip_fc_loss_weight=0.1`, `num_aug_clip=4`, `aug_scale_min=0.7`, and `augemntations=affine` are confirmed verbatim from `config.py`; the CLIP normalization constants and the RandomPerspective/RandomResizedCrop augmentation scheme are confirmed by CLIPasso's text and the CLIPDraw++ appendix. The precise `RandomResizedCrop` scale tuple and per-layer reduction code should be verified against `models/loss.py` directly.)*

**Why augmentations prevent adversarial sketches:** optimizing directly against a frozen CLIP embedding invites degenerate, high-frequency "adversarial" stroke patterns that satisfy CLIP but look like noise to humans. Requiring the loss to be low across several random perspective/crop views (the CLIPDraw augmentation trick, adopted here) forces solutions that are robust to viewpoint — i.e. genuine, structurally coherent depictions.

**Architecture/layer ablations (supplementary):** The authors compared RN50, RN101, ViT-B/32, ViT-B/16. ViT layers 1–2 gave non-recognizable results; layers 3–5 combined geometry and semantics, trending more semantic with depth; the final FC layer alone (equivalent to using text) destroyed spatial coherence. ViT-B/16 strokes were "messier" with poorly-defined endpoints vs ViT-B/32. They favored CNNs, choosing **ResNet-101** as the best balance of geometric fidelity and semantics; within RN101 they select layers 3 and 4. They also report that replacing the CLIP geometric loss with **LPIPS** makes the result "too geometric" (close to simple contours), and **L2** is likewise inferior — CLIP activations capture a better geometry/semantics trade-off. A loss-weight ablation shows dropping $L_{semantic}$ preserves geometry but loses class semantics, and dropping $L_{geometric}$ loses instance-level recognizability.

### 5. Saliency-guided initialization

With `attention_init = 1` and `saliency_model = clip`, `saliency_clip_model = ViT-B/32`:
1. **Relevancy extraction (Chefer et al., 2021, no text).** `interpret()` runs `model.encode_image`, then over the 12 ViT attention blocks (`visual.transformer.resblocks`) reads each block's `attn_probs` (shape `[12 heads, 50, 50]` for 7×7=49 patches + CLS). Per block it clamps to ≥0 and **averages over heads**, then accumulates the relevancy matrix `R = R + attn @ R` (identity-initialized). It takes the CLS-to-patches row `cams_avg[:, 0, 1:]`, averages across the 12 layers, reshapes to 7×7, bicubically upsamples to 224×224, and min-max normalizes.
2. **XDoG intersection.** With `xdog_intersec = 1`, the relevancy map is multiplied by an **XDoG** (Winnemöller et al., 2012) edge map of the input, focusing initial strokes on salient *and* edge-rich regions.
3. **Softmax normalization** (temperature `softmax_temp = 0.3`) turns the enhanced map into a sampling distribution; the code squares the attention (`torch.pow(attn, 2)`) and multiplies by the object mask before deriving sampling indices.
4. **Sampling.** *n* first-control-points are drawn from this distribution (`inds` → `inds_normalised`), and the remaining three control points of each stroke are jittered within radius 0.05 (see §2).

**Why it matters:** the objective is highly non-convex in control-point space; random initialization frequently lands in poor local minima. Seeding strokes on salient/edge regions gives a strong prior, and running 3 seeds with lowest-eval-loss selection hedges against the residual variance.

### 6. Background handling (U²-Net)

For images with background, `--mask_object 1` runs `utils.get_mask_u2net` (U²-Net, Qin et al. 2020) to segment the salient object and mask out the background before both sketching and attention initialization (`mask_object_attention` is tied to the same flag by the runner). The weights `u2net.pth` are fetched from Google Drive on first run. **Limitation:** at high abstraction (very few strokes) the masking helps less — the method is designed for single salient objects, and complex scenes / multiple objects / heavy background are out of scope (this motivated the follow-up CLIPascene).

### 7. Evaluation

- **User study, N = 121** participants (60 evaluating CLIPasso's sketches, 38 and the rest split among competitors). Each saw sketches from one method and performed **category-level** recognition (pick the correct class among 4 confounders) and **instance-level** recognition. Category-level accuracy averaged **36%** at 4 strokes (rising to ~76% when uncertain responses are excluded), improving with more strokes; the qualitative "breaking point" where recognizability collapses is at **4 strokes**. A per-class supplementary breakdown found 'dog' and 'horse' below average, while 'cat' and 'giraffe' were well above (attributed to unambiguous features — triangular ears, long neck).
- **Automatic classifiers.** Sketches were run through a **ResNet34** sketch classifier and **CLIP ViT-B/32** zero-shot on 200 sketches from 10 categories. With the CLIP classifier, CLIPasso reached **78% (16 strokes)** and **91% (32 strokes)** top-1; the paper reports Top-1/Top-3 tables and shows CLIPasso outperforming photo-sketch baselines (Kampelmühler & Pinz; Li et al.) especially under the CLIP classifier, with instance-level numbers in the mid-90s% under some settings. (The competing Kampelmühler & Pinz method scores highest on the ResNet34 metric partly because it was trained on the same model/dataset the classifier uses.)

### 8. Colab notebook and Replicate

- **`CLIPasso.ipynb`** (Colab): installs `torch==1.7.1+cu101 torchvision==0.8.2+cu101`, `git+https://github.com/openai/CLIP.git`, builds **diffvg** from source (`git clone https://github.com/BachiLi/diffvg; git submodule update --init --recursive; python setup.py install`), and downloads U²-Net. It requires a GPU runtime and a runtime restart after install. It exposes `target_image`, `mask_object`, `fix_scale`, and `num_strokes` (suggested values 32/16/8/4), runs `run_object_sketching.py --num_sketches 3 -colab`, and lets you download `output_sketches/<name>/best_iter.svg`. Typical runtime is a few minutes per sketch on a Colab GPU (the paper's ~6 min/2000 iters is for a Tesla V100; slower hardware or CPU is much slower).
- **Replicate (`yael-vinker/clipasso`)** wraps the same code via `cog.yaml`, runs on an **Nvidia T4**, costs ≈**$0.33/run** (~3 runs/$1), with predictions "typically complete within 25 minutes." **As of this writing the model is flagged unrunnable on Replicate** because it was built with an unsupported Cog/Python version — a practical gotcha if you intended to call it as an API.

---

## Recommendations

**If your goal is to reproduce CLIPasso:**
1. **Use the authors' Docker image** (`docker pull yaelvinker/clipasso_docker`) or match the exact stack (**Python 3.7, PyTorch 1.7.1+cu101, torchvision 0.8.2+cu101**). diffvg does not compile cleanly on arbitrary environments; the dominant failure mode is a **CUDA-version mismatch** between the CUDA that built your PyTorch and your local toolkit (e.g. "detected CUDA 12.x mismatches the version used to compile PyTorch 11.x"), plus Windows-specific `ptxas fatal: Unresolved extern function` and `ninja` build errors on newer PyTorch. Build diffvg on Linux; verify `pydiffvg` imports before touching CLIPasso.
2. **Start from defaults** (16 strokes, 3 seeds, 2001 iters). Only deviate on `num_strokes` to change abstraction, and set `--mask_object 1` for images with background and `--fix_scale 1` for non-square inputs.
3. **Benchmark to change decisions:** if sketches look like noise/adversarial, confirm augmentations are active (`num_aug_clip=4`, `augment_both=1`). If class is right but shape is wrong, increase the geometric weight (lower `w_s` below 0.1 or raise layer-3/4 weights). If shape is right but the object is unrecognizable, raise `clip_fc_loss_weight`. If results are unstable, increase `num_sketches` beyond 3.
4. **Get the exact `loss.py`** from the raw source or a faithful mirror (ximinng/PyTorch-SVGRender, swiftsketch/SwiftSketch) to confirm the precise augmentation scale tuple and per-layer reduction before publishing reproductions.

**If your goal is to build on CLIPasso for image→sketch research:**
- Treat CLIPasso as the canonical "optimize diffvg Béziers against a frozen CLIP perceptual loss" recipe and swap components: replace the guidance model (diffusion/SDS instead of CLIP), the primitive (splatting/other curves for speed — diffvg's open-curve backward is a known bottleneck, motivating Bézier Splatting), or the initialization (segmentation-driven point distribution, K-means clustering as in ControlSketch/SwiftSketch).
- **Research map of follow-ups by the same group / line:** **CLIPascene** (CVPR 2023) extends to full scenes with type/level-of-abstraction control; **SketchAgent** (2024) does language-driven sequential sketching; **VectorFusion** and **DiffSketcher** replace CLIP guidance with diffusion/SDS for text-to-vector; **CLIPDraw / CLIPDraw++** are the text-to-drawing predecessors that supplied the augmentation trick; **SwiftSketch/ControlSketch** add feed-forward speed and smarter initialization. For evaluation, the **SEVA** benchmark uses CLIPasso as the machine-sketch generator against ~90k human sketches at matched 4/8/16/32 abstraction levels.

---

## Caveats

- **Sourcing on `models/loss.py`:** I could not retrieve the file's raw text through the available tools. All numeric hyperparameters cited (RN101, L2 conv loss, layer weights `0,0,1.0,1.0,0`, `clip_fc_loss_weight=0.1`, `num_aug_clip=4`, `aug_scale_min=0.7`, augmentations `affine`) are **verbatim-confirmed from `config.py`**; the CLIP normalization constants and the RandomPerspective(distortion_scale=0.5)+RandomResizedCrop scheme are confirmed by the paper text and the CLIPDraw++ appendix, but the exact `RandomResizedCrop` scale tuple and the precise per-layer L2 reduction code should be verified directly against the source file.
- **`painter_params.py` details** (the Chefer `interpret()` relevancy loop, radius-0.05 jitter, single-Adam optimizer with `betas=(0.9,0.9), eps=1e-6`) are quoted from the SwiftSketch/ControlSketch fork, which reuses CLIPasso's code near-verbatim; the original CLIPasso file uses the same structure but line numbers differ, and the fork adds diffusion-attention and K-means options not present in the 2022 CLIPasso release.
- **Timing figures** (~2000 iterations, ~6 minutes, Tesla V100) are from the paper; the repo's early-stopping means actual iteration counts vary per image, and the default `num_iter` is 2001 (runner) vs 500 (config).
- **Numbers vs. hedges:** the "36% at 4 strokes / 76% excluding uncertain" and "78%/91% CLIP classifier at 16/32 strokes" figures come from the paper and its scribd/researchgate reproductions; the user-study split (60 evaluated CLIPasso out of 121) is confirmed, but exact per-abstraction category/instance percentages beyond these are reported only in supplementary figures.
- **Replicate is currently non-functional** (unsupported Cog/Python), so treat it as documentation of intended inputs/outputs (default 16 strokes, mask/fix-scale toggles, SVG output) rather than a live API.
- **Scope:** CLIPasso is explicitly single-salient-object; scenes, multiple objects, and heavy backgrounds are out of scope and were the motivation for CLIPascene. The number of strokes must be chosen a priori (the authors note making it a learned parameter as future work).