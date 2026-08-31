# CLIPasso, runnable

A fork of [CLIPasso: Semantically-Aware Object Sketching](https://clipasso.github.io/clipasso/)
that installs on a current machine, runs on a current GPU, and draws about three
times faster than the original at the same settings.

The original code was published in 2022 against Python 3.7, PyTorch 1.7.1 and
CUDA 10.1. On a recent machine it does not install, and on a GPU newer than
Ampere it cannot run at all. This fork changes what is needed to fix that, plus
four optimisations that leave the output unchanged.

Nothing here changes the method. The sketches are the same sketches.

For the paper, the project page, the Colab demo and the interactive notebooks, go
to the [original repository](https://github.com/yael-vinker/CLIPasso).


## What is different

| | upstream | this fork |
|---|---|---|
| Python | 3.7 | 3.8 or newer |
| PyTorch | 1.7.1 / CUDA 10.1 | current, built for your GPU |
| GPU support | up to Ampere | whatever card you have, including Blackwell |
| diffvg | does not build against CUDA 12 | built by `setup.sh` |
| Time for 3 sketches at 16 strokes | 6 min 19 s | 2 min 0 s |

Timings are for the default settings on one image, measured on an RTX PRO 6000
(Blackwell). See [Speed](#speed) for the breakdown and the caveats.


## Install

Linux, an NVIDIA GPU with a working driver, and two things diffvg needs in order to
compile:

- **Python development headers** (`Python.h`). `sudo apt install python3-dev` on
  Debian or Ubuntu, `sudo dnf install python3-devel` on Fedora or RHEL.
- **A CUDA toolkit** providing `nvcc`. PyTorch bundles a CUDA *runtime*, not a
  compiler, so the driver alone is not enough. `sudo apt install nvidia-cuda-toolkit`,
  or from https://developer.nvidia.com/cuda-downloads.

`setup.sh` checks for both before it downloads anything and tells you what is
missing, so a bad environment costs you a second rather than ten minutes.

    git clone https://github.com/daviidmparedes/CLIPasso-warp
    cd CLIPasso-warp
    ./setup.sh

It creates a virtualenv, installs PyTorch and the dependencies, then clones and
compiles diffvg against the compute capability of the GPU in your machine. It
finishes by rendering a test image and checking the gradients are finite, so if it
prints "Setup complete" the install works.

On a shared machine where you have no root:

    CUDA_HOME=/path/to/cuda ALLOW_LOCAL_PYTHON_HEADERS=1 ./setup.sh

`CUDA_HOME` points at an existing toolkit; `ALLOW_LOCAL_PYTHON_HEADERS=1` downloads
and unpacks the Python headers into `third_party/` without installing anything
system-wide. Both were used to test this script on a cluster node with neither
installed.

To pin a particular PyTorch CUDA build, set `CUDA_TAG` (default `cu128`):

    CUDA_TAG=cu121 ./setup.sh


## Use

    source .venv/bin/activate
    python run_sketch.py --target target_images/camel.png

The result is written to `output_sketches/camel/`, with the chosen sketch at
`best_sketch.svg`.

Useful options:

    --num-strokes N     how many strokes (default 16). Fewer is more abstract.
    --num-sketches N    how many sketches to draw (default 3); the best is kept
    --num-iter N        optimisation steps per sketch (default 2001)
    --mask-object       remove the background first; use when the subject is not isolated
    --fix-scale         use when the image is not square
    --output-dir DIR    write somewhere other than output_sketches/
    --save-interval N   also save an SVG every N steps, to watch the sketch form

The first run downloads the U2Net saliency model (about 170 MB) once.

Drawing several sketches and keeping the best is how CLIPasso is meant to be
used: the result depends on where the strokes are initialised, and some
initialisations fail. `--num-sketches 3` is the default for that reason, and
because of how this fork works it costs much less than three times one sketch.

`painterly_rendering.py` still works exactly as upstream if you want the
single-sketch entry point.


## Speed

Measured at 16 strokes over five images and three sketches each, as seconds per
sketch:

| change | speedup | cumulative | s/sketch |
|---|---|---|---|
| upstream | - | 1.00x | 126.3 |
| release the autograd graph each iteration | 1.48x | 1.48x | 85.3 |
| draw all sketches in one process | 1.67x | 2.47x | 51.1 |
| freeze the CLIP encoder, skip the unused loss | 1.28x | 3.16x | 40.0 |

Every one of these is a change to how the work is scheduled, not to what is
computed:

- **Releasing the autograd graph.** The training loop kept the previous
  iteration's graph alive across the next forward pass. diffvg allocates outside
  PyTorch's caching allocator, so with its graph pinned the next backward falls
  back to raw `cudaMalloc`/`cudaFree`, and `cudaFree` synchronises the device.
  diffvg's backward costs 26.4 ms with the graph retained and 7.7 ms without.

- **Drawing all sketches in one process.** Upstream runs one process per sketch,
  each reloading CLIP and the saliency model and then handing the GPU one small
  sketch at a time. Because the sketches have disjoint parameters, stepping them
  together under one optimiser is exactly equivalent to running them separately,
  and it gives the GPU enough work to be worth launching.

- **Freezing the CLIP encoder.** The encoder is frozen guidance; only the stroke
  control points are optimised. Its parameters still had `requires_grad=True`, so
  autograd allocated and freed gradient buffers for 119.7 M parameters every
  iteration. The same commit stops building a second CLIP model that the default
  configuration never calls.

Quality was checked rather than assumed, on the repository's own evaluation loss
and on two measures that do not use the model being optimised against: zero-shot
classification and sketch-to-photo retrieval with a different CLIP backbone. All
differences are smaller than the difference between two runs of the *unmodified*
code with the same seed, which is substantial: CLIPasso is not deterministic,
because diffvg's backward accumulates gradients with `atomicAdd`, and over 2001
steps that grows to about 57 px of mean control-point drift on a 224 px canvas.

Two caveats on the numbers. They were taken on a GPU shared with another job, so
the ratios are paired and trustworthy but the absolute times are upper bounds.
And they are one card and one image set; your mileage will differ.

The measurement harness, the full log and several things that did not work are on
the `experiments` branch.


## Credit

All of the method and nearly all of the code is the work of the original authors.
If you use this, cite their paper:

    @article{vinker2022clipasso,
        title={CLIPasso: Semantically-Aware Object Sketching},
        author={Vinker, Yael and Pajouheshgar, Ehsan and Bo, Jessica Y and
                Bachmann, Roman Christian and Bermano, Amit Haim and
                Cohen-Or, Daniel and Zamir, Amir and Shamir, Ariel},
        journal={ACM Transactions on Graphics (TOG)},
        volume={41},
        number={4},
        pages={1--11},
        year={2022},
        publisher={ACM New York, NY, USA}
    }

Licensed as upstream; see LICENSE.
