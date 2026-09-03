# TransDG: OfficeHome Reproduction Code

Thank you for taking the time to look at our code. This package reproduces the
OfficeHome leave-one-domain-out experiments described in the paper and its
appendix. The implementation follows the complete two-stage pipeline, from
CLIP feature adaptation and prototype construction to graph propagation and
target-domain evaluation.

We recommend starting with the smoke test below. It takes only a few seconds
and does not require the OfficeHome dataset or a CLIP checkpoint.

## Quick start

Install the dependencies:

```text
python -m pip install -r requirements.txt
```

Then run the core tests:

```text
python smoke_test.py
```

A successful run ends with:

```text
SMOKE_OK: 8 core tests passed
```

These tests cover the prototype counts, the losses corresponding to Eqs. (13)
and (20), the graph parameter count, and the bidirectional instance bridge at
every propagation layer.

## Environment

The experiments reported in the paper used Python 3.9.13, PyTorch 2.4.0,
torchvision 0.19.0, and CUDA 12.1 on one NVIDIA GeForce RTX 4090. The remaining
package versions are listed in `requirements.txt`. The smoke test also passes
with PyTorch 2.5.1, torchvision 0.20.1, and NumPy 2.2.6.

## Data and model checkpoint

Full training needs a local OfficeHome dataset and a local CLIP ViT-B/32
checkpoint. They are not included because their combined size is larger than
the 50 MB supplementary-material limit.

The simplest layout is:

```text
TransDG/
  ViT-B-32.pt
  data/OfficeHome/
    Art/<class_name>/<image>
    Clipart/<class_name>/<image>
    Product/<class_name>/<image>
    RealWorld/<class_name>/<image>
```

The checkpoint used in our tests has the following SHA-256 value:

```text
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

OfficeHome contains 65 classes across four domains. Please keep the original
class-directory names. The loader searches each class directory recursively
for JPG, JPEG, and PNG files.

Before starting a long run, the split can be checked without loading CLIP:

```text
python train.py --dry-run --lodo Art --seeds 108
```

## Running the experiments

To run one held-out domain with one seed:

```text
python train.py --lodo Art --seeds 108 \
  --dataset-root ./data/OfficeHome --model-path ./ViT-B-32.pt
```

To reproduce all four target domains and the three reported seeds:

```text
python train.py --lodo all --seeds 108 113 115 \
  --dataset-root ./data/OfficeHome --model-path ./ViT-B-32.pt
```

Candidate-conditioned visual enhancement is evaluated in class chunks. The
default `--visual-class-chunk-size 1` has the lowest memory cost. A larger value
can improve throughput when more GPU memory is available without changing the
underlying computation.

## What the pipeline does

For each seed and held-out target domain, the code:

1. splits every source class-domain group into 80% training and 20% validation;
2. trains the adaptive prompts and visual enhancement module for 15 epochs,
   while keeping both CLIP encoders frozen;
3. applies the complete reliability-weighted matched-prototype consistency in
   Eq. (13) with bounded-memory mini-batch recomputation;
4. selects the Stage-I checkpoint using source validation only;
5. constructs and freezes the source-training prototype bank;
6. produces 3 real, 4 extrapolated, and 21 interpolated anchors per class in
   the three-source-domain setting;
7. trains the shared graph module for 10 epochs and again selects the checkpoint
   using source validation only; and
8. evaluates once on the held-out target domain.

The target images and labels do not participate in training, prototype
construction, hyperparameter selection, or checkpoint selection.

Within each local class graph, all real anchors and the three most relevant
anchors of each pseudo-domain type are retained. At every propagation layer,
the visual instance receives textual messages and the textual instance receives
visual messages. The two bridge states are updated synchronously; prototype
nodes remain fixed throughout propagation.

The implementation checks the appendix parameter counts at runtime:

- Stage-I trainable modules: 12,378,705 parameters
- shared graph module: 1,050,670 parameters

## Outputs

Results are saved under `outputs/officehome/seed_<seed>/<target>/`. Each run
contains the selected checkpoints, training histories, and a summary. After all
runs finish, `all_results.json` records target-wise results, sample standard
deviations, and the overall domain average.

## File guide

- `train.py` contains the data protocol, two training stages,
  prototype expansion, graph propagation, evaluation, and aggregation.
- `transclip.py` contains the frozen CLIP encoders, adaptive prompts,
  and visual feature enhancement.
- `clip/` contains the local CLIP implementation used to expose intermediate
  ViT states. It loads local checkpoints only.
- `tests/test_core.py` contains the formula- and graph-level tests.
- `NOTICE.md` records the notice for the included CLIP-derived
  code.

## License

The original TransDG code in this package is released under the MIT License in
`LICENSE`. The CLIP-derived files under `clip/` retain their original license,
which is reproduced in `NOTICE.md`.
