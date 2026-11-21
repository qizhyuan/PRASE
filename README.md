# PRASE

PRASE aligns the entities of two knowledge graphs without labeled pairs, by combining probabilistic reasoning with semantic embeddings ([Qi et al., IJCAI 2021](https://www.ijcai.org/proceedings/2021/278)).

## Supported Methods

| Method | What It Does | Python | Run File `[method]` |
| --- | --- | --- | --- |
| PRASE | Runs PARIS, trains an embedding model on the PARIS matches, and runs PARIS again with the model's proposed pairs and similarity | `methods.prase("bootea")` | `name = "prase"`, `model = "bootea"` |
| PARIS | Matches entities, relations, and attributes by probabilistic reasoning over the graphs' facts and equal literals ([Suchanek et al., VLDB 2011](https://doi.org/10.14778/2078331.2078332)) | `methods.paris()` | `name = "paris"` |
| Embedding only | Trains an embedding model on labeled seed pairs | `methods.embedding_only("bootea")` | `name = "embedding-only"`, `model = "bootea"` |

PRASE and embedding-only training take any of six embedding models in PyTorch, or [your own](#embedding-models):

| Model | Paper | `model` |
| --- | --- | --- |
| BootEA | Sun et al., IJCAI 2018 | `"bootea"` |
| GCNAlign | Wang et al., EMNLP 2018 | `"gcnalign"` |
| IMUSE | He et al., DASFAA 2019 | `"imuse"` |
| IPTransE | Zhu et al., IJCAI 2017 | `"iptranse"` |
| MTransE | Chen et al., IJCAI 2017 | `"mtranse"` |
| MultiKE | Zhang et al., IJCAI 2019 | `"multike"` |

PARIS runs on a C++ engine or a NumPy engine, which give the same alignments up to floating-point rounding. Every run records its intermediate alignments and scores, and a [manual review](#manual-review) keeps approved pairs and never matches rejected ones. The PARIS authors publish their Java implementation at [dig-team/PARIS](https://github.com/dig-team/PARIS). The implementation released with the IJCAI 2021 paper is on the [`legacy`](https://github.com/qizhyuan/PRASE/tree/legacy) branch.

## PRASEMap Web Application

[PRASEMap](https://prasemap.qizhy.com/) is a web application built on PRASE, first presented as a demo at CIKM 2021 ([Qi et al., CIKM 2021](https://doi.org/10.1145/3459637.3481972)). In the browser you can upload two graphs or choose a built-in pair, run PARIS or PRASE, and review the matches. Approved and rejected pairs carry over to the next run.

## Installation

PRASE requires Python 3.11 or newer. PyPI and the repository install the same `prase` package, so an installation from one replaces the other in the same environment.

### From PyPI

```console
# PARIS, the pipelines, and the benchmark downloader.
python -m pip install prase
# Or with PyTorch and RapidFuzz for the embedding models.
python -m pip install "prase[train]"
# Or with RDFLib for RDF input.
python -m pip install "prase[rdf]"
# Or with both.
python -m pip install "prase[train,rdf]"
```

The wheels for Linux (x86_64 and aarch64), Windows (x64), and macOS (Apple silicon) on Python 3.11 to 3.15 include the compiled PARIS engine. On other platforms pip builds the engine from source, which requires CMake 3.18+ and a C++17 compiler.

### From the Repository

Installing from the repository builds the C++ engine, which requires CMake 3.18+ and a C++17 compiler:

```console
# The latest code on main, straight from GitHub.
python -m pip install "prase[train] @ git+https://github.com/qizhyuan/PRASE.git"
# Or a release, by its tag.
python -m pip install "prase[train] @ git+https://github.com/qizhyuan/PRASE.git@v1.0.1"
```

A clone also holds the [benchmark recipes](#benchmarks):

```console
git clone https://github.com/qizhyuan/PRASE.git
cd PRASE
# Optional: a release instead of main.
git switch --detach v1.0.1
# Build PRASE with the embedding models and RDF input.
python -m pip install ".[train,rdf]"
# Or, on a release, install the prebuilt package of the same version.
python -m pip install "prase[train]==1.0.1"
# Or PARIS with the NumPy engine alone, without building the C++ engine.
python -m pip install . --config-settings wheel.cmake=false
```

### GPU Training

Embedding models train on the CPU by default. For GPU training, install the CUDA build of PyTorch that the [PyTorch installer](https://pytorch.org/get-started/locally/) selects for your system first, then PRASE with the `train` extra. pip keeps the CUDA build because it already meets PRASE's PyTorch requirement.

```console
# 1. The CUDA build of PyTorch.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
# 2. PRASE with the embedding models.
python -m pip install "prase[train]"
```

Python selects the GPU with `options={"device": "cuda"}`, a run file with `options = { device = "cuda" }`, and the benchmark runner with `--device cuda`.

## Quick Start

### Align Two Graphs

```python
from prase import KG, methods

# Relation triples (head, predicate, tail) and attribute triples (entity, predicate, value).
left = KG.from_triples(
    [("l:paris", "capital_of", "l:france")],
    [("l:paris", "name", "Paris"), ("l:france", "name", "France")],
)
right = KG.from_triples(
    [("r:paris", "capitalOf", "r:france")],
    [("r:paris", "label", "Paris"), ("r:france", "label", "France")],
)

result = methods.paris().run(left, right)
print(result.alignment.to_ids(left, right))  # [(left ID, right ID, score), ...]
```

### Run a Benchmark

After [signing in to Hugging Face](#datasets):

```python
from prase import load_benchmark, methods

# Download and verify D-W-15K-V2 on first use.
data = load_benchmark("D-W-15K-V2")

# PRASE: PARIS, BootEA, and PARIS again with BootEA's proposals and embeddings.
pipeline = methods.prase("bootea", options={"device": "cuda"})
result = pipeline.run(data.left, data.right, reference=data.reference())
print(result.scores)  # precision, recall, and F1 of the final alignment
```

The same experiment as a recipe of the paper, from a [clone of the repository](#from-the-repository):

```console
# Write the run to runs/reproduction/prase-bootea/D-W-15K-V2/ and tabulate it.
python benchmarks/reproduce.py run --method prase --model bootea --dataset D-W-15K-V2 --device cuda --output runs/reproduction
python benchmarks/reproduce.py report runs/reproduction
```

## Align Your Own Graphs

A run file names the graphs, the method, and the output directory. Paths are relative to the run file:

```toml
# run.toml
[input]
seeds = "seeds.tsv"          # Optional known pairs, one left<TAB>right ID pair per line.
reference = "reference.tsv"  # Optional pairs that the result is scored against, in the same format.
# dataset = "path/to/openea-folder"  # Or both graphs from an OpenEA folder, instead of [input.left] and [input.right].
# benchmark = "D-W-15K-V2"           # Or both graphs from a benchmark dataset.

[input.left]
relations = "left-relations.tsv"    # One head<TAB>predicate<TAB>tail triple per line.
attributes = "left-attributes.tsv"  # Optional: one entity<TAB>predicate<TAB>value triple per line.
# rdf = "left.ttl"                  # Or one RDF file, with the rdf extra.

[input.right]
relations = "right-relations.tsv"
attributes = "right-attributes.tsv"

[method]
name = "prase"                # "paris", "prase", or "embedding-only"
model = "gcnalign"            # The embedding model of "prase" and "embedding-only".
options = { device = "cpu" }  # The model's options.

[output]
directory = "runs/my-alignment"
```

```console
# Align the graphs and write the results to runs/my-alignment.
prase run run.toml
# Score an alignment against the ent_links file of an OpenEA folder.
prase evaluate runs/my-alignment/alignment.tsv --dataset path/to/openea-folder
# List the embedding models.
prase models
```

The same run in Python:

```python
from prase import Alignment, Dataset, KG, methods, read_links

# KG.from_rdf("left.ttl") reads RDF, and load_dataset("path/to/openea-folder") reads an OpenEA folder.
left = KG.from_tsv("left-relations.tsv", "left-attributes.tsv")
right = KG.from_tsv("right-relations.tsv", "right-attributes.tsv")
seeds = Alignment.read("seeds.tsv", left, right)
reference = Dataset("mine", left, right, read_links("reference.tsv")).reference()

pipeline = methods.prase("gcnalign", options={"device": "cpu"})
result = pipeline.run(left, right, seeds=seeds, reference=reference, output="runs/my-alignment")
print(result.scores)
```

The output directory holds the final alignment, the run settings, and one folder per step:

```text
runs/my-alignment/
├── alignment.tsv        left ID, right ID, score
├── run.json             settings and per-step scores
├── 01-paris/            alignment.tsv, predicates.tsv, step.json
├── 02-embed-gcnalign/   proposals.tsv, step.json with the training summary
├── 03-paris/            alignment.tsv, predicates.tsv, step.json
└── checkpoint/          finished steps and training state, removed when the run finishes
```

### Progress and Checkpoints

`prase run` logs each step, each PARIS iteration, and each embedding model's training at most every 30 seconds, with the time elapsed and the time left. Training with early stopping states the epoch cap as an upper bound and the epoch at which it stops if validation stops improving:

```text
Step 2 of 3 (embed-gcnalign) started
GCN-Align training: epoch 340 of at most 2000, 2 min 10 s elapsed, at most 10 min 35 s left; validation MRR 0.8123, best 0.8150 at epoch 330, stops at epoch 380 without improvement
GCN-Align training: 380 epochs in 2 min 25 s (patience exhausted); selected epoch 330 at validation MRR 0.8150 on 512 seed pairs
Step 2 of 3 (embed-gcnalign) took 146.0 s
```

In Python, the same lines go to the `prase` logger:

```python
import logging

import prase.progress

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
prase.progress.INTERVAL_SECONDS = 10  # log every 10 seconds instead of every 30
```

A run with an output directory saves checkpoints in its `checkpoint` folder: the result of every finished step, and the state of an embedding model every 10 minutes of training. After an interruption, `--resume` continues the run in the same directory. It reuses every finished step whose settings, files, code, and input are unchanged, and continues training from the last saved epoch, with the same result as an uninterrupted run. A finished model fit stays saved until its step's result is, so a failure later in the step does not repeat the training. Every checkpoint carries a SHA-256 digest of its content, and one that is damaged, incomplete, or altered is ignored: its step runs again, or its training starts from the first epoch. A finished run removes its checkpoints, so resuming it is an error unless the checkpoints were kept:

```console
# Continue an interrupted run from its checkpoints.
prase run run.toml --resume
# Save training state every 2 minutes and keep the checkpoint folder after the run finishes.
prase run run.toml --checkpoint-minutes 2 --keep-checkpoints
# Save no checkpoints.
prase run run.toml --no-checkpoints
```

The run file sets the same policy:

```toml
[output]
directory = "runs/my-alignment"
# checkpoints = false        # Save no checkpoints.

[output.checkpoints]
steps = true                 # Save the result of every finished step.
training = true              # Save the training state of embedding models.
training_minutes = 2         # Minutes between training checkpoints; 0 saves after every epoch.
keep = true                  # Keep the folder after the run finishes, so that --resume reuses every step.
# directory = "checkpoints"  # Another folder: new, empty, or the checkpoint folder of an earlier run.
```

A checkpoint folder outside the output directory must not contain it. PRASE refuses a folder that holds files it did not write, and the cleanup after a run removes only the checkpoints PRASE wrote.

In Python, `CheckpointPolicy` chooses the checkpoints, and `training_checkpoints` protects a model fitted outside a pipeline. It yields the files it saves, which stay until you remove them:

```python
import numpy as np

from prase import CheckpointPolicy, TrainingData, create_model, methods, training_checkpoints

# left, right, and seeds come from the example above.
pipeline = methods.prase("bootea", options={"device": "cuda"})
policy = CheckpointPolicy(training_minutes=2, keep=True)
result = pipeline.run(left, right, output="runs/my-alignment", checkpoints=policy)
# After an interruption: reuse the finished steps and continue training.
result = pipeline.run(left, right, output="runs/my-alignment", checkpoints=policy, resume=True)

# Train on the seed pairs; the other entities are the candidates.
pairs = np.stack((seeds.left, seeds.right), axis=1)
candidates = (np.setdiff1d(np.arange(left.num_entities), seeds.left),
              np.setdiff1d(np.arange(right.num_entities), seeds.right))
data = TrainingData(left, right, pairs, candidates)
with training_checkpoints("checkpoints", minutes=5) as files:
    embeddings = create_model("bootea", device="cuda").fit(data, seed=0)
embeddings.save("bootea.npz")
for path in files:  # the result is saved, so the finished fit can go
    path.unlink()
```

### Manual Review

To correct an alignment, list approved and rejected pairs in a TSV file and rerun with it into a new output directory. Approved pairs stay fixed, and rejected pairs are never matched:

```text
left	right	decision
http://dbpedia.org/resource/Illinois	http://www.wikidata.org/entity/Q1204	approve
http://dbpedia.org/resource/Illinois	http://www.wikidata.org/entity/Q1297	reject
```

```console
# Or set decisions = "review.tsv" in the run file's [input].
prase run run.toml --decisions review.tsv --output runs/my-alignment-reviewed
```

`Session` keeps the graphs indexed between PARIS runs for interactive review:

```python
from prase import Session, load_benchmark

data = load_benchmark("D-W-15K-V2")
session = Session(data.left, data.right)
session.run()

# PARIS matches DBpedia's Illinois to Chicago (Q1297). Illinois (Q1204) is the runner-up.
illinois = "http://dbpedia.org/resource/Illinois"
print(session.match(illinois))  # the accepted counterpart and the probability the run recorded
# Candidate scores repeat the evidence computation on the run's final state.
for right_id, candidate in session.candidates(illinois, limit=3):
    print(right_id, round(candidate.score, 3))

session.approve(illinois, "http://www.wikidata.org/entity/Q1204")
session.fix_predicate("http://dbpedia.org/ontology/birthPlace", "http://www.wikidata.org/entity/P19")
session.run()
print(session.matches()[:5])
```

## Benchmarks

### Datasets

The benchmark collection [qizhy777/prase-benchmarks](https://huggingface.co/datasets/qizhy777/prase-benchmarks) contains the six datasets of the paper: EN-FR-100K-V2, EN-DE-100K-V2, D-W-100K-V2, D-Y-100K-V2, and D-W-15K-V2 from [OpenEA](https://github.com/nju-websoft/OpenEA), and MED-BBK-9K from [industry-eval-EA](https://github.com/ZihengZZH/industry-eval-EA). Downloads require a Hugging Face account with read access to the collection. PRASE pins a revision of the collection and checks the SHA-256 of every file it downloads. The `hf` command comes with PRASE:

```console
# Sign in with an access token from https://huggingface.co/settings/tokens.
hf auth login
# Download and verify all six datasets into the Hugging Face cache.
prase download
# Or download some of them.
prase download D-W-15K-V2 MED-BBK-9K
```

MultiKE and the EMB-Match baseline read the English FastText vectors [wiki-news-300d-1M.vec](https://fasttext.cc/docs/en/english-vectors.html):

```console
# Download the 682 MB archive, extract the vectors into data/, and check their SHA-256.
prase download --word-vectors
# Or extract them into another folder.
prase download --word-vectors --vectors-dir path/to/vectors
```

### Recipes

The recipes are in the repository, not in the PyPI package. Each method has one stable config, and the paper's suite builds its recipes from them:

```text
benchmarks/
├── reproduce.py      runs, lists, and reports the recipes
├── ijcai2021.toml    the recipes of the IJCAI 2021 paper
├── prase/            PARIS, and PRASE with each embedding model
└── supervised/       each embedding model trained on labeled pairs
```

A stable config holds a run file's `[method]` table and the benchmark's per-dataset settings. The `[method]` table works in your own run file, where MultiKE's `"@word_vectors"` becomes the path of the vector file:

```toml
# benchmarks/prase/bootea.toml
[method]
name = "prase"
model = "bootea"

# BootEA options for the 100K datasets, merged over [method].options.
[dataset_options]
EN-FR-100K-V2 = { batch_size = 20000, epsilon = 0.98 }
```

A recipe names a config and the settings it changes. The paper's name-matching baselines, STR-Match and EMB-Match, define their method in place. The comments at the top of [`ijcai2021.toml`](https://github.com/qizhyuan/PRASE/blob/main/benchmarks/ijcai2021.toml) document every recipe setting.

```toml
# benchmarks/ijcai2021.toml
[experiments.prase-bootea-m]
description = "PRASE with BootEA, proposed pairs only"
config = "prase/bootea.toml"
method = { share_embeddings = false }
```

`benchmarks/reproduce.py` runs each recipe on its datasets and writes the runs to `<output>/<recipe>/<dataset>/`, logging each run's number and the time the suite has left. Pass the same `--output` directory to `list` and `report`. A folder that already holds a run stops the suite unless `--resume` is given. With `--resume`, the suite:

- skips current runs of the requested `--seed` and `--device`, which finished with the recipe's present settings and evaluation, the installed PRASE build (its version and the digests of its sources and PARIS extension), and the installed NumPy and PyTorch versions;
- continues interrupted runs from their checkpoints, and keeps the supervised folds that finished with the same identity;
- moves every other finished run in the way, such as a run of an older recipe, build, seed, or device, to `<output>/history/<time>/` and runs it again.

`--rounds` with the recipe's own round count names the recipe's base run. `list` and `report` mark runs that are not current as historical:

```console
# PARIS on every dataset.
python benchmarks/reproduce.py run --method paris --all-datasets --output runs/reproduction
# PRASE with BootEA and with MultiKE on every dataset.
python benchmarks/reproduce.py run --method prase --model bootea --all-datasets --device cuda --output runs/reproduction
python benchmarks/reproduce.py run --method prase --model multike --all-datasets --device cuda --word-vectors data/wiki-news-300d-1M.vec --output runs/reproduction
# A supervised embedding model: train on each benchmark fold and rank its test links.
python benchmarks/reproduce.py run --method se --model gcnalign --dataset D-W-15K-V2 --device cuda --output runs/reproduction
# One recipe by name, such as PRASE-BootEA with proposed pairs only.
python benchmarks/reproduce.py run --experiment prase-bootea-m --all-datasets --device cuda --output runs/reproduction
# Every recipe and its status.
python benchmarks/reproduce.py list --output runs/reproduction
# Tabulate the completed runs.
python benchmarks/reproduce.py report runs/reproduction
# The entire suite in its own directory.
python benchmarks/reproduce.py run --all --device cuda --word-vectors data/wiki-news-300d-1M.vec --output runs/full-suite
# Continue the suite after an interruption, or after a recipe changed.
python benchmarks/reproduce.py run --all --device cuda --word-vectors data/wiki-news-300d-1M.vec --output runs/full-suite --resume
```

### Results

All runs use seed 0 and train the embedding models on an NVIDIA RTX 5090. Embedding training is not bit-for-bit reproducible, so scores vary slightly from run to run.

F1 of the final alignment of PRASE with one round, and of PARIS alone. The best run on each dataset is in bold. The values in parentheses are reported in the [PRASE paper](https://www.ijcai.org/proceedings/2021/278) (Tables 2 and 4).

| Dataset | PRASE-MTransE | PRASE-IPTransE | PRASE-GCNAlign | PRASE-IMUSE | PRASE-BootEA | PRASE-MultiKE | PARIS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EN-FR-100K-V2 | 0.946 (0.938) | TBD (0.947) | 0.943 (0.939) | TBD (0.941) | 0.957 (0.954) | **0.964** (0.954) | 0.935 (0.926) |
| EN-DE-100K-V2 | 0.960 (0.956) | TBD (0.961) | 0.956 (0.954) | TBD (0.959) | 0.966 (0.965) | **0.977** (0.972) | 0.953 (0.948) |
| D-W-100K-V2 | TBD (0.865) | TBD (0.873) | 0.873 (0.861) | TBD (0.873) | **0.890** (0.890) | 0.869 (0.859) | 0.868 (0.854) |
| D-Y-100K-V2 | TBD (0.989) | TBD (0.992) | 0.989 (0.986) | TBD (0.992) | 0.995 (0.993) | **0.998** (0.996) | 0.987 (0.983) |
| D-W-15K-V2 | TBD (0.914) | TBD (0.910) | 0.908 (0.904) | TBD (0.913) | **0.917** (0.923) | 0.908 (0.907) | 0.904 (0.897) |
| MED-BBK-9K | TBD (0.485) | TBD (0.517) | 0.528 (0.517) | TBD (0.501) | 0.561 (0.557) | **0.721** (0.711) | 0.481 (0.499) |

Hits@1 of each embedding model trained on labeled pairs alone, without PARIS: the mean over the five folds of each OpenEA dataset, where each fold trains on its training links and ranks its test links, and the single split of MED-BBK-9K. The best run on each dataset is in bold. The values in parentheses are reported by [OpenEA](https://www.vldb.org/pvldb/vol13/p2326-sun.pdf) (Sun et al., VLDB 2020, Table 5) and, for MED-BBK-9K, by the PRASE paper (Table 2).

| Dataset | MTransE | IPTransE | GCNAlign | IMUSE | BootEA | MultiKE |
| --- | --- | --- | --- | --- | --- | --- |
| EN-FR-100K-V2 | 0.080 (0.090) | TBD (0.234) | 0.250 (0.257) | TBD (0.461) | **0.634** (0.640) | TBD (0.642) |
| EN-DE-100K-V2 | 0.095 (0.115) | TBD (0.346) | 0.372 (0.375) | TBD (0.457) | **0.739** (0.739) | TBD (0.661) |
| D-W-100K-V2 | 0.129 (0.148) | TBD (0.319) | **0.344** (0.353) | TBD (0.431) | TBD (0.766) | TBD (0.319) |
| D-Y-100K-V2 | 0.097 (0.100) | TBD (0.456) | **0.575** (0.620) | TBD (0.629) | TBD (0.886) | TBD (0.853) |
| D-W-15K-V2 | 0.235 (0.271) | TBD (0.412) | **0.482** (0.506) | TBD (0.581) | TBD (0.821) | TBD (0.495) |
| MED-BBK-9K | 0.004 (0.002) | TBD (0.054) | **0.060** (0.057) | TBD (0.186) | TBD (0.307) | TBD (0.410) |

`python benchmarks/reproduce.py report` also tabulates the suite's feedback ablations, round sweeps, name-matching baselines, and the Hits@1 of each embedding model trained on the PARIS matches.

## Custom Pipelines and Models

A pipeline is a list of steps. Each step receives the current state (graphs, alignment, proposals, embeddings) and returns an updated copy. One PRASE round is `[ParisStep(), Embed(...), ParisStep()]`:

```python
from dataclasses import dataclass, replace

from prase import Embed, ParisConfig, ParisStep, Pipeline, load_benchmark


@dataclass
class KeepConfident:
    minimum: float = 0.2
    name = "keep-confident"

    def run(self, state, context):
        alignment = state.alignment.above(self.minimum)
        context.report(kept=len(alignment))
        return replace(state, alignment=alignment)


    def checkpoint_key(self):
        # Optional: what decides the result, so that a resumed run reuses it.
        return {"minimum": self.minimum}


data = load_benchmark("D-W-15K-V2")
paris = ParisConfig(iterations=10, embedding_weight=0.3)
pipeline = Pipeline([
    ParisStep(paris),
    Embed("gcnalign", options={"device": "cuda"}, seed_threshold=0.1),
    ParisStep(paris),
    KeepConfident(),
])
result = pipeline.run(data.left, data.right, reference=data.reference())
```

A resumed run reuses a finished step when its identity, its seed, and the state it receives are unchanged, under the same PRASE build and NumPy and PyTorch versions, and, for an embedding model, the same PyTorch default type, matrix-product precision, TF32, and determinism settings. A custom step has an identity only with a `checkpoint_key()` method, which returns the step's settings as JSON values, with a digest of every file the step reads, such as `prase.checkpoints.file_digest(path)`; the identity also covers the source of the step's module. A step without the method runs again on resume, and a later step is reused when the step's new result is the same as before. Custom embedding models and literal matchers follow the same rule: without `checkpoint_key()`, a model trains without checkpoints and a PARIS step with the matcher runs again.

### Embedding Feedback

`Embed` trains on PARIS matches scoring at least `seed_threshold` and passes two kinds of feedback to the next `ParisStep`:

- **Proposed pairs** become starting matches. Their scores, multiplied by `proposal_weight` (default 1.0), are the starting probabilities, and PARIS replaces a pair when it finds a better counterpart.
- **Embeddings** enter every candidate score after the first `warmup` iterations: `(1 - embedding_weight) * structural + embedding_weight * cosine`, with `embedding_weight` defaulting to 0.2.

```python
ParisStep(paris, proposal_weight=0.5)   # proposed pairs start at half their scores
ParisStep(paris, use_embeddings=False)  # proposed pairs only
ParisStep(paris, use_proposals=False)   # embedding similarity only
```

```toml
# The same settings in a run file.
[method]
name = "prase"
model = "bootea"
proposal_weight = 0.5
# share_embeddings = false  # proposed pairs only
# propose = false           # embedding similarity only

[method.paris]
embedding_weight = 0.3
```

### Literal Matching

PARIS starts from pairs of equal literals. A `literal_matcher` returns other scored pairs of literal indices:

```python
from prase import Alignment, methods


def casefold_literals(left, right):
    """Pair literals that are equal after case folding."""
    index = {value.casefold(): j for j, value in enumerate(right.literals)}
    return Alignment.from_pairs(
        (i, index[value.casefold()], 1.0) for i, value in enumerate(left.literals) if value.casefold() in index
    )


result = methods.paris(literal_matcher=casefold_literals).run(data.left, data.right)
```

### Embedding Models

An embedding model has a `name` and a method `fit(data, *, seed) -> Embeddings`. `data` holds both graphs (`data.left`, `data.right`), the seed pairs as an `(n, 2)` index array (`data.seeds`), the candidate entities of each graph (`data.candidates`), and the rejected pairs, which must not become training pairs (`data.rejected`). The optional `propose(embeddings, data) -> Alignment` chooses the proposals and should skip the rejected pairs. Without it, each candidate proposes its nearest eligible neighbor.

```python
# my_models.py
import zlib

import numpy as np

from prase import Embeddings
from prase.kg import entity_names


class NameTrigrams:
    """Embed each entity by the hashed character trigrams of its name."""

    name = "name-trigrams"

    def __init__(self, width: int = 256):
        self.width = width

    def fit(self, data, *, seed):
        return Embeddings(self.vectors(data.left), self.vectors(data.right), "cosine")

    def propose(self, embeddings, data):
        # Propose mutual nearest neighbors instead of each entity's nearest neighbor.
        return embeddings.reciprocal_nearest(*data.candidates, excluded=data.rejected)

    def vectors(self, kg):
        vectors = np.zeros((kg.num_entities, self.width))
        for row, name in enumerate(entity_names(kg)):
            text = f"  {(name or '').casefold()} "
            for start in range(len(text) - 2):
                vectors[row, zlib.crc32(text[start:start + 3].encode()) % self.width] += 1
        return vectors
```

```python
from my_models import NameTrigrams

result = methods.prase(NameTrigrams()).run(data.left, data.right, reference=data.reference())
```

### Custom Steps in Run Files

Run files list `[[steps]]` instead of `[method]` and load models and steps by import path:

```toml
[[steps]]
type = "paris"

[[steps]]
type = "embed"
model = "my_models:NameTrigrams"
options = { width = 512 }

[[steps]]
type = "paris"
embedding_weight = 0.3

# KeepConfident from the pipeline example, saved in my_steps.py.
[[steps]]
type = "my_steps:KeepConfident"
minimum = 0.2
```

```console
# Run from the folder that holds my_models.py and my_steps.py, so that Python can import them.
python -m prase run run.toml
```

[`config.py`](https://github.com/qizhyuan/PRASE/blob/main/src/prase/config.py) documents every run-file setting.

## Citation

If you use PRASE, please cite:

```bibtex
@inproceedings{qi2021prase,
  title     = {Unsupervised Knowledge Graph Alignment by Probabilistic Reasoning and Semantic Embedding},
  author    = {Qi, Zhiyuan and Zhang, Ziheng and Chen, Jiaoyan and Chen, Xi and Xiang, Yuejia and Zhang, Ningyu and Zheng, Yefeng},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {2019--2025},
  year      = {2021},
  doi       = {10.24963/ijcai.2021/278},
  url       = {https://doi.org/10.24963/ijcai.2021/278}
}

@inproceedings{qi2021prasemap,
  title     = {{PRASEMap}: A Probabilistic Reasoning and Semantic Embedding based Knowledge Graph Alignment System},
  author    = {Qi, Zhiyuan and Zhang, Ziheng and Chen, Jiaoyan and Chen, Xi and Zheng, Yefeng},
  booktitle = {Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  publisher = {Association for Computing Machinery},
  pages     = {4779--4783},
  year      = {2021},
  doi       = {10.1145/3459637.3481972},
  url       = {https://doi.org/10.1145/3459637.3481972}
}
```

## License

PRASE is released under the [MIT License](LICENSE).
