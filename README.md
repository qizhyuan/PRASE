# PRASE

PRASE aligns the entities of two knowledge graphs without labeled pairs. It combines probabilistic reasoning with semantic embeddings ([Qi et al., IJCAI 2021](https://www.ijcai.org/proceedings/2021/278)):

1. [PARIS](https://doi.org/10.14778/2078331.2078332) (Suchanek et al., VLDB 2011) matches entities, relations, and attributes by probabilistic reasoning over the graphs' facts and equal literals.
2. An embedding model trains on the confident matches, proposes counterparts for the remaining entities, and scores entity similarity.
3. A second PARIS pass starts from the proposed pairs and adds the embedding similarity to its evidence.

The package provides:

- PARIS reasoning with a C++ engine and a NumPy engine that give identical alignments. The PARIS authors publish their Java implementation at [dig-team/PARIS](https://github.com/dig-team/PARIS).
- Six embedding models in PyTorch: BootEA, GCNAlign, IMUSE, IPTransE, MTransE, and MultiKE.
- Pipelines built from interchangeable steps, with records of every intermediate alignment and score.
- Optional manual review: approved pairs stay fixed and rejected pairs are never matched.
- Benchmark recipes that reproduce the paper's experiments on six datasets.

The original implementation released with the IJCAI 2021 paper is on the [`legacy`](https://github.com/qizhyuan/PRASE/tree/legacy) branch.

## PRASEMap Web Application

[PRASEMap](https://prasemap.qizhy.com/) is a web application built on PRASE, first presented as a demo at
CIKM 2021 ([Qi et al., CIKM 2021](https://doi.org/10.1145/3459637.3481972)). In the browser you can upload
two graphs or choose a built-in pair, run PARIS or PRASE, and review the matches. Approved and rejected
pairs carry over to the next run.

## Installation

PRASE requires Python 3.11 or newer and installs from [PyPI](https://pypi.org/project/prase/):

```console
# PARIS, the pipelines, and the benchmark downloader.
python -m pip install prase
# PyTorch and RapidFuzz for the embedding models.
python -m pip install "prase[train]"
# RDFLib for RDF input.
python -m pip install "prase[rdf]"
```

The wheels for Linux (x86_64 and aarch64), Windows (x64), and macOS (Apple silicon) on Python 3.11 to 3.15 include the compiled PARIS engine. On other platforms pip builds the engine from source, which requires CMake 3.18+ and a C++17 compiler.

To install from a clone of the repository:

```console
git clone https://github.com/qizhyuan/PRASE.git
cd PRASE
# PARIS with the C++ engine, built from source.
python -m pip install .
# The NumPy engine alone, without building the C++ engine.
python -m pip install . --config-settings wheel.cmake=false
```

Embedding models train on the CPU by default. For GPU training, install the CUDA build of PyTorch that the [PyTorch installer](https://pytorch.org/get-started/locally/) selects for your system, for example:

```console
python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
```

The examples below select the GPU with `options={"device": "cuda"}`. Run files take `device = "cuda"` in the model options, and the benchmark runner takes `--device cuda`.

## Quick Start

PARIS on two small graphs:

```python
from prase import KG, methods

left = KG.from_triples(
    [("l:paris", "capital_of", "l:france")],
    [("l:paris", "name", "Paris"), ("l:france", "name", "France")],
)
right = KG.from_triples(
    [("r:paris", "capitalOf", "r:france")],
    [("r:paris", "label", "Paris"), ("r:france", "label", "France")],
)
result = methods.paris().run(left, right)
print(result.alignment.to_ids(left, right))
```

PRASE and SE on a benchmark dataset, after [signing in to Hugging Face](#benchmarks):

```python
from prase import Alignment, load_benchmark, methods

# Download and verify D-W-15K-V2 on first use.
data = load_benchmark("D-W-15K-V2")

# PRASE: PARIS, BootEA, and PARIS again with BootEA's proposals and embeddings.
prase = methods.prase("bootea", options={"device": "cuda"})
result = prase.run(data.left, data.right, reference=data.reference())
print(result.scores)

# SE: GCNAlign trains on the first fold's training links and predicts the other entities.
fold = data.splits[0]
train = data.reference(fold.train)
se = methods.embedding_only("gcnalign", options={"device": "cuda"})
result = se.run(data.left, data.right, seeds=Alignment(train[:, 0], train[:, 1]),
                reference=data.reference(fold.valid + fold.test))
print(result.scores)
```

## Align Your Own Graphs

Each graph consists of relation triples `head<TAB>predicate<TAB>tail` and optional attribute triples `entity<TAB>predicate<TAB>value`. A TOML run file names the inputs, the method, and the output directory:

```toml
# run.toml
[input]
reference = "reference.tsv"  # optional pairs for evaluation
seeds = "seeds.tsv"          # optional known pairs

[input.left]
relations = "left-relations.tsv"
attributes = "left-attributes.tsv"

[input.right]
relations = "right-relations.tsv"
attributes = "right-attributes.tsv"

[method]
name = "prase"
model = "gcnalign"
options = { device = "cpu" }

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

Paths are relative to the run file. Seed and reference files have one `left<TAB>right` pair per line. Every input and method is also available in Python:

| Input | Run File | Python |
| --- | --- | --- |
| TSV triples | `relations` and `attributes` in `[input.left]` and `[input.right]` | `KG.from_tsv("left-relations.tsv", "left-attributes.tsv")` |
| RDF | `rdf = "left.ttl"` in `[input.left]` and `[input.right]` | `KG.from_rdf("left.ttl")` |
| OpenEA folder | `dataset = "path/to/folder"` in `[input]` | `load_dataset("path/to/folder")` |
| Benchmark dataset | `benchmark = "D-W-15K-V2"` in `[input]` | `load_benchmark("D-W-15K-V2")` |

| Method | `[method]` in a Run File | Python |
| --- | --- | --- |
| PARIS | `name = "paris"` | `methods.paris()` |
| PRASE | `name = "prase"`, `model = "bootea"`, `rounds = 1` | `methods.prase("bootea", rounds=1)` |
| SE | `name = "embedding-only"`, `model = "gcnalign"`, with `[input].seeds` | `methods.embedding_only("gcnalign")`, run with `seeds=` |
| STR-Match: edit similarity of names | `name = "str-match"` | `methods.name_match()` |
| EMB-Match: cosine of averaged name word vectors | `name = "emb-match"`, `word_vectors = "data/wiki-news-300d-1M.vec"` | `methods.name_embedding_match("data/wiki-news-300d-1M.vec")` |

The output directory holds the final alignment, the run settings, and one folder per step:

```text
runs/my-alignment/
├── alignment.tsv        left ID, right ID, score
├── run.json             settings and per-step scores
├── 01-paris/            alignment.tsv, predicates.tsv, step.json
├── 02-embed-gcnalign/   proposals.tsv, step.json with the training summary
└── 03-paris/            alignment.tsv, predicates.tsv, step.json
```

### Manual Review

Every method runs without review. To correct an alignment, list approved and rejected pairs in a TSV file and rerun with it into a new output directory:

```text
left	right	decision
http://dbpedia.org/resource/Illinois	http://www.wikidata.org/entity/Q1204	approve
http://dbpedia.org/resource/Illinois	http://www.wikidata.org/entity/Q1297	reject
```

```console
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
for right_id, candidate in session.candidates(illinois, limit=3):
    print(right_id, round(candidate.score, 3))

session.approve(illinois, "http://www.wikidata.org/entity/Q1204")
session.fix_predicate("http://dbpedia.org/ontology/birthPlace", "http://www.wikidata.org/entity/P19")
session.run()
print(session.matches()[:5])
```

## Custom Pipelines and Models

A pipeline is a list of steps. Each step receives the current state (graphs, alignment, proposals, embeddings) and returns an updated copy. One PRASE round is `[ParisStep(), Embed(...), ParisStep()]`:

```python
from dataclasses import dataclass, replace

from prase import Embed, ParisConfig, ParisStep, Pipeline


@dataclass
class KeepConfident:
    minimum: float = 0.2
    name = "keep-confident"

    def run(self, state, context):
        alignment = state.alignment.above(self.minimum)
        context.report(kept=len(alignment))
        return replace(state, alignment=alignment)


paris = ParisConfig(iterations=10, embedding_weight=0.3)
pipeline = Pipeline([
    ParisStep(paris),
    Embed("gcnalign", options={"device": "cuda"}, seed_threshold=0.1),
    ParisStep(paris),
    KeepConfident(),
])
result = pipeline.run(data.left, data.right, reference=data.reference())
```

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


result = methods.paris(literal_matcher=casefold_literals).run(left, right)
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

## Benchmarks

The benchmark collection [qizhy777/prase-benchmarks](https://huggingface.co/datasets/qizhy777/prase-benchmarks) contains the six datasets of the paper: EN-FR-100K-V2, EN-DE-100K-V2, D-W-100K-V2, D-Y-100K-V2, and D-W-15K-V2 from [OpenEA](https://github.com/nju-websoft/OpenEA), and MED-BBK-9K from [industry-eval-EA](https://github.com/ZihengZZH/industry-eval-EA). PRASE pins a revision of the collection and checks the SHA-256 of every file it downloads. Downloads require a Hugging Face account with read access to the collection. The `hf` command comes with PRASE:

```console
# Sign in with an access token from https://huggingface.co/settings/tokens.
hf auth login
# Download and verify all six datasets into the Hugging Face cache.
prase download
# Or download some of them.
prase download D-W-15K-V2 MED-BBK-9K
```

MultiKE and EMB-Match read the English FastText vectors [wiki-news-300d-1M.vec](https://fasttext.cc/docs/en/english-vectors.html):

```console
# Download the 682 MB archive, extract the vectors into data/, and check their SHA-256.
prase download --word-vectors
# Or extract them into another folder.
prase download --word-vectors --vectors-dir path/to/vectors
```

To fetch the vectors by hand:

```console
curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
python -m zipfile -e wiki-news-300d-1M.vec.zip data
```

`benchmarks/reproduce.py` runs the recipes in [`benchmarks/experiments.toml`](https://github.com/qizhyuan/PRASE/blob/main/benchmarks/experiments.toml) and writes each run to `<output>/<experiment>/<dataset>/`. A run stops when its folder already holds a run, so give each reproduction a new `--output` directory and pass the same directory to `list` and `report`. The script and the recipes are in the repository:

```console
# Get the benchmark scripts and install PRASE with the embedding models.
git clone https://github.com/qizhyuan/PRASE.git
cd PRASE
python -m pip install "prase[train]"
# PARIS on every dataset.
python benchmarks/reproduce.py run --method paris --all-datasets --output runs/reproduction
# PRASE with BootEA and with MultiKE on every dataset.
python benchmarks/reproduce.py run --method prase --model bootea --all-datasets --device cuda --output runs/reproduction
python benchmarks/reproduce.py run --method prase --model multike --all-datasets --device cuda --word-vectors data/wiki-news-300d-1M.vec --output runs/reproduction
# SE: train on the benchmark folds and rank the test links.
python benchmarks/reproduce.py run --method se --model gcnalign --dataset D-W-15K-V2 --output runs/reproduction
# One recipe by name, such as PRASE-BootEA with proposed pairs only.
python benchmarks/reproduce.py run --experiment prase-bootea-m --all-datasets --device cuda --output runs/reproduction
# Every recipe and its status, including the feedback ablations, round sweeps, and name-matching baselines.
python benchmarks/reproduce.py list --output runs/reproduction
# Tabulate the completed runs.
python benchmarks/reproduce.py report runs/reproduction
# The entire suite in its own directory.
python benchmarks/reproduce.py run --all --device cuda --word-vectors data/wiki-news-300d-1M.vec --output runs/full-suite
```

### Results

F1 of the final alignment with one PRASE round, seed 0, and embedding training on an NVIDIA RTX 5090. Embedding training is not bit-for-bit reproducible, so scores vary slightly from run to run. The values in parentheses are from the paper.

| Dataset | PARIS | PRASE-BootEA | PRASE-MultiKE |
| --- | --- | --- | --- |
| EN-FR-100K-V2 | 0.933 (0.926) | 0.954 (0.954) | 0.962 (0.954) |
| EN-DE-100K-V2 | 0.952 (0.948) | 0.966 (0.965) | 0.977 (0.972) |
| D-W-100K-V2 | 0.864 (0.854) | 0.891 (0.890) | 0.868 (0.859) |
| D-Y-100K-V2 | 0.986 (0.983) | 0.994 (0.993) | 0.998 (0.996) |
| D-W-15K-V2 | 0.905 (0.897) | 0.917 (0.923) | 0.911 (0.907) |
| MED-BBK-9K | 0.498 (0.499) | 0.578 (0.557) | 0.728 (0.711) |

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
