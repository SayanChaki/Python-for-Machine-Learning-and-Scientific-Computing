# Python and Machine Learning with PyTorch

A Graduate level teaching course for a beginner-to-intermediate for Institut d'Optique Graduate School M1/M2 level students batch of 2026. Beamer presentations and  Colab notebooks are provided here.



## Prerequisites

**For students:** no prior Python required. No local installation required either,
since everything runs in Colab. A Google account is the only hard requirement.


## Running the notebooks

1. Upload the `.ipynb` to Google Drive, or open it directly at
   [colab.research.google.com](https://colab.research.google.com/).
2. Enable the GPU: **Runtime > Change runtime type > Hardware accelerator > GPU**.
   Every notebook works on CPU, but the timing comparisons in notebooks 1 and 3
   are only interesting with a GPU.
3. Run the cells in order with `Shift + Enter`.

Each notebook follows the same shape:

```
Markdown header  ->  Setup  ->  Theory in Practice  ->  Challenge
```

Cells marked **TODO** raise `NotImplementedError` on purpose. Fill them
in; worked solutions follow in the next cells, so you can self-check.


User-adjustable parameters (`K`, `BATCH_SIZE`, `TRUE_K`, `SIZE`, `EPOCHS`) sit at
the top of the section that uses them, so you can re-run a block with a
different value without hunting through the code.

---

## Modules for Day 1

### Module 1: Python Fundamentals and PyTorch Tensors


Python variables and types, lists, dictionaries, tuples and sets. Control flow
and functions, including `*args` and `**kwargs`. Then tensors: what rank means,
creation, indexing, reductions, reshaping, broadcasting, NumPy interoperability,
devices, and a first look at autograd.

By the end a student can:

- read and write basic Python
- explain the difference between a list and a tensor
- create tensors and check `shape`, `dtype` and `device` when something breaks
- distinguish `*` (elementwise) from `@` (matrix product)
- move a computation to the GPU

**Challenge:** implement `standardise`, `pairwise_sq_dist` and `min_max_scale`
using only tensor operations. `pairwise_sq_dist` is deliberate groundwork for
`torch.cdist` in Module 4.



### Module 2: Functions, Classes and `nn.Module`


Scope and the LEGB rule, the mutable default argument trap, lambdas, generators.
Then classes: `__init__`, `self`, methods, dunder methods, inheritance and
`super()`. Finally `nn.Module`, a full training loop, evaluation mode, and saving
weights.

The single idea to land: **a PyTorch model is a Python class.** `__init__`
declares what the model owns, `forward` declares what it does.

**Challenge:** build `DeepNet` whose depth is a constructor argument, using
`nn.ModuleList`, then demonstrate why a plain Python list of layers silently
fails to register parameters.

---

### Module 3: DataLoaders and Exploratory Data Analysis



The `Dataset` / `DataLoader` split: what a sample is, versus how samples are
served. Custom datasets for images and for CSV files. `transforms`, batching,
shuffling, workers. Then EDA: batch statistics, image grids, class balance,
correlation matrices computed with tensor operations.

- **Part A (images):** MNIST via `torchvision`, then a custom `Dataset` class
  producing provably identical output.
- **Part B (CSV):** Iris written to disk as a real CSV, then a `CSVDataset` that
  takes normalisation statistics as arguments so the training split's mean and
  standard deviation can be reused for validation.
- **Part C (EDA):** the checklist, applied to both datasets.

---

### Module 4: KNN and K-Means


KNN as supervised, non-parametric and lazy. Distance metrics and why feature
scaling is mandatory. The effect of `k` on the decision boundary. The curse of
dimensionality. Then K-Means: the within-cluster sum of squares objective,
Lloyd's algorithm as expectation-maximisation, the elbow method, k-means++, and
the cases where K-Means fails.

Both algorithms reduce to one primitive, `torch.cdist`, plus `topk` or `argmin`.
Because they are written in tensor operations, they run unchanged on the GPU.

- **Part D:** `TorchKNN` from scratch, a `k` sweep, `L1` versus `L2`, and decision
  boundaries plotted for `k = 1, 5, 25, n`.
- **Part E:** Lloyd's algorithm with a vectorised M-step, inertia tracking, the
  elbow curve, an initialisation study, and a two-moons failure case.

**Challenges:** distance-weighted KNN, a from-scratch silhouette score used to
select `k`, and both algorithms applied to the Iris data from Part B.


## Common student errors, and where they are addressed

These are the mistakes that recur every year. Each is raised explicitly rather
than left for students to discover under time pressure.

| Error | Where it is covered |
|---|---|
| `*` used where `@` was meant | Module 1, slides and notebook |
| Tensors on different devices | Module 1 notebook, with the error triggered on purpose |
| `from_numpy` sharing memory unexpectedly | Module 1 notebook |
| Forgetting `super().__init__()` | Module 2, demonstrated on a plain class first |
| Calling `model.forward(x)` instead of `model(x)` | Module 2 slides |
| Forgetting `optimizer.zero_grad()` | Module 2 slides |
| A plain list of layers not registering parameters | Module 2 challenge |
| Mutable default arguments | Module 2 notebook |
| `Normalize` placed before `ToTensor` | Module 3 slides |
| Forgetting `.permute(1, 2, 0)` before `imshow` | Module 3 slides and notebook |
| Normalising with statistics from the full dataset | Module 3, Part B |
| Distance methods used on unscaled features | Module 4 slides and Part D |
| Treating the two meanings of `k` as the same thing | Module 4, opening slide |

---

## Dependencies

Colab already provides all of these. The first cell of each notebook installs
them anyway, so the notebooks also run outside Colab.

```
torch
torchvision
matplotlib
pandas
numpy
scikit-learn
```

---
