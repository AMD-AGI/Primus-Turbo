# Contributing to Primus-Turbo

Welcome! We appreciate your interest in contributing to **Primus-Turbo**. This document outlines guidelines and best practices to help you contribute effectively.


## Table of Contents
- [Contributing to Primus-Turbo](#contributing-to-primus-turbo)
  - [Table of Contents](#table-of-contents)
  - [📋 Before You Start](#-before-you-start)
  - [📂 Project Structure](#-project-structure)
  - [🧩 Kernel File Naming Convention](#-kernel-file-naming-convention)
  - [✅ Conventions Are Checked, Not Reviewed](#-conventions-are-checked-not-reviewed)
  - [🎨 Code Style \& Linting](#-code-style--linting)
  - [🌿 Branch Naming Convention](#-branch-naming-convention)
    - [Type](#type)
    - [Scope (optional)](#scope-optional)
  - [📝 Commit Message Convention](#-commit-message-convention)



## 📋 Before You Start

Before contributing, please:

- Read our README.md to understand the project's goals and architecture.

- Check existing issues and discussions.

- If your contribution is significant (new op, backend integration, design refactor), please open an issue first to discuss your approach.

- Make sure you have a working ROCm environment (ROCm ≥ 6.3 recommended, GFX942/GFX950 tested).


## 📂 Project Structure
```
Primus-Turbo/
├── csrc/                  # Core C++/CUDA/HIP sources
│   ├── include/           # Public headers for kernels & common utilities
│   ├── kernels/           # Core CUDA/HIP kernel implementations
│   ├── pytorch/           # PyTorch C++ bindings for custom operators
│   └── jax/               # JAX C++ bindings (XLA FFI handlers)
├── primus_turbo/          # Python package
│   ├── pytorch/           # PyTorch Python frontend
│   ├── jax/               # JAX Python frontend
│   └── triton/            # Triton kernel implementations
├── tests/                 # Unit & Integration Tests
└── benchmark/             # Performance benchmarks
```


## 🧩 Kernel File Naming Convention

Kernel files follow a uniform **`<op>_<dtype>_<role>.py`** pattern so new kernels
slot in predictably. Read the name left-to-right: **operation → data type → role**.

| Segment   | Values | Notes |
|-----------|--------|-------|
| `<op>`    | `gemm`, `grouped_gemm`, `attention`, `rmsnorm`, `swiglu`, ... | The mathematical operation. **Always first**; matches the containing directory. |
| `<dtype>` | see below | **Optional** — omit for dtype-agnostic files (e.g. `grouped_gemm_utils.py`). |
| `<role>`  | `impl`, `kernel`, `utils`, `heuristic`, `registry` | What the file *is*. **Always last**, except that an architecture tag may follow it (see *Architecture-specific kernels*). |

### Two layers, two `<dtype>` vocabularies (intentional)

The frontend and the FlyDSL kernel layer use `<dtype>` at **different abstraction
levels** — do not "unify" them, the difference carries meaning:

* **PyTorch frontend** — `primus_turbo/pytorch/kernels/<op>/<op>_<family>_impl.py`.
  Here `<dtype>` names the **dtype *family* the dispatcher handles**, not one
  concrete format:
  * `fp8` → the fp8 family (tensorwise / rowwise / blockwise / **mx**-blockwise)
  * `fp4` → the fp4 family (currently mx; routed across hipBLASLt / AITER backends)
  * So `gemm_fp8_impl.py` legitimately dispatches into **both** the plain-fp8 and
    the mxfp8 kernels — do **not** rename it to `mxfp8`.

* **FlyDSL kernel layer** — `primus_turbo/flydsl/<op>/<op>_<format>_kernel.py`.
  Here `<dtype>` names **one concrete on-wire format**, because each file is a
  single format-specific kernel: `bf16`, `fp16`, `fp8` (tensorwise), `mxfp8`, `mxfp4`.

### `<role>` vocabulary

* `impl`      — PyTorch frontend op implementation / backend dispatch
* `kernel`    — a FlyDSL / Triton compute kernel
* `utils`     — shared helpers for one op
* `heuristic` — parameter-selection logic for a kernel
* `registry`  — the operation's kernel registry, data only (see *Architecture-specific kernels*)

### Examples

```
primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py               # frontend, fp8 family
primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_utils.py  # dtype-agnostic helpers
primus_turbo/flydsl/gemm/gemm_mxfp8_kernel.py                    # concrete mxfp8 kernel
primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_kernel.py      # concrete fp8 grouped kernel
```

When adding a kernel: take `<op>` from the directory name, pick the **narrowest
correct** `<dtype>` for that layer, and end with the matching `<role>`.

### Architecture-specific kernels

A kernel written for one GPU architecture stays **in the directory of its operation** and
says which architecture it is for with a **tag in its name**: the gfx ID (`gfx950`,
`gfx1250`) as a whole name segment, the way the C++ sources do (`*_gfx950.cu`, which the
build filters by). Product names (`MI455X`) are not used in paths.

| What | Name | Example |
|------|------|---------|
| FlyDSL kernel file | `<op>_<dtype>_<role>_<arch>.py` | `flydsl/grouped_gemm/grouped_gemm_bf16_kernel_gfx1250.py` |
| FlyDSL kernel package (several files) | `<op>/<arch>[_<variant>]/` | `flydsl/attention/gfx1250_mla/`, `flydsl/mega/gfx1250/` |
| PyTorch frontend | `<op>_<backend>_<arch>[_<variant>]_impl.py` | `attention_flydsl_gfx1250_mla_impl.py` |
| C++ / HIP sources | `<name>_<arch>.cu`, or a tagged directory | `hk_attn_fwd_d128_gfx950.cu`, `hipkittens/gfx1250_mla/` |
| Tests, benchmarks | the tag anywhere in the file name | `test_mega_bf16_gfx1250_gemm.py` |

* **Register it in the operation's kernel registry**,
  `primus_turbo/pytorch/kernels/<op>/<op>_registry.py`: one entry per implementation with
  its name, architectures, backend, dtypes, where the code is (`kernel`, `impl`, `sources`),
  the code that decides when it applies (`gates`), and what verifies it (`tests`,
  `benchmarks`). The file is data only and each operation has its own, so pull requests on
  different operations do not meet in one table. `primus_turbo/registry.py` documents the
  fields; `python -m primus_turbo.registry --arch gfx1250` lists the entries, and
  `python -m primus_turbo.registry --owning $(git diff --name-only main) --tests` prints
  the tests of the registered kernels a change touches (their code, the modules of their
  gates, their tests; exit code 1 when the change touches none). gfx1250 code must be
  registered today; the older gfx942 / gfx950 kernels are not yet.
* **Tests on gfx1250.** Tests are skipped on gfx1250 unless they opt in: a test file whose
  name carries the `gfx1250` tag runs there, and a test (or a whole file) without it opts
  in with `pytest.mark.gfx1250`. `tests/conftest.py` needs no edit.
* **Tell other repositories through `primus_turbo.capabilities`.** Code outside
  Primus-Turbo must not probe directories or import internal modules to find out whether
  a feature is there; `capabilities.has("<entry name>", arch="gfx1250")` answers from the
  registry.
* **Keep a launcher and the helpers it calls in one directory.** FlyDSL folds the source
  of a plain Python helper into a kernel's disk-cache key only when the helper is in the
  launcher's directory. A helper that has to be shared across directories must be made
  part of the key by the package that uses it (hash its source into a module-level
  constant the launchers read, as `flydsl/attention/gfx1250_mla/__init__.py` does);
  otherwise an edit to it keeps serving kernels compiled before the edit.
* **Tuning tables and selection logic** belong in a `*_heuristic_<arch>.py` next to the
  kernel (new ones; the existing tables are still inside the kernel files).


## ✅ Conventions Are Checked, Not Reviewed

The conventions above that a program can judge are rules of `tools/conventions`, which
pre-commit runs on every commit and in CI (the `code-lint` job). A finding names the rule;
the rule says why it exists and how to fix the finding:

```bash
python tools/conventions                  # check the tree, new unstaged files included (also: --format json)
python tools/conventions --list-rules     # ARCH001 ... REG004, NAME001
python tools/conventions --explain REG003
```

| Rules | What they hold |
|-------|----------------|
| `ARCH001`–`ARCH003` | Code tagged with an architecture is owned by a kernel registry entry; the tag is the gfx ID as a whole name segment, never a product name. |
| `REG001`–`REG004` | Registry files are well-formed data; everything an entry names exists; the tests it lists run on its architecture; a tagged test or benchmark belongs to an entry. |
| `NAME001` | A kernel file ends with its role (`<op>_<dtype>_<role>[_<arch>].py`). |

* **Adding a convention** means adding a rule: a function `(tree) -> findings` with
  `@rule("<ID>", "<title>")` in a module of `tools/conventions/rules/`, with a docstring
  that gives the reason and the fix, and a test in `tests/pytorch/core/test_conventions.py`
  that shows it firing. Prefer this to a paragraph here that a reviewer has to remember.
* **Files that predate a rule** are listed in `tools/conventions/baseline.txt` (only for a
  rule declared `baseline=True`). A listed file passes, any other file must follow the
  rule, and a listed file that no longer breaks the rule must be removed from the list
  (`--update-baseline`), so the list only shrinks. Adding a line is accepting a new
  exception: say why in the pull request.


## 🎨 Code Style & Linting

We use [**Ruff**](https://docs.astral.sh/ruff/) for Python linting and formatting (it replaces black / isort / autoflake), and `clang-format` for C++/HIP code. All checks run through [pre-commit](https://pre-commit.com/) and are enforced in CI.

Set up once after cloning:
```bash
pip install -r requirements.txt
pre-commit install
```
After this, the hooks run automatically on every `git commit`. To check all files manually (e.g. before opening a PR):
```bash
pre-commit run --all-files
```
You can also run Ruff directly without pre-commit:
```bash
ruff check --fix .   # lint + auto-fix (import sorting, unused imports, ...)
ruff format .        # format code
```
Ruff rules live in `pyproject.toml`; tool versions are pinned in `.pre-commit-config.yaml` and `requirements.txt`.


## 🌿 Branch Naming Convention
Please follow this branch naming convention for all feature and bug fix branches:
```
<type>/<scope>/<short-description>
```

### Type
| Type       | Purpose                                     |
| ---------- | ------------------------------------------- |
| `feat`     | New feature or functionality                |
| `opt`      | Performance optimization or tuning          |
| `fix`      | Bug fix                                     |
| `docs`     | Documentation update                        |
| `refactor` | Code refactoring (no functionality change)  |
| `test`     | Tests and test-related changes              |
| `chore`    | Miscellaneous changes (e.g., build scripts) |
| `ci`       | Continuous integration-related changes      |


### Scope (optional)
The scope typically refers to a module, operator, backend, or feature area. Examples:

- `gemm`, `fp8`, `rmsnorm`

- `pytorch`, `jax`, `triton`, `kernels`

- `build`, `docsite`, `bench`

Use your judgment to choose an appropriate scope that improves readability.


## 📝 Commit Message Convention
We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.
```
<type>(<scope>): <short description>
```
Good Example:
```
feat(gemm): add fp16/bf16 gemm kernel
opt(fp8): improve quantization performance
fix(attention): correct masking in causal attention
```
Bad Example:
```
update gemm
fix bug
```
