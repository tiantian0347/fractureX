# Copilot instructions for FEALPy

## Big picture (read this first)
- FEALPy is a finite-element toolkit organized as: `mesh` (geometry/topology) → `functionspace` (DoF mapping/basis) → `fem` (integrators/forms) → `solver` (linear/nonlinear solves).
- Typical data flow is visible in `test/mesh/test_triangle_mesh.py`: build mesh, create `LagrangeFESpace`, assemble `BilinearForm`/`LinearForm`, apply `DirichletBC`, solve.
- The codebase is backend-abstracted: most numerical code should use `from fealpy.backend import backend_manager as bm` instead of raw NumPy/Torch/JAX APIs.

## Core architecture patterns
- Backend dispatch is centralized in `fealpy/backend/manager.py`; backend functions are proxied from `fealpy/backend/base.py` mappings.
- `Mesh` base APIs live in `fealpy/mesh/mesh_base.py`; concrete meshes (e.g. `TriangleMesh`) implement geometry-specific formulas.
- FE spaces wrap mesh + DoF logic (`fealpy/functionspace/lagrange_fe_space.py`), then forms/integrators consume spaces (`fealpy/fem/integrator.py`, `fealpy/fem/bilinear_form.py`).
- Sparse matrices are FEALPy-native (`fealpy/sparse/COOTensor`, `CSRTensor`) with SciPy-like constructors in `fealpy/sparse/__init__.py`.

## Project-specific coding conventions
- Use FEALPy naming conventions from `Develop_note.md`: `NN/NE/NF/NC`, `GD/TD`, and entity names `node/edge/face/cell`.
- Preserve backend/device/dtype context with `bm.context(...)` and backend creation ops (`bm.zeros`, `bm.tensor`, `bm.arange`, etc.).
- Prefer backend-safe update ops (`bm.set_at`, `bm.index_add`) over direct in-place tensor mutation when adding new cross-backend code.
- Keep new APIs consistent with existing mesh/space method names (`entity_measure`, `cell_to_dof`, `basis`, `grad_basis`, etc.).

## Testing and validation workflow
- Install dev deps in editable mode:
  - `pip install -e .[dev]` (or `pip install -e .[dev,optional]` when needed)
- Main CI pattern is in `.github/workflows/python-package.yml`:
  - `pytest test`
- Prefer focused runs while iterating, e.g.:
  - `pytest test/mesh/test_triangle_mesh.py -k grad_shape_function`
- Backend compatibility is important: many tests parametrize `backend in ['numpy', 'pytorch', 'jax']`; preserve this pattern for new tests.
- Ignore legacy test trees unless explicitly working there: root `pytest.ini` excludes `efficiency/pinn`; `test/pytest.ini` excludes `old/opt`.

## Integration points and optional dependencies
- Optional compiled/third-party integrations are controlled by env vars in `external_deps/config.json` (`WITH_PANGULU`, `WITH_MUMPS`, `WITH_P4EST`) and wired via `external_deps/builder.py` in `setup.py`.
- Keep SciPy interop at boundaries (e.g., solver or conversion helpers), not as a replacement for backend-agnostic core math.

## Practical guidance for AI agents
- Prefer editing active modules under `fealpy/` and tests under `test/`; avoid introducing new code into `fealpy/old/` unless request explicitly targets legacy paths.
- When adding numerical features, implement through mesh/space/integrator abstractions first, then expose via package `__init__.py` exports if needed.
- If behavior differs by backend, add/extend parametrized tests to demonstrate parity rather than backend-specific branches in high-level APIs.
