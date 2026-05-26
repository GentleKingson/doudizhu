# nanopokerlib

Python bindings for the `nanopokerlib` C++ engines.

## API Reference

- Runtime types + endpoint surface: `../docs/nanopokerlib-runtime-api.md`
- Cross-repo semantic conventions: `../docs/repo-conventions.md`

## Development

From the repo root:

```bash
python -m pip install -e nanopokerlib
```

Editable builds cache CMake artifacts under `build/pip/<python-tag>/<env-tag>/` (at the repo root), so re-running
the pip command is usually fast after the first build. The editable build now resets stale CMake cache entries
automatically when the interpreter prefix, generator, or cached tool paths no longer match the current environment.

To rebuild manually (example using Ninja, outputting into the editable package):

```bash
cmake -S nanopokerlib -B build -G Ninja \
  -DPython3_EXECUTABLE="$(which python)" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$(pwd)/nanopokerlib/python/nanopokerlib"
cmake --build build --target _nanopokerlib --parallel
```

## Build profiles

Default builds everything:

```bash
cmake -S nanopokerlib -B build -G Ninja \
  -DNANOPOKERLIB_BUILD=all \
  -DPython3_EXECUTABLE="$(which python)" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$(pwd)/nanopokerlib/python/nanopokerlib"
cmake --build build --target _nanopokerlib --parallel
```
