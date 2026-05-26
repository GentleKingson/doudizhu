# setup.py
import hashlib
import multiprocessing  # To get CPU count for parallel builds
import os
import platform
import shutil
import subprocess
import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pathlib import Path

# ---- Project Configuration ----

# Basic package information (modify as needed)
PACKAGE_NAME = "nanopokerlib"
BASE_VERSION = "0"
FALLBACK_VERSION = f"{BASE_VERSION}b0"
AUTHOR = "Zhiyuan Fan (nanomena)"
AUTHOR_EMAIL = "nanomena@outlook.com"
DESCRIPTION = "A vectorized poker environment (CMake build)"
URL = ""  # Add your project URL (e.g., GitHub repository)
LICENSE = ""  # Specify your license (e.g., "MIT", "Apache 2.0")

# Python requirements
PYTHON_REQUIRES = ">=3.7"
# Runtime dependencies
# These are dependencies required AFTER installation to run the Python code.
INSTALL_REQUIRES = ['numpy>=1.20.0']

# Python package details
PYTHON_PACKAGE_NAME = "nanopokerlib"

# C++ Extension details
# The name 'nanopokerlib.nanopokerlib' means the compiled C++ module 'nanopokerlib.so/.pyd'
# will be placed inside the 'nanopokerlib' Python package directory.
CPP_EXTENSION_NAME = f"{PYTHON_PACKAGE_NAME}._{PYTHON_PACKAGE_NAME}"
# Directory containing the root CMakeLists.txt for the C++ code
CMAKE_SOURCE_DIR = str(Path(__file__).resolve().parent)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_build_jobs() -> int:
    for key in ("NANOPOKERLIB_BUILD_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL"):
        raw = os.environ.get(key)
        if not raw:
            continue
        try:
            jobs = int(raw)
        except ValueError:
            continue
        if jobs > 0:
            return jobs
    try:
        return max(1, multiprocessing.cpu_count())
    except NotImplementedError:
        return 2


def _default_cmake_build_dir(ext_name: str, build_type: str) -> str:
    cache_tag = getattr(sys.implementation, "cache_tag", None)
    if not cache_tag:
        cache_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    prefix_tag = _build_prefix_tag()
    return os.path.join(
        str(REPO_ROOT),
        "build",
        "pip",
        cache_tag,
        prefix_tag,
        build_type.lower(),
        ext_name.replace(".", "_"),
    )


def _build_prefix_tag() -> str:
    prefix_path = os.path.realpath(sys.prefix)
    prefix_name = "".join(
        ch if ch.isalnum() else "-"
        for ch in os.path.basename(prefix_path)
    )
    prefix_name = prefix_name.strip("-") or "env"
    prefix_hash = hashlib.sha1(prefix_path.encode("utf-8")).hexdigest()[:12]
    return f"{prefix_name}-{prefix_hash}"


def _read_cmake_cache_entries(cache_path: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    with open(cache_path, "r", encoding = "utf-8") as cache_file:
        for raw_line in cache_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            key_type, separator, value = line.partition("=")
            if not separator:
                continue
            key, type_separator, _value_type = key_type.partition(":")
            if not type_separator:
                continue
            entries[key] = value
    return entries


def _cache_path_is_missing(path_value: str | None) -> bool:
    if not path_value:
        return False
    if path_value.endswith("-NOTFOUND"):
        return False
    if not os.path.isabs(path_value):
        return False
    return not os.path.exists(path_value)


def _collect_cmake_cache_reset_reasons(
    cache_path: str,
    *,
    expected_generator: str | None,
    expected_python_executable: str,
    expected_make_program: str | None,
) -> list[str]:
    cache_entries = _read_cmake_cache_entries(cache_path)
    reset_reasons: list[str] = []

    critical_path_keys = (
        "CMAKE_MAKE_PROGRAM",
        "CMAKE_C_COMPILER",
        "CMAKE_CXX_COMPILER",
        "Python3_EXECUTABLE",
        "_Python3_EXECUTABLE",
    )
    for key in critical_path_keys:
        cached_path = cache_entries.get(key)
        if _cache_path_is_missing(cached_path):
            reset_reasons.append(f"{key} points to missing path: {cached_path}")

    cached_generator = cache_entries.get("CMAKE_GENERATOR")
    if expected_generator and cached_generator and cached_generator != expected_generator:
        reset_reasons.append(
            f"CMAKE_GENERATOR changed from {cached_generator!r} to {expected_generator!r}"
        )

    cached_python_executable = cache_entries.get("Python3_EXECUTABLE")
    if (
        cached_python_executable
        and os.path.realpath(cached_python_executable)
        != os.path.realpath(expected_python_executable)
    ):
        reset_reasons.append(
            "Python3_EXECUTABLE changed from "
            f"{cached_python_executable!r} to {expected_python_executable!r}"
        )

    cached_make_program = cache_entries.get("CMAKE_MAKE_PROGRAM")
    if expected_make_program and cached_make_program:
        if os.path.realpath(cached_make_program) != os.path.realpath(expected_make_program):
            reset_reasons.append(
                f"CMAKE_MAKE_PROGRAM changed from {cached_make_program!r} to {expected_make_program!r}"
            )

    return reset_reasons


def _clear_cmake_build_state(build_dir: str) -> None:
    generated_entries = (
        "CMakeCache.txt",
        "CMakeFiles",
        "build.ninja",
        ".ninja_deps",
        ".ninja_log",
        "cmake_install.cmake",
        "compile_commands.json",
        "_deps",
        "pheval_build",
    )
    for entry_name in generated_entries:
        entry_path = os.path.join(build_dir, entry_name)
        if not os.path.lexists(entry_path):
            continue
        if os.path.isdir(entry_path) and not os.path.islink(entry_path):
            shutil.rmtree(entry_path)
            continue
        os.remove(entry_path)


def _select_cmake_generator() -> tuple[str | None, str | None]:
    env_generator = os.environ.get("CMAKE_GENERATOR")
    if env_generator:
        requested_make_program = (
            shutil.which("ninja")
            if env_generator == "Ninja"
            else None
        )
        return env_generator, requested_make_program

    if platform.system() == "Windows":
        return None, None

    ninja_path = shutil.which("ninja")
    if ninja_path:
        return "Ninja", ninja_path
    return None, None


def _run_git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd = REPO_ROOT,
        stderr = subprocess.DEVNULL,
        text = True,
    ).strip()


def compute_version() -> str:
    """Return a PEP 440-compatible beta version with git date+hash when available."""
    override = os.environ.get("NANOPOKERLIB_VERSION") or os.environ.get("NANOGAMEKITS_VERSION")
    if override:
        return str(override).strip()

    try:
        sha = _run_git(["rev-parse", "--short", "HEAD"])
        date = _run_git(["show", "-s", "--format=%cd", "--date=format:%Y%m%d", "HEAD"])
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return FALLBACK_VERSION

    if not sha or not date:
        return FALLBACK_VERSION

    # Example: 0b20251227+gabc1234
    return f"{BASE_VERSION}b{date}+g{sha}"


# ---- Helper Classes for CMake Build ----

class CMakeExtension(Extension):
    """
    Custom setuptools Extension class that holds the path to the CMake source directory.
    """

    def __init__(self, name, sourcedir = ''):
        super().__init__(name, sources = [])  # No sources needed here, CMake handles it
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """
    Custom setuptools build_ext command to drive the CMake build process.
    """

    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        # Ensure paths are absolute
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_source_dir = ext.sourcedir
        package_version = self.distribution.get_version()  # Get version from setup()

        print(f"\n--- Building extension for {ext.name} ---")
        print(f"CMake source directory: {cmake_source_dir}")
        print(f"Output directory: {extdir}")
        print(f"Package version: {package_version}")

        repo_root = os.path.abspath(os.path.join(cmake_source_dir, os.pardir))

        # --- CMake Configuration Arguments ---
        cmake_args = [
            # Tell CMake where to put the final Python extension library
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            # Explicitly tell CMake to use the Python interpreter that is running
            # this build.  Without this hint CMake may locate another interpreter
            # (e.g. the system Python 3.10 on some deployments) which does not have
            # the matching development headers available, leading to build errors.
            f"-DPython3_EXECUTABLE={sys.executable}",
            # Pass package version info to CMake (can be used to define C++ macro)
            f"-DPACKAGE_VERSION_INFO={package_version}"
        ]

        requested_generator, requested_make_program = _select_cmake_generator()
        if requested_generator:
            cmake_args = ["-G", requested_generator, *cmake_args]

        # Prefer local pybind11 sources when available to avoid FetchContent network work.
        local_pybind11 = os.path.join(repo_root, "external", "pybind11")
        if os.path.exists(os.path.join(local_pybind11, "CMakeLists.txt")):
            cmake_args.append(f"-DFETCHCONTENT_SOURCE_DIR_PYBIND11={os.path.abspath(local_pybind11)}")
            print(f"Using local pybind11: {os.path.abspath(local_pybind11)}")

        # --- Optional offline dependency hints ---
        # CMakeLists.txt supports `-DPHEVAL_DIR=...` to skip FetchContent cloning of PokerHandEvaluator.
        pheval_dir = os.environ.get("PHEVAL_DIR")
        if not pheval_dir:
            # Look for previously-fetched deps in common build layouts:
            # - <repo-root>/build/_deps/... when building with `-B build`
            pheval_candidates = [
                os.path.join(repo_root, "external", "PokerHandEvaluator", "cpp"),
                os.path.join(repo_root, "build", "_deps", "pokerhandeval-src", "cpp"),
            ]
            for local_pheval in pheval_candidates:
                if os.path.exists(os.path.join(local_pheval, "include")):
                    pheval_dir = local_pheval
                    break
        if pheval_dir:
            pheval_dir = os.path.abspath(pheval_dir)
            cmake_args.append(f"-DPHEVAL_DIR={pheval_dir}")
            print(f"Using PHEVAL_DIR: {pheval_dir}")

        # --- Build Type (Debug/Release) ---
        build_type = "Debug" if self.debug else "Release"
        build_args = ["--config", build_type]
        print(f"Build type: {build_type}")

        num_jobs = _get_build_jobs()
        print(f"Parallel build jobs: {num_jobs}")

        # --- Platform-Specific Configuration ---
        if platform.system() == "Windows":
            # Add output directory for multi-config generators (Visual Studio)
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{build_type.upper()}={extdir}"]
            # Specify generator architecture (needed for VS)
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
        else:
            # Specify build type for single-config generators (Makefiles, Ninja)
            cmake_args += [f"-DCMAKE_BUILD_TYPE={build_type}"]

        build_args += ["--parallel", str(num_jobs)]

        # --- Environment Variables (Optional) ---
        env = os.environ.copy()
        # Signal to CMake that we are building via setuptools (optional, for CMake logic)
        env['SETUPTOOLS_BUILD'] = '1'
        # Avoid setting CXXFLAGS here if possible; prefer passing defines via cmake_args

        # --- Run CMake ---
        # Create temporary build directory
        # Use a unique name based on the extension to avoid conflicts
        build_temp_override = os.environ.get("NANOPOKERLIB_CMAKE_BUILD_DIR")
        if build_temp_override:
            build_temp = os.path.abspath(build_temp_override)
        elif _env_truthy("NANOPOKERLIB_PERSISTENT_BUILD", default = True):
            build_temp = _default_cmake_build_dir(ext.name, build_type)
        else:
            build_temp = os.path.join(self.build_temp, ext.name.replace('.', '_'))
        os.makedirs(build_temp, exist_ok = True)
        print(f"Temporary build directory: {build_temp}")

        cache_path = os.path.join(build_temp, "CMakeCache.txt")
        if os.path.exists(cache_path):
            reset_reasons = _collect_cmake_cache_reset_reasons(
                cache_path,
                expected_generator = requested_generator,
                expected_python_executable = sys.executable,
                expected_make_program = requested_make_program,
            )
            if reset_reasons:
                print("\n--- Clearing stale CMake cache ---")
                for reason in reset_reasons:
                    print(f"Reason: {reason}")
                _clear_cmake_build_state(build_temp)

        print("\n--- Running CMake configure ---")
        print(f"Command: cmake {cmake_source_dir} {' '.join(cmake_args)}")
        try:
            subprocess.check_call(
                ["cmake", cmake_source_dir] + cmake_args, cwd = build_temp, env = env
            )
        except subprocess.CalledProcessError:
            if not os.path.exists(cache_path):
                raise
            print("\n--- Retrying CMake configure after clearing cached state ---")
            _clear_cmake_build_state(build_temp)
            subprocess.check_call(
                ["cmake", cmake_source_dir] + cmake_args, cwd = build_temp, env = env
            )

        print("\n--- Running CMake build ---")
        # Target name should match pybind11_add_module name in CMakeLists.txt
        # Extracts 'nanopokerlib' from 'nanopokerlib.nanopokerlib'
        target_name = ext.name.split('.')[-1]
        print(f"Command: cmake --build . --target {target_name} {' '.join(build_args)}")
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target_name] + build_args, cwd = build_temp
        )

        print("\n--- Build finished successfully for target:", target_name, "---\n")


# ---- Helper function to read README ----
def read_readme(fname = "README.md"):
    """Reads the README file for the long description."""
    try:
        with open(Path(CMAKE_SOURCE_DIR) / fname, "r", encoding = "utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        print("Warning: README.md not found, using short description.")
        return DESCRIPTION  # Fallback to short description


# ---- Main Setup Function ----
setup(
    # --- Core Metadata ---
    name = PACKAGE_NAME,
    version = compute_version(),
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    description = DESCRIPTION,
    long_description = read_readme(),
    long_description_content_type = "text/markdown",
    url = URL,
    license = LICENSE,

    # --- Build Configuration ---
    # Define the C++ extension module using CMake
    ext_modules = [CMakeExtension(CPP_EXTENSION_NAME, sourcedir = CMAKE_SOURCE_DIR)],
    # Use the custom CMake build command
    cmdclass = {"build_ext": CMakeBuild},

    # --- Python Package Information ---
    packages = find_packages(where = "python", include = ["nanopokerlib*"]),
    package_dir = {"": "python"},
    include_package_data = False,

    # --- Dependencies ---
    install_requires = INSTALL_REQUIRES,
    python_requires = PYTHON_REQUIRES,

    # --- PyPI Classifiers ---
    # Choose appropriate classifiers from https://pypi.org/classifiers/
    classifiers = [
        "Development Status :: 3 - Alpha",  # Or "4 - Beta", "5 - Production/Stable"
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        f"License :: OSI Approved :: {LICENSE} License" if LICENSE else "License :: OSI Approved",
        "Operating System :: OS Independent",  # Or specify supported OSs
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        # Add future Python versions if compatible
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Games/Entertainment :: Simulation",  # Relevant topic
    ],

    # --- Other Options ---
    zip_safe = False,  # C++ extensions require this to be False
)
