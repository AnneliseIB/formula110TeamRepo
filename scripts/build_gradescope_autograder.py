#!/usr/bin/env python3
"""Build a Gradescope autograder archive for two Formula 110 controllers."""

from __future__ import annotations

import argparse
import json
import re
import stat
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = PROJECT_ROOT / "autograder" / "gradescope"
RACING_SOURCE = PROJECT_ROOT / "src" / "racing"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "formula110-gradescope-autograder.zip"
MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


def module_name(value: str) -> str:
    """Validate a dotted Python module name supplied on the command line."""
    if value.endswith(".py") or MODULE_NAME_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(f"not a dotted Python module name: {value!r}")
    return value


def build_config(first_module: str, second_module: str) -> dict[str, object]:
    """Return the instructor-visible grading configuration."""
    return {
        "schema_version": 1,
        "modules": {
            "minimum_viable": first_module,
            "improved": second_module,
        },
        "control_function": "control",
        "seeds": [110, 2026],
        "duration_seconds": 30.0,
        "trial_timeout_seconds": 30.0,
        "rubric": {
            "minimum_pyright_strict": 5.0,
            "improved_pyright_strict": 5.0,
            "minimum_ruff_lint": 2.5,
            "improved_ruff_lint": 2.5,
            "minimum_ruff_format": 2.5,
            "improved_ruff_format": 2.5,
            "minimum_control": 5.0,
            "improved_control": 5.0,
            "minimum_lap": 20.0,
            "minimum_no_damage": 15.0,
            "minimum_no_walls": 15.0,
            "improved_survival": 10.0,
            "improved_distance": 10.0,
        },
    }


def _write_bundle_tree(root: Path, config: dict[str, object]) -> None:
    for template_name in ("setup.sh", "run_autograder", "grade.py", "race_worker.py", "control_worker.py"):
        source = TEMPLATE_ROOT / template_name
        destination = root / template_name
        destination.write_bytes(source.read_bytes())
        if template_name in ("setup.sh", "run_autograder", "race_worker.py", "control_worker.py"):
            destination.chmod(0o755)

    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    trusted_root = root / "trusted" / "racing"
    for source in RACING_SOURCE.rglob("*"):
        if not source.is_file() or "__pycache__" in source.parts or source.suffix == ".pyc":
            continue
        destination = trusted_root / source.relative_to(RACING_SOURCE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def _write_zip(tree: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(path for path in tree.rglob("*") if path.is_file()):
            relative = source.relative_to(tree)
            info = zipfile.ZipInfo.from_file(source, arcname=str(relative))
            info.compress_type = zipfile.ZIP_DEFLATED
            if relative.as_posix() in {"setup.sh", "run_autograder", "race_worker.py", "control_worker.py"}:
                info.external_attr = (stat.S_IFREG | 0o755) << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_archive(first_module: str, second_module: str, output: Path) -> Path:
    """Create and return a Gradescope-ready zip archive."""
    if first_module == second_module:
        raise ValueError("the minimum-viable and improved module names must differ")
    if not TEMPLATE_ROOT.is_dir() or not RACING_SOURCE.is_dir():
        raise FileNotFoundError("run this script from a complete Formula 110 project checkout")

    config = build_config(first_module, second_module)
    with tempfile.TemporaryDirectory(prefix="formula110-autograder-") as temporary_directory:
        tree = Path(temporary_directory)
        _write_bundle_tree(tree, config)
        _write_zip(tree, output.resolve())
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("minimum_viable_module", type=module_name, help="first required controller module")
    parser.add_argument("improved_module", type=module_name, help="second, improved controller module")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"archive path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        output = build_archive(args.minimum_viable_module, args.improved_module, args.output)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(output)


if __name__ == "__main__":
    main()
