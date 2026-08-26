#!/usr/bin/env python3
"""Export only student controller modules as a submission-ready zip file."""

from __future__ import annotations

import argparse
import ast
import re
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
CONTROLLERS_ROOT = SOURCE_ROOT / "controllers"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "formula110-student-controllers.zip"
MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


def controller_module_name(value: str) -> str:
    """Validate a dotted module name inside the controllers package."""
    if value.endswith(".py") or MODULE_NAME_PATTERN.fullmatch(value) is None or not value.startswith("controllers."):
        raise argparse.ArgumentTypeError(f"expected a dotted controllers module name, got {value!r}")
    return value


def module_source(module_name: str) -> Path:
    """Resolve one controller module to its project source file."""
    return SOURCE_ROOT.joinpath(*module_name.split(".")).with_suffix(".py")


def selected_sources(module_names: tuple[str, ...], *, all_controllers: bool) -> tuple[Path, ...]:
    """Return controller files to include in the student archive."""
    if all_controllers:
        if module_names:
            raise ValueError("do not provide module names together with --all-controllers")
        sources = tuple(
            sorted(
                path
                for path in CONTROLLERS_ROOT.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )
        )
    else:
        if not module_names:
            raise ValueError("provide at least one controller module or use --all-controllers")
        pending = [module_source(module_name) for module_name in module_names]
        sources_by_path: set[Path] = set()
        while pending:
            source = pending.pop()
            if source in sources_by_path:
                continue
            if not source.is_file():
                raise FileNotFoundError(f"controller source file not found: {source.relative_to(PROJECT_ROOT)}")
            sources_by_path.add(source)
            pending.extend(
                dependency for dependency in local_controller_dependencies(source) if dependency not in sources_by_path
            )
        package_init = CONTROLLERS_ROOT / "__init__.py"
        if package_init.is_file():
            sources_by_path.add(package_init)
        sources = tuple(sorted(sources_by_path))

    missing = tuple(path for path in sources if not path.is_file())
    if missing:
        names = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(f"controller source file not found: {names}")
    return sources


def local_controller_dependencies(source: Path) -> tuple[Path, ...]:
    """Find statically imported modules within the controllers package."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names.update(alias.name for alias in node.names if alias.name.startswith("controllers."))
        elif isinstance(node, ast.ImportFrom):
            imported_from = node.module or ""
            if node.level > 0:
                imported_from = f"controllers.{imported_from}".rstrip(".")
            if imported_from.startswith("controllers."):
                module_names.add(imported_from)
            elif imported_from == "controllers":
                module_names.update(f"controllers.{alias.name}" for alias in node.names)
    return tuple(sorted(candidate for name in module_names if (candidate := module_source(name)).is_file()))


def export_controllers(
    module_names: tuple[str, ...],
    output: Path,
    *,
    all_controllers: bool = False,
) -> Path:
    """Write selected controller sources beneath controllers/ in a zip file."""
    sources = selected_sources(module_names, all_controllers=all_controllers)
    resolved_output = output.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(resolved_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sources:
            archive.write(source, arcname=source.relative_to(SOURCE_ROOT))
    return resolved_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "modules",
        nargs="*",
        type=controller_module_name,
        help="controller modules to export, such as controllers.minimum_viable",
    )
    parser.add_argument(
        "--all-controllers",
        action="store_true",
        help="export the complete src/controllers directory instead of selected modules",
    )
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
        output = export_controllers(tuple(args.modules), args.output, all_controllers=bool(args.all_controllers))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(output)


if __name__ == "__main__":
    main()
