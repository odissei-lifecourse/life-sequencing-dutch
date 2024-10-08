"""Translate requirements from Snellius to regular pip.

This script translates requirements from Snellius to regular pip, handling the following special cases:
- Packages that need installing from a link with `pip install -f`, such as some torch packages.
- Drops the certain packages as they seem specific to snellius.
- Packages for development are collected in PKG_DEV; their versions are not pinned and
  they are added at the end of the resulting txt file.
"""

import re
from itertools import compress
from pathlib import Path

# Input and output file paths
infile = "requirements/source.txt"
outfile_regular = "requirements/regular.txt"
outfile_snellius = "requirements/snellius.txt"

# URL for finding specific torch packages
FIND_TORCH_LINKS = "https://data.pyg.org/whl/torch-2.1.2+cu121.html"

# Packages to find using special links
PKG_FIND_LINKS = []

# Packages to drop, packages without version requirements
PKG_NOT_PIN = ["pyparsing", "torch_scatter", "torch_sparse"]
PKG_DEV = ["pre-commit", "ruff"]
PKG_DROP = ["blist", "triton", "mpi4py", "nvidia-nccl-cu12"]
PKG_IGNORE = PKG_DROP + PKG_DEV


def read_lines(filename):
    """Read lines from a file and strip trailing whitespace."""
    with Path(filename).open("r") as file:
        return [line.rstrip() for line in file]


def write_lines(filename, lines):
    """Write lines to a file."""
    with Path(filename).open("w") as f:
        for line in lines:
            f.write(f"{line}\n")


def parse_line(line, pkg_not_pin=PKG_NOT_PIN, pkg_find_links=PKG_FIND_LINKS):
    """Parse a line to determine if it should be transformed or skipped.

    Args:
        line (str): line read from a txt file with output from `pip freeze`.
        pkg_not_pin (list, optional): packages whose version should not be pinned.
        pkg_find_links (list, optional): packages whose installation files should be
        found from certain URLs.

    """
    check_ignore = any(pkg in line for pkg in PKG_IGNORE)
    if check_ignore:
        return False, None

    if " @ " not in line:  # identifies non-module packages
        check_find_links = [pkg in line for pkg in pkg_find_links]
        if any(check_find_links):
            pkg_name = next(compress(pkg_find_links, check_find_links))
            return True, pkg_name

        check_no_version = [pkg in line for pkg in pkg_not_pin]
        if any(check_no_version):
            pkg_name = next(compress(pkg_not_pin, check_no_version))
            pkg_name = pkg_name.split("==")[0]
            return True, pkg_name

        return True, line

    return False, convert_linked_package(line, pkg_not_pin)


def convert_linked_package(line, pkg_not_pin):
    """Convert a package link to a regular pip requirement.

    Args:
      line (str): line in a requirements.txt file from snellius. Of the form `"package @ /path/to/file"`.
      pkg_not_pin (list, optional): Packages for which not to require a specific version.
        This can be useful when it is not possible to have the exact same versions on snellius and
        in a regular virtual environment.
    """
    expected_n_parts = 3
    parts = line.split()
    assert parts[1] == "@"  # noqa: S101
    assert len(parts) == expected_n_parts  # noqa: S101
    pkg, loc = parts[0], parts[2]
    pkg_version = loc.split("/")[-1]

    if pkg in pkg_not_pin:
        return pkg

    # Find version in the package URL
    pattern = r"\d+\.\d+\.\d+|\d+\.\d+"
    result = re.search(pattern, pkg_version)
    if result:
        version = result.group(0)
        return f"{pkg}=={version}"
    return None


def main():
    """Main function to read, process, and write requirements."""
    lines = read_lines(infile)

    regular = [f"--find-links {FIND_TORCH_LINKS}"]
    snellius = [f"--find-links {FIND_TORCH_LINKS}"]
    for line in lines:
        for_snellius, parsed_line = parse_line(line)
        if parsed_line:
            regular.append(parsed_line)
            if for_snellius:
                snellius.append(parsed_line)

    for pkg in PKG_DEV:
        regular.append(pkg)
        snellius.append(pkg)

    write_lines(outfile_regular, regular)
    write_lines(outfile_snellius, snellius)


if __name__ == "__main__":
    main()
