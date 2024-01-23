""" Scripts that deletes sections in the RKF files that are not needed for the tests in order to reduce the file size. """
import pathlib as pl
from typing import Sequence

from scm.plams import KFFile

REMOVE_SECTIONS = [
    "SCF",
    "Core",
]

SECTION_PATTERNS = [
    "ZlmFit_",
]


def get_rkf_files(path: str | pl.Path) -> list[pl.Path]:
    """Returns a list of all RKF files in the directory."""
    path = pl.Path(path) if isinstance(path, str) else path
    return [f for f in path.glob("*.rkf")]


def get_sections_to_remove(kf_file: KFFile) -> list[str]:
    """Returns a list of all sections in the RKF file that are not needed for the tests."""
    sections = list(kf_file.get_skeleton().keys())
    remove_sections = [section for section in sections if section in REMOVE_SECTIONS]
    for pattern in SECTION_PATTERNS:
        remove_sections.extend([section for section in sections if section.startswith(pattern)])
    return remove_sections


def remove_sections(kf_file: KFFile, sections_to_be_removed: Sequence[str], name: str) -> None:
    """Removes all sections in the RKF file that are not needed for the tests."""
    print(f"Removing sections: {sections_to_be_removed} in {name}")
    for section in sections_to_be_removed:
        kf_file.delete_section(section)


def main():
    file_path = pl.Path(__file__).parent
    rkf_files = get_rkf_files(file_path)
    rkf_files = [file_path / rkf_file for rkf_file in rkf_files]

    for rkf in rkf_files:
        kf_file = KFFile(str(file_path / rkf))
        sections_to_be_removed = get_sections_to_remove(kf_file)
        remove_sections(kf_file, sections_to_be_removed, rkf.stem)


if __name__ == "__main__":
    main()
