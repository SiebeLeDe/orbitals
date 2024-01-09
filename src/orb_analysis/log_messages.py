""" Module containing messages for logging and providing feedback to the user. """
import textwrap

OVERLAP_MATRIX_NOTE = "Overlap (S in arbitrary units [a.u.]) is guaranteed 0.0 when the irreps do not match and when both are unoccupied [LUMO-LUMO; non-physical]. "
OVERLAP_MATRIX_NOTE += "HOMO-HOMO overlaps are related to Pauli repulsion, and HOMO-LUMO overlaps to stabilizing orbital interactions"

INTERACTION_MATRIX_NOTE = """
SFO interaction matrix that contains the information for:
1) HOMO-HOMO interactions: Pauli repulsion indicator; calculated through S^2 * 100 with units [au^2]. Higher means more Pauli repulsion.
2) HOMO-LUMO interactions: favorable orbital interactions indicator (=SCF process); calculated through S^2 / energy gap * 100 with units [au^2 / eV].
Higher is better with the assumption that the SFOs are *not* degenerate. Otherwise, -S * 100 is the relevant indicator for the orbital interaction stabilization.
3) LUMO-LUMO interactions: simply 0.0 because they have no physical meaning in this Kohm-Sham MO fragment-based theory. Note: the "* 100" factor is just for easier reading.
Source: Orbital Interactions in Chemistry, Albright, p24
"""


def format_message(text, width=130):
    return "\n" + textwrap.fill(text, width) + "\n"


def _get_header(length=130) -> str:
    header_text = "Calculation Analysis"
    fill_char = "-"
    padding = (length - len(header_text)) // 2
    return f"\n{fill_char * padding}{header_text}{fill_char * padding}"


def _get_calc_message(restricted: bool, calc_name: str) -> str:
    if restricted:
        return f"This is a restricted calculation for {calc_name}."
    return f"This is a unrestricted calculation for {calc_name}."


def _get_irrep_message(irrep: str | None):
    irrep_message = "The selected orbitals belong to "
    if irrep is None:
        irrep_message += "all irreps present in the calculation."
    else:
        irrep_message += f"the {irrep} irrep."
    return irrep_message


def _get_orb_range_message(orb_range: tuple[int, int]) -> str:
    return f"Orbitals are selected from HOMO-{orb_range[0]-1} to LUMO+{orb_range[1]-1}."


def _get_spin_message(spin: str):
    return f"All considered orbitals belong to spin {spin}. Note that this is only relevant for unrestricted calculations"


def calc_analyzer_call_message(restricted: bool, calc_name: str, orb_range: tuple[int, int], irrep: str | None, spin: str) -> str:
    header_message = _get_header()
    log_message = _get_calc_message(restricted, calc_name) + " "
    log_message += _get_irrep_message(irrep) + " "
    log_message += _get_orb_range_message(orb_range) + " "
    log_message += _get_spin_message(spin)
    return header_message + format_message(log_message, 130) + "\n"
