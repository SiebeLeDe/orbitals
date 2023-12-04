""" Module containing messages for logging and providing feedback to the user. """


def _get_calc_message(restricted: bool, calc_name: str) -> str:
    if restricted:
        return f"This is a restricted calculation for {calc_name}."
    return f"This is a unrestricted calculation for {calc_name}."


def _get_irrep_message(irrep: str | None):
    irrep_message = "The selected orbitals belong to "
    if irrep is not None:
        irrep_message += "all irreps present."
    else:
        irrep_message += f"the {irrep} irrep."
    return irrep_message


def _get_orb_range_message(orb_range: tuple[int, int]) -> str:
    return f"Orbitals are selected from HOMO-{orb_range[0]} to LUMO+{orb_range[1]}."


def calc_analyzer_call_message(restricted: bool, calc_name: str, orb_range: tuple[int, int], irrep: str | None) -> str:
    log_message = _get_calc_message(restricted, calc_name) + " "
    log_message += _get_irrep_message(irrep) + " "
    log_message += _get_orb_range_message(orb_range) + " "
    return log_message
