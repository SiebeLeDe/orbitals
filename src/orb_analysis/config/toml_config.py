import pathlib as pl
from typing import Tuple, Type

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, TomlConfigSettingsSource


class RKFReadingSettings(BaseSettings, validate_assignment=True):
    orbital_energy_unit: str = Field("eV", description="Unit of orbital energies that are extracted from the rkf file")
    orbital_energy_key: str = Field("escale", description="variable in the SFO section of a rkf file that is used. Other options are 'energy' and 'site-energies'")

    @field_validator("orbital_energy_unit")
    @classmethod
    def validate_orbital_energy_unit(cls, value: str) -> str:
        if value not in ["eV", "hartree"]:
            raise ValueError(f"Invalid unit {value}. Must be either 'eV' or 'hartree'")
        return value

    @field_validator("orbital_energy_key")
    @classmethod
    def validate_orbital_energy_key(cls, value: str) -> str:
        if value not in ["escale", "energy", "site-energies"]:
            raise ValueError(f"Invalid key {value}. Must be either 'escale', 'energy' or 'site-energies'")
        return value


class OrbAnalysisConfig(BaseSettings):
    rkf_reading: RKFReadingSettings = RKFReadingSettings()  # type: ignore # Gets instantiated in the constructor

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls, toml_file=pl.Path(__file__).resolve().parent / "config.toml"), init_settings, env_settings, dotenv_settings, file_secret_settings)


orb_config = OrbAnalysisConfig()
