from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from typing import Any, Dict, List, Optional

from pdf_fitting.user_paths import user_cache_dir


STRUCTURE_METADATA_CACHE_VERSION = "structure_metadata_v2"

STRUCTURE_METADATA_CACHE_DIR = user_cache_dir(
    "structure_metadata_cache"
)


def clean_element_symbol(value: Any) -> str:
    text = str(
        value
        or ""
    ).strip()

    match = re.match(
        r"([A-Za-z]{1,2})",
        text,
    )

    if match is None:
        return text

    symbol = match.group(1)

    if len(symbol) == 1:
        return symbol.upper()

    return (
        symbol[0].upper()
        + symbol[1:].lower()
    )


def lambda_safe_species_label(value: Any) -> str:
    text = str(
        value
        or ""
    ).strip()

    text = text.replace(
        "+",
        "plus",
    )

    text = text.replace(
        "-",
        "minus",
    )

    text = re.sub(
        r"[^A-Za-z0-9]+",
        "",
        text,
    )

    return text.lower()


def parse_cif_number(
    value: Any,
) -> Optional[float]:
    if value is None:
        return None

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        try:
            return float(
                value
            )
        except Exception:
            return None

    text = str(
        value
    ).strip()

    if (
        not text
        or text in (
            ".",
            "?",
        )
    ):
        return None

    if "(" in text:
        text = text.split(
            "(",
            1,
        )[0].strip()

    try:
        return float(
            text
        )
    except Exception:
        return None


def tokenize_cif_line(
    line: str,
) -> List[str]:
    text = str(
        line
    ).strip()

    if not text:
        return []

    try:
        return shlex.split(
            text,
            posix=True,
        )
    except Exception:
        return text.split()


def read_text(
    path: str,
) -> str:
    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        return file.read()


def scalar_token(
    text: str,
    tag_names,
) -> Optional[str]:
    if isinstance(
        tag_names,
        str,
    ):
        tag_names = [
            tag_names
        ]

    for tag_name in tag_names:
        pattern = re.compile(
            rf"(?im)^\s*{re.escape(tag_name)}\s+(.+?)\s*$"
        )

        match = pattern.search(
            text
        )

        if match is None:
            continue

        tokens = tokenize_cif_line(
            match.group(1)
        )

        if tokens:
            return str(
                tokens[0]
            ).strip()

    return None


def scalar_float(
    text: str,
    tag_names,
) -> Optional[float]:
    token = scalar_token(
        text,
        tag_names,
    )

    return parse_cif_number(
        token
    )


def scalar_int(
    text: str,
    tag_names,
) -> Optional[int]:
    value = scalar_float(
        text,
        tag_names,
    )

    if value is None:
        return None

    try:
        return int(
            round(
                float(value)
            )
        )
    except Exception:
        return None


def crystal_system_from_space_group(
    space_group_number: Optional[int],
) -> str:
    if space_group_number is None:
        return ""

    number = int(
        space_group_number
    )

    if 1 <= number <= 2:
        return "triclinic"

    if 3 <= number <= 15:
        return "monoclinic"

    if 16 <= number <= 74:
        return "orthorhombic"

    if 75 <= number <= 142:
        return "tetragonal"

    if 143 <= number <= 167:
        return "trigonal"

    if 168 <= number <= 194:
        return "hexagonal"

    if 195 <= number <= 230:
        return "cubic"

    return ""


def required_contrast_terms(
    space_group_number: Optional[int],
) -> List[str]:
    if space_group_number is None:
        return []

    number = int(
        space_group_number
    )

    if 1 <= number <= 2:
        return [
            f"E{index}"
            for index in range(
                1,
                16,
            )
        ]

    if 3 <= number <= 15:
        return [
            f"E{index}"
            for index in range(
                1,
                10,
            )
        ]

    if 16 <= number <= 74:
        return [
            f"E{index}"
            for index in range(
                1,
                7,
            )
        ]

    if 75 <= number <= 88:
        return [
            f"E{index}"
            for index in range(
                1,
                6,
            )
        ]

    if 89 <= number <= 142:
        return [
            f"E{index}"
            for index in range(
                1,
                5,
            )
        ]

    if 143 <= number <= 148:
        return [
            "E1",
            "E2",
            "E3",
            "E4",
            "E5",
        ]

    if 149 <= number <= 167:
        return [
            "E1",
            "E2",
            "E3",
            "E4",
        ]

    if 168 <= number <= 194:
        return [
            "E1",
            "E2",
            "E3",
        ]

    if 195 <= number <= 230:
        return [
            "E1",
            "E2",
        ]

    return []


def parse_atom_site_loop(
    text: str,
) -> List[Dict[str, Any]]:
    lines = text.splitlines()

    for line_index, line in enumerate(
        lines
    ):
        if not line.strip().lower().startswith(
            "loop_"
        ):
            continue

        current_index = (
            line_index
            + 1
        )

        headers = []

        while (
            current_index < len(lines)
            and lines[current_index].strip().startswith(
                "_"
            )
        ):
            headers.append(
                lines[current_index].strip()
            )

            current_index += 1

        headers_lower = [
            header.lower()
            for header in headers
        ]

        required_headers = {
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        }

        if not required_headers.issubset(
            set(headers_lower)
        ):
            continue

        def header_index(
            name: str,
        ) -> Optional[int]:
            try:
                return headers_lower.index(
                    name.lower()
                )
            except Exception:
                return None

        label_index = header_index(
            "_atom_site_label"
        )

        element_index = header_index(
            "_atom_site_type_symbol"
        )

        x_index = header_index(
            "_atom_site_fract_x"
        )

        y_index = header_index(
            "_atom_site_fract_y"
        )

        z_index = header_index(
            "_atom_site_fract_z"
        )

        occupancy_index = header_index(
            "_atom_site_occupancy"
        )

        biso_index = header_index(
            "_atom_site_b_iso_or_equiv"
        )

        uiso_index = header_index(
            "_atom_site_u_iso_or_equiv"
        )

        records = []

        pending_tokens = []

        while current_index < len(
            lines
        ):
            raw_line = lines[
                current_index
            ]

            stripped = raw_line.strip()
            lowered = stripped.lower()

            if not stripped:
                current_index += 1
                continue

            if (
                lowered.startswith(
                    "loop_"
                )
                or lowered.startswith(
                    "data_"
                )
                or lowered.startswith(
                    "save_"
                )
                or stripped.startswith(
                    "_"
                )
            ):
                break

            pending_tokens.extend(
                tokenize_cif_line(
                    stripped
                )
            )

            while len(
                pending_tokens
            ) >= len(
                headers
            ):
                tokens = pending_tokens[
                    :len(headers)
                ]

                pending_tokens = pending_tokens[
                    len(headers):
                ]

                if label_index is not None:
                    label = str(
                        tokens[
                            label_index
                        ]
                    ).strip()
                else:
                    label = (
                        f"site{len(records) + 1}"
                    )

                if not label:
                    label = (
                        f"site{len(records) + 1}"
                    )

                if element_index is not None:
                    raw_species = str(
                        tokens[
                            element_index
                        ]
                    ).strip()
                else:
                    raw_species = label

                element = clean_element_symbol(
                    raw_species
                )

                x_value = parse_cif_number(
                    tokens[
                        x_index
                    ]
                )

                y_value = parse_cif_number(
                    tokens[
                        y_index
                    ]
                )

                z_value = parse_cif_number(
                    tokens[
                        z_index
                    ]
                )

                if (
                    x_value is None
                    or y_value is None
                    or z_value is None
                ):
                    continue

                occupancy = 1.0

                if occupancy_index is not None:
                    parsed_occupancy = parse_cif_number(
                        tokens[
                            occupancy_index
                        ]
                    )

                    if parsed_occupancy is not None:
                        occupancy = float(
                            parsed_occupancy
                        )

                biso = None

                if biso_index is not None:
                    biso = parse_cif_number(
                        tokens[
                            biso_index
                        ]
                    )

                elif uiso_index is not None:
                    uiso = parse_cif_number(
                        tokens[
                            uiso_index
                        ]
                    )

                    if uiso is not None:
                        biso = (
                            8.0
                            * 3.141592653589793 ** 2
                            * float(uiso)
                        )

                records.append(
                    {
                        "label": label,
                        "el": element,
                        "species_label": raw_species,
                        "frac": [
                            float(x_value),
                            float(y_value),
                            float(z_value),
                        ],
                        "occ": float(
                            occupancy
                        ),
                        "biso": (
                            None
                            if biso is None
                            else float(biso)
                        ),
                    }
                )

            current_index += 1

        if records:
            return records

    return []


def metadata_cache_path(
    path: str,
) -> str:
    absolute_path = os.path.abspath(
        path
    )

    try:
        stat_result = os.stat(
            absolute_path
        )

        modification_time_ns = int(
            stat_result.st_mtime_ns
        )

        file_size = int(
            stat_result.st_size
        )

    except Exception:
        modification_time_ns = 0
        file_size = 0

    raw_key = (
        f"{STRUCTURE_METADATA_CACHE_VERSION}|"
        f"{absolute_path}|"
        f"{modification_time_ns}|"
        f"{file_size}"
    )

    cache_key = hashlib.md5(
        raw_key.encode(
            "utf-8"
        )
    ).hexdigest()

    return os.path.join(
        STRUCTURE_METADATA_CACHE_DIR,
        f"structure_{cache_key}.json",
    )


def read_structure_metadata(
    path: str,
) -> Dict[str, Any]:
    path = str(
        path
        or ""
    ).strip()

    if not path:
        raise ValueError(
            "Structure path is empty."
        )

    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            path
        )

    if not path.lower().endswith(
        ".cif"
    ):
        return {
            "cache_version": (
                STRUCTURE_METADATA_CACHE_VERSION
            ),
            "source_path": os.path.abspath(
                path
            ),
            "supported": False,
            "lattice": {},
            "space_group_number": None,
            "crystal_system": "",
            "contrast_terms": [],
            "elements": [],
            "pair_labels": [],
            "site_records": [],
        }

    cache_path = metadata_cache_path(
        path
    )

    if os.path.exists(
        cache_path
    ):
        try:
            with open(
                cache_path,
                "r",
                encoding="utf-8",
            ) as file:
                cached = json.load(
                    file
                )

            if (
                isinstance(
                    cached,
                    dict,
                )
                and cached.get(
                    "cache_version"
                )
                == STRUCTURE_METADATA_CACHE_VERSION
            ):
                cached["cache_hit"] = True
                return cached

        except Exception:
            pass

    text = read_text(
        path
    )

    lattice = {
        "a": scalar_float(
            text,
            "_cell_length_a",
        ),
        "b": scalar_float(
            text,
            "_cell_length_b",
        ),
        "c": scalar_float(
            text,
            "_cell_length_c",
        ),
        "alpha": scalar_float(
            text,
            "_cell_angle_alpha",
        ),
        "beta": scalar_float(
            text,
            "_cell_angle_beta",
        ),
        "gamma": scalar_float(
            text,
            "_cell_angle_gamma",
        ),
    }

    lattice = {
        key: value
        for key, value in lattice.items()
        if value is not None
    }

    space_group_number = scalar_int(
        text,
        [
            "_space_group_it_number",
            "_symmetry_int_tables_number",
            "_space_group_it_number",
        ],
    )

    crystal_system = crystal_system_from_space_group(
        space_group_number
    )

    site_records = parse_atom_site_loop(
        text
    )

    elements = sorted(
        {
            clean_element_symbol(
                record.get(
                    "el",
                    "",
                )
            )
            for record in site_records
            if clean_element_symbol(
                record.get(
                    "el",
                    "",
                )
            )
        },
        key=lambda value: value.lower(),
    )

    lambda_species = sorted(
        {
            lambda_safe_species_label(
                record.get(
                    "species_label",
                    record.get(
                        "el",
                        "",
                    ),
                )
            )
            for record in site_records
            if lambda_safe_species_label(
                record.get(
                    "species_label",
                    record.get(
                        "el",
                        "",
                    ),
                )
            )
        },
        key=lambda value: value.lower(),
    )

    pair_labels = []

    for first_index, first_species in enumerate(
        lambda_species
    ):
        for second_species in lambda_species[
            first_index:
        ]:
            pair_labels.append(
                f"{first_species}-{second_species}"
            )

    metadata = {
        "cache_version": (
            STRUCTURE_METADATA_CACHE_VERSION
        ),
        "source_path": os.path.abspath(
            path
        ),
        "supported": True,
        "lattice": lattice,
        "space_group_number": (
            None
            if space_group_number is None
            else int(
                space_group_number
            )
        ),
        "crystal_system": crystal_system,
        "contrast_terms": required_contrast_terms(
            space_group_number
        ),
        "elements": elements,
        "pair_labels": pair_labels,
        "site_records": site_records,
        "cache_hit": False,
    }

    temporary_path = (
        cache_path
        + ".tmp"
    )

    try:
        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )

        os.replace(
            temporary_path,
            cache_path,
        )

    except Exception:
        try:
            if os.path.exists(
                temporary_path
            ):
                os.remove(
                    temporary_path
                )
        except Exception:
            pass

    return metadata