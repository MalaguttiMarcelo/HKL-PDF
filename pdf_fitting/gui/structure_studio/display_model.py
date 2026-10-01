from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pymatgen.core import Structure

from .chemistry import (
    clean_element_symbol,
    covalent_radius,
    display_radius,
    element_color,
)


StyleResolver = Callable[[int], Dict[str, Any]]


def build_supercell_structure(
    structure: Structure,
    repeats: Tuple[int, int, int],
) -> Structure:
    """
    Build a supercell while retaining the original/base site index.

    Every supercell site receives:

        studio_base_index

    which identifies the corresponding site in the original structure.
    """
    repeat_a = max(
        1,
        int(repeats[0]),
    )

    repeat_b = max(
        1,
        int(repeats[1]),
    )

    repeat_c = max(
        1,
        int(repeats[2]),
    )

    site_properties = {
        str(name): list(values)
        for name, values in (
            structure.site_properties
            or {}
        ).items()
    }

    site_properties["studio_base_index"] = list(
        range(
            len(structure.sites)
        )
    )

    source = Structure(
        structure.lattice,
        structure.species,
        structure.frac_coords,
        site_properties=site_properties,
        coords_are_cartesian=False,
        to_unit_cell=False,
    )

    source.make_supercell(
        [
            repeat_a,
            repeat_b,
            repeat_c,
        ]
    )

    return source


def base_site_index(
    site,
    fallback: int,
) -> int:
    """
    Read studio_base_index from a supercell site.
    """
    try:
        return int(
            site.properties.get(
                "studio_base_index",
                fallback,
            )
        )
    except Exception:
        return int(
            fallback
        )


def display_atom_key(
    display_site_index: int,
    image: Tuple[int, int, int],
) -> Tuple[int, int, int, int]:
    return (
        int(display_site_index),
        int(image[0]),
        int(image[1]),
        int(image[2]),
    )


def make_display_atom_record(
    *,
    display_structure: Structure,
    original_structure: Structure,
    display_site_index: int,
    image: Tuple[int, int, int],
    style_resolver: StyleResolver,
    boundary_image: bool,
) -> Dict[str, Any]:
    """
    Build one renderer-independent displayed atom record.
    """
    display_site = display_structure[
        int(display_site_index)
    ]

    base_index = base_site_index(
        display_site,
        int(display_site_index),
    )

    original_site = original_structure[
        base_index
    ]

    style = dict(
        style_resolver(
            base_index
        )
        or {}
    )

    symbol = clean_element_symbol(
        str(
            original_site.specie
        )
    )

    image_array = np.asarray(
        image,
        dtype=float,
    )

    fractional = (
        np.asarray(
            display_site.frac_coords,
            dtype=float,
        )
        + image_array
    )

    matrix = np.asarray(
        display_structure.lattice.matrix,
        dtype=float,
    )

    position = fractional @ matrix

    label = getattr(
        original_site,
        "label",
        None,
    )

    if not label:
        label = f"{symbol}{base_index + 1}"

    key = display_atom_key(
        display_site_index,
        image,
    )

    return {
        "key": key,
        "display_site_index": int(
            display_site_index
        ),
        "base_site_index": int(
            base_index
        ),
        "label": str(
            label
        ),
        "element": str(
            symbol
        ),
        "species": str(
            display_site.species_string
        ),
        "fractional": fractional,
        "position": position,
        "image": tuple(
            int(value)
            for value in image
        ),
        "boundary_image": bool(
            boundary_image
        ),
        "color": str(
            style.get(
                "color",
                element_color(symbol),
            )
        ),
        "radius": float(
            style.get(
                "radius",
                display_radius(symbol),
            )
        ),
        "opacity": float(
            style.get(
                "opacity",
                1.0,
            )
        ),
        "visible": bool(
            style.get(
                "visible",
                True,
            )
        ),
        "show_label": bool(
            style.get(
                "show_label",
                False,
            )
        ),
    }


def initial_display_atom_records(
    *,
    display_structure: Structure,
    original_structure: Structure,
    style_resolver: StyleResolver,
    include_boundary_images: bool,
    boundary_tolerance: float = 1.0e-7,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[Tuple[int, int, int, int], Dict[str, Any]],
]:
    """
    Build atoms inside the displayed supercell.

    When boundary images are enabled, sites on a zero fractional boundary are
    copied to the equivalent positive boundary. Edge and corner combinations
    are included.
    """
    records: List[Dict[str, Any]] = []
    lookup: Dict[
        Tuple[int, int, int, int],
        Dict[str, Any],
    ] = {}

    for display_index, site in enumerate(
        display_structure.sites
    ):
        record = make_display_atom_record(
            display_structure=display_structure,
            original_structure=original_structure,
            display_site_index=display_index,
            image=(0, 0, 0),
            style_resolver=style_resolver,
            boundary_image=False,
        )

        records.append(
            record
        )

        lookup[
            record["key"]
        ] = record

        if not include_boundary_images:
            continue

        fractional = np.asarray(
            site.frac_coords,
            dtype=float,
        )

        image_choices = []

        for component in fractional:
            if abs(float(component)) <= float(
                boundary_tolerance
            ):
                image_choices.append(
                    (
                        0,
                        1,
                    )
                )
            else:
                image_choices.append(
                    (
                        0,
                    )
                )

        for image in itertools.product(
            *image_choices
        ):
            image = tuple(
                int(value)
                for value in image
            )

            if image == (
                0,
                0,
                0,
            ):
                continue

            key = display_atom_key(
                display_index,
                image,
            )

            if key in lookup:
                continue

            boundary_record = make_display_atom_record(
                display_structure=display_structure,
                original_structure=original_structure,
                display_site_index=display_index,
                image=image,
                style_resolver=style_resolver,
                boundary_image=True,
            )

            records.append(
                boundary_record
            )

            lookup[
                key
            ] = boundary_record

    return records, lookup


def detect_display_bonds(
    *,
    display_structure: Structure,
    original_structure: Structure,
    style_resolver: StyleResolver,
    atom_records: List[Dict[str, Any]],
    atom_lookup: Dict[
        Tuple[int, int, int, int],
        Dict[str, Any],
    ],
    include_boundary_images: bool,
    tolerance_scale: float,
    minimum_distance: float,
    maximum_distance: Optional[float],
    bond_radius: float,
    bond_opacity: float,
    two_color: bool,
) -> List[Dict[str, Any]]:
    """
    Detect bonds in the displayed supercell.

    If boundary images are disabled, bonds crossing the outer supercell
    boundary are omitted.

    If boundary images are enabled, periodic endpoint atoms required by those
    bonds are added to atom_records.
    """
    symbols = [
        clean_element_symbol(
            str(site.specie)
        )
        for site in display_structure.sites
    ]

    if maximum_distance is None:
        maximum_covalent_radius = max(
            (
                covalent_radius(symbol)
                for symbol in symbols
            ),
            default=1.5,
        )

        search_radius = (
            2.0
            * maximum_covalent_radius
            * float(tolerance_scale)
            + 0.25
        )

    else:
        search_radius = float(
            maximum_distance
        )

    (
        center_indices,
        neighbor_indices,
        images,
        distances,
    ) = display_structure.get_neighbor_list(
        r=float(search_radius),
        numerical_tol=1.0e-8,
        exclude_self=True,
    )

    center_indices = np.asarray(
        center_indices,
        dtype=np.int32,
    )

    neighbor_indices = np.asarray(
        neighbor_indices,
        dtype=np.int32,
    )

    images = np.asarray(
        images,
        dtype=np.int32,
    )

    distances = np.asarray(
        distances,
        dtype=float,
    )

    matrix = np.asarray(
        display_structure.lattice.matrix,
        dtype=float,
    )

    fractional = np.asarray(
        display_structure.frac_coords,
        dtype=float,
    )

    bonds: List[Dict[str, Any]] = []
    seen = set()

    for center, neighbor, image, distance in zip(
        center_indices,
        neighbor_indices,
        images,
        distances,
    ):
        center = int(
            center
        )

        neighbor = int(
            neighbor
        )

        image_tuple = tuple(
            int(value)
            for value in image
        )

        if (
            not include_boundary_images
            and image_tuple
            != (
                0,
                0,
                0,
            )
        ):
            continue

        direct_key = (
            center,
            neighbor,
            image_tuple[0],
            image_tuple[1],
            image_tuple[2],
        )

        reverse_key = (
            neighbor,
            center,
            -image_tuple[0],
            -image_tuple[1],
            -image_tuple[2],
        )

        canonical_key = min(
            direct_key,
            reverse_key,
        )

        if canonical_key in seen:
            continue

        seen.add(
            canonical_key
        )

        center_site = display_structure[
            center
        ]

        neighbor_site = display_structure[
            neighbor
        ]

        center_base_index = base_site_index(
            center_site,
            center,
        )

        neighbor_base_index = base_site_index(
            neighbor_site,
            neighbor,
        )

        center_style = style_resolver(
            center_base_index
        )

        neighbor_style = style_resolver(
            neighbor_base_index
        )

        if not bool(
            center_style.get(
                "visible",
                True,
            )
        ):
            continue

        if not bool(
            neighbor_style.get(
                "visible",
                True,
            )
        ):
            continue

        center_element = clean_element_symbol(
            str(
                center_site.specie
            )
        )

        neighbor_element = clean_element_symbol(
            str(
                neighbor_site.specie
            )
        )

        if maximum_distance is None:
            pair_maximum = (
                float(tolerance_scale)
                * (
                    covalent_radius(
                        center_element
                    )
                    + covalent_radius(
                        neighbor_element
                    )
                )
            )
        else:
            pair_maximum = float(
                maximum_distance
            )

        if float(distance) < float(
            minimum_distance
        ):
            continue

        if float(distance) > pair_maximum:
            continue

        center_key = display_atom_key(
            center,
            (
                0,
                0,
                0,
            ),
        )

        neighbor_key = display_atom_key(
            neighbor,
            image_tuple,
        )

        center_record = atom_lookup.get(
            center_key
        )

        if center_record is None:
            center_record = make_display_atom_record(
                display_structure=display_structure,
                original_structure=original_structure,
                display_site_index=center,
                image=(0, 0, 0),
                style_resolver=style_resolver,
                boundary_image=False,
            )

            atom_records.append(
                center_record
            )

            atom_lookup[
                center_key
            ] = center_record

        neighbor_record = atom_lookup.get(
            neighbor_key
        )

        if neighbor_record is None:
            neighbor_record = make_display_atom_record(
                display_structure=display_structure,
                original_structure=original_structure,
                display_site_index=neighbor,
                image=image_tuple,
                style_resolver=style_resolver,
                boundary_image=True,
            )

            atom_records.append(
                neighbor_record
            )

            atom_lookup[
                neighbor_key
            ] = neighbor_record

        start = fractional[
            center
        ] @ matrix

        end = (
            fractional[
                neighbor
            ]
            + np.asarray(
                image_tuple,
                dtype=float,
            )
        ) @ matrix

        midpoint = 0.5 * (
            start
            + end
        )

        bonds.append(
            {
                "key": canonical_key,
                "center_key": center_key,
                "neighbor_key": neighbor_key,
                "center_display_index": center,
                "neighbor_display_index": neighbor,
                "center_base_index": center_base_index,
                "neighbor_base_index": neighbor_base_index,
                "center_element": center_element,
                "neighbor_element": neighbor_element,
                "center_label": center_record["label"],
                "neighbor_label": neighbor_record["label"],
                "start": start,
                "midpoint": midpoint,
                "end": end,
                "distance": float(
                    distance
                ),
                "image": image_tuple,
                "color_center": str(
                    center_style.get(
                        "color",
                        element_color(
                            center_element
                        ),
                    )
                ),
                "color_neighbor": str(
                    neighbor_style.get(
                        "color",
                        element_color(
                            neighbor_element
                        ),
                    )
                ),
                "radius": float(
                    bond_radius
                ),
                "opacity": float(
                    bond_opacity
                ),
                "two_color": bool(
                    two_color
                ),
            }
        )

    return bonds


def build_display_model(
    *,
    original_structure: Structure,
    repeats: Tuple[int, int, int],
    include_boundary_images: bool,
    style_resolver: StyleResolver,
    bonds_enabled: bool,
    tolerance_scale: float,
    minimum_distance: float,
    maximum_distance: Optional[float],
    bond_radius: float,
    bond_opacity: float,
    two_color: bool,
) -> Dict[str, Any]:
    """
    Build the supercell atoms and periodic bonds used by the viewer.
    """
    display_structure = build_supercell_structure(
        original_structure,
        repeats,
    )

    atom_records, atom_lookup = initial_display_atom_records(
        display_structure=display_structure,
        original_structure=original_structure,
        style_resolver=style_resolver,
        include_boundary_images=include_boundary_images,
    )

    if bonds_enabled:
        bond_records = detect_display_bonds(
            display_structure=display_structure,
            original_structure=original_structure,
            style_resolver=style_resolver,
            atom_records=atom_records,
            atom_lookup=atom_lookup,
            include_boundary_images=include_boundary_images,
            tolerance_scale=tolerance_scale,
            minimum_distance=minimum_distance,
            maximum_distance=maximum_distance,
            bond_radius=bond_radius,
            bond_opacity=bond_opacity,
            two_color=two_color,
        )
    else:
        bond_records = []

    return {
        "display_structure": display_structure,
        "atoms": atom_records,
        "atom_lookup": atom_lookup,
        "bonds": bond_records,
    }


def visible_periodic_structure(
    *,
    display_structure: Structure,
    style_resolver: StyleResolver,
) -> Structure:
    """
    Return the visible periodic supercell without redundant boundary images.

    This is appropriate for CIF export.
    """
    species = []
    fractional = []

    retained_properties: Dict[str, List[Any]] = {}

    property_names = set()

    for site in display_structure.sites:
        property_names.update(
            site.properties.keys()
        )

    property_names.discard(
        "studio_base_index"
    )

    for property_name in property_names:
        retained_properties[
            property_name
        ] = []

    for display_index, site in enumerate(
        display_structure.sites
    ):
        base_index = base_site_index(
            site,
            display_index,
        )

        style = style_resolver(
            base_index
        )

        if not bool(
            style.get(
                "visible",
                True,
            )
        ):
            continue

        species.append(
            site.species
        )

        fractional.append(
            np.asarray(
                site.frac_coords,
                dtype=float,
            )
        )

        for property_name in property_names:
            retained_properties[
                property_name
            ].append(
                site.properties.get(
                    property_name
                )
            )

    if not species:
        raise RuntimeError(
            "No visible periodic atoms are available for export."
        )

    return Structure(
        display_structure.lattice,
        species,
        fractional,
        site_properties=retained_properties,
        coords_are_cartesian=False,
        to_unit_cell=False,
    )