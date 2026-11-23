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


def site_species_components(site) -> List[Dict[str, Any]]:
    """
    Return a list of chemical components and occupancies for a pymatgen site.

    Ordered site:
        [{"element": "Zr", "occupancy": 1.0}]

    Partial site:
        [{"element": "N", "occupancy": 0.575}]

    Antisite / mixed site:
        [
            {"element": "Fe", "occupancy": 0.70},
            {"element": "Co", "occupancy": 0.30},
        ]
    """
    components: List[Dict[str, Any]] = []

    try:
        for specie, occupancy in site.species.items():
            element = clean_element_symbol(
                getattr(
                    specie,
                    "symbol",
                    str(specie),
                )
            )

            components.append(
                {
                    "element": element,
                    "occupancy": max(
                        0.0,
                        min(
                            1.0,
                            float(occupancy),
                        ),
                    ),
                }
            )

    except Exception:
        pass

    if not components:
        components.append(
            {
                "element": "X",
                "occupancy": 1.0,
            }
        )

    return components


def primary_site_element(site) -> str:
    """
    Return the element having the largest occupancy on this site.

    This is used for atom labels and bond-radius calculations.
    """
    components = site_species_components(site)

    component = max(
        components,
        key=lambda item: float(
            item.get(
                "occupancy",
                0.0,
            )
        ),
    )

    return str(
        component.get(
            "element",
            "X",
        )
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

    source_species = [
        site.species
        for site in structure.sites
    ]

    source = Structure(
        structure.lattice,
        source_species,
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

    symbol = primary_site_element(
        original_site
    )

    occupancy_components = site_species_components(
        original_site
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
        "occupancy_components": occupancy_components,
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
        primary_site_element(site)
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

        center_element = primary_site_element(
            center_site
        )

        neighbor_element = primary_site_element(
            neighbor_site
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


def bonds_between_displayed_atoms(
    *,
    atom_records: List[Dict[str, Any]],
    style_resolver: StyleResolver,
    tolerance_scale: float,
    minimum_distance: float,
    maximum_distance: Optional[float],
    bond_radius: float,
    bond_opacity: float,
    two_color: bool,
) -> List[Dict[str, Any]]:
    """
    Bonds only between atoms that are already displayed (unit cell(s) plus
    boundary images). No atoms are added, so the display stays inside the
    cell box (VESTA-like). Uses a k-d tree on Cartesian positions.
    """
    from scipy.spatial import cKDTree

    records = [r for r in atom_records if bool(r.get("visible", True))]

    if len(records) < 2:
        return []

    positions = np.asarray([r["position"] for r in records], dtype=float)
    elements = [str(r.get("element") or r.get("symbol") or "") for r in records]

    if maximum_distance is None:
        search_radius = (
            2.0 * max((covalent_radius(e) for e in elements), default=1.5)
            * float(tolerance_scale)
            + 0.25
        )
    else:
        search_radius = float(maximum_distance)

    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=float(search_radius), output_type="ndarray")

    bonds: List[Dict[str, Any]] = []

    for i, j in pairs:
        a, b = records[int(i)], records[int(j)]
        start = positions[int(i)]
        end = positions[int(j)]
        distance = float(np.linalg.norm(end - start))

        if distance < float(minimum_distance):
            continue

        ea, eb = elements[int(i)], elements[int(j)]

        if maximum_distance is None:
            pair_maximum = float(tolerance_scale) * (covalent_radius(ea) + covalent_radius(eb))
        else:
            pair_maximum = float(maximum_distance)

        if distance > pair_maximum:
            continue

        style_a = style_resolver(int(a["base_site_index"]))
        style_b = style_resolver(int(b["base_site_index"]))
        key = tuple(sorted((tuple(a["key"]), tuple(b["key"]))))
        image = tuple(int(x) for x in np.asarray(b["key"][1:]) - np.asarray(a["key"][1:]))

        bonds.append(
            {
                "key": key,
                "center_key": a["key"],
                "neighbor_key": b["key"],
                "center_display_index": int(a["display_site_index"]),
                "neighbor_display_index": int(b["display_site_index"]),
                "center_base_index": int(a["base_site_index"]),
                "neighbor_base_index": int(b["base_site_index"]),
                "center_element": ea,
                "neighbor_element": eb,
                "center_label": a.get("label", ""),
                "neighbor_label": b.get("label", ""),
                "start": start,
                "midpoint": 0.5 * (start + end),
                "end": end,
                "distance": distance,
                "image": image,
                "color_center": str(style_a.get("color", element_color(ea))),
                "color_neighbor": str(style_b.get("color", element_color(eb))),
                "radius": float(bond_radius),
                "opacity": float(bond_opacity),
                "two_color": bool(two_color),
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
    add_bonded_outside: bool = False,
) -> Dict[str, Any]:
    """
    Build the supercell atoms and periodic bonds used by the viewer.

    add_bonded_outside: add the bonded neighbours that lie outside the
    displayed cell(s) (molecule completion). Off by default: bonds are drawn
    only between displayed atoms, so the view stays inside the cell box.
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

    if bonds_enabled and not add_bonded_outside:
        bond_records = bonds_between_displayed_atoms(
            atom_records=atom_records,
            style_resolver=style_resolver,
            tolerance_scale=tolerance_scale,
            minimum_distance=minimum_distance,
            maximum_distance=maximum_distance,
            bond_radius=bond_radius,
            bond_opacity=bond_opacity,
            two_color=two_color,
        )
    elif bonds_enabled:
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