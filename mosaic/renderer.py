from pathlib import Path
import random

from PIL import Image, ImageChops

from .matcher import recolor_tile
from target.orientation import calculate_orientation


def load_tiles(tiles_path):
    """
    Carica tutte le tile presenti nella cartella.
    """

    tiles = []

    tile_files = sorted(
        Path(tiles_path).glob("tile_*.png")
    )

    for tile_file in tile_files:

        tile = Image.open(
            tile_file
        ).convert("RGBA")

        tiles.append(tile)

    if not tiles:
        raise ValueError(
            f"Nessuna tile trovata in: {tiles_path}"
        )

    return tiles


def prepare_tiles(tiles):
    """
    Calcola l'orientamento originale di ogni tile.
    """

    prepared_tiles = []

    for tile in tiles:

        orientation = calculate_orientation(
            tile
        )

        prepared_tiles.append({
            "image": tile,
            "orientation": orientation
        })

    return prepared_tiles


def render_mosaic(
    target,
    cells,
    tiles,
    tile_scale=3,
    scale_randomness=0.15,
    position_randomness=0.15,
    rotation_randomness=8
):
    """
    Genera la photomosaic.

    Per ogni cella con soggetto:
    - sceglie una tile casuale;
    - ricolora la tile verso il colore della cella;
    - calcola la rotazione verso l'orientamento del target;
    - aggiunge una variazione casuale alla rotazione;
    - applica una variazione casuale alla scala;
    - applica una variazione casuale alla posizione;
    - ridimensiona la tile;
    - la sovrappone alle altre tile.

    Alla fine il collage viene ritagliato usando
    l'alpha dell'immagine target scontornata.
    """

    target = target.convert("RGBA")

    output = Image.new(
        "RGBA",
        target.size,
        (0, 0, 0, 0)
    )

    prepared_tiles = prepare_tiles(
        tiles
    )

    for cell in cells:

        if cell["subject_ratio"] == 0:
            continue

        tile_data = random.choice(
            prepared_tiles
        )

        tile = tile_data["image"]
        tile_orientation = tile_data["orientation"]

        tile = recolor_tile(
            tile,
            cell["color"]
        )

        # Orientamento

        target_orientation = cell["orientation"]

        if (
            target_orientation is not None
            and tile_orientation is not None
        ):

            rotation = (
                target_orientation
                - tile_orientation
            )

            rotation += random.uniform(
                -rotation_randomness,
                rotation_randomness
            )

            tile = tile.rotate(
                rotation,
                resample=Image.Resampling.BICUBIC,
                expand=True
            )

        # Scala casuale

        random_scale = random.uniform(
            1 - scale_randomness,
            1 + scale_randomness
        )

        current_scale = (
            tile_scale
            * random_scale
        )

        tile_width = int(
            cell["width"]
            * current_scale
        )

        tile_height = int(
            cell["height"]
            * current_scale
        )

        tile = tile.resize(
            (tile_width, tile_height),
            Image.Resampling.LANCZOS
        )

        # Posizione casuale

        max_offset_x = (
            cell["width"]
            * position_randomness
        )

        max_offset_y = (
            cell["height"]
            * position_randomness
        )

        offset_x = random.uniform(
            -max_offset_x,
            max_offset_x
        )

        offset_y = random.uniform(
            -max_offset_y,
            max_offset_y
        )

        x = (
            cell["x"]
            + cell["width"] / 2
            - tile_width / 2
            + offset_x
        )

        y = (
            cell["y"]
            + cell["height"] / 2
            - tile_height / 2
            + offset_y
        )

        output.alpha_composite(
            tile,
            (int(x), int(y))
        )

    # Ritaglia il collage usando l'alpha del target

    target_alpha = target.getchannel("A")
    mosaic_alpha = output.getchannel("A")

    clipped_alpha = ImageChops.multiply(
        mosaic_alpha,
        target_alpha
    )

    output.putalpha(
        clipped_alpha
    )

    return output