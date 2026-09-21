from pathlib import Path

from PIL import Image
import numpy as np

from segmentation.background_removal import BackgroundRemover
from target.orientation import calculate_orientation_field


def load_target(
    image_path,
    segmented_output_path=Path("output/target/target_segmented.png")
):
    """
    Carica il target.

    Se esiste gia una versione segmentata, viene caricata direttamente.
    Altrimenti rimuove lo sfondo con RMBG-2.0 e salva il risultato.
    """

    segmented_output_path = Path(
        segmented_output_path
    )

    if segmented_output_path.exists():

        print(
            "Caricamento del target segmentato..."
        )

        return Image.open(
            segmented_output_path
        ).convert("RGBA")

    print(
        "Caricamento del target originale..."
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    print(
        "Rimozione dello sfondo dal target..."
    )

    remover = BackgroundRemover()

    image = remover.remove_background(
        image
    )

    segmented_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(
        segmented_output_path
    )

    print(
        f"Target segmentato salvato: "
        f"{segmented_output_path}"
    )

    return image


def calculate_grid(
    image,
    columns,
    rows
):
    """
    Divide il target RGBA in una griglia.

    Per ogni cella calcola:
    - posizione
    - dimensioni
    - percentuale di soggetto
    - colore medio del soggetto

    L'orientamento viene calcolato successivamente
    attraverso un orientation field smussato.
    """

    image = image.convert("RGBA")

    image_array = np.array(
        image
    )

    width, height = image.size

    cell_width = width / columns
    cell_height = height / rows

    cells = []

    for row in range(rows):

        for column in range(columns):

            x_start = int(
                column * cell_width
            )

            x_end = int(
                (column + 1) * cell_width
            )

            y_start = int(
                row * cell_height
            )

            y_end = int(
                (row + 1) * cell_height
            )

            cell = image_array[
                y_start:y_end,
                x_start:x_end
            ]

            rgb = cell[:, :, :3]
            alpha = cell[:, :, 3]

            subject_pixels = (
                alpha > 0
            )

            total_pixels = alpha.size

            subject_pixel_count = (
                subject_pixels.sum()
            )

            if subject_pixel_count == 0:

                mean_color = None
                subject_ratio = 0.0

            else:

                visible_pixels = rgb[
                    subject_pixels
                ]

                mean_color = tuple(
                    int(value)
                    for value in visible_pixels.mean(
                        axis=0
                    )
                )

                subject_ratio = (
                    subject_pixel_count
                    / total_pixels
                )

            cells.append({
                "row": row,
                "column": column,
                "x": x_start,
                "y": y_start,
                "width": x_end - x_start,
                "height": y_end - y_start,
                "subject_ratio": subject_ratio,
                "color": mean_color,
                "orientation": None
            })

    orientation_field = calculate_orientation_field(
        image,
        cells,
        columns,
        rows,
        window_size=45,
        contour_distance=30,
        smoothing_radius=1,
        smoothing_iterations=1
    )

    for cell in cells:

        row = cell["row"]
        column = cell["column"]

        index = (
            row * columns
            + column
        )

        cell["orientation"] = (
            orientation_field[index]
        )

    return cells