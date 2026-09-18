from pathlib import Path

import cv2
from PIL import Image

from video.sampler import sample_video, select_sharpest_frames
from tiles.normalize import normalize_image
from segmentation.background_removal import BackgroundRemover

from target.grid import load_target, calculate_grid

from mosaic.renderer import load_tiles, render_mosaic
from target.orientation import calculate_orientation


RUN_PREPROCESSING = True


video_path = Path("input/video/video.mp4")

frames_output_path = Path("output/frames")
segmented_output_path = Path("output/segmented")
tiles_output_path = Path("output/tiles")
mosaic_output_path = Path("output/mosaic")

target_path = Path(
    "input/target/target.jpg"
)


frames_output_path.mkdir(
    parents=True,
    exist_ok=True
)

segmented_output_path.mkdir(
    parents=True,
    exist_ok=True
)

tiles_output_path.mkdir(
    parents=True,
    exist_ok=True
)

mosaic_output_path.mkdir(
    parents=True,
    exist_ok=True
)


if RUN_PREPROCESSING:

    print("Pulizia dei file precedenti...")

    for folder in [
        frames_output_path,
        segmented_output_path,
        tiles_output_path
    ]:

        for file in folder.glob("*.png"):
            file.unlink()

    print("Pulizia completata.")

    print()
    print("Campionamento del video...")

    frames = sample_video(
        video_path,
        90
    )

    selected_frames = select_sharpest_frames(
        frames,
        30
    )

    print(
        f"Frame campionati: {len(frames)}"
    )

    print(
        f"Frame selezionati: {len(selected_frames)}"
    )


    print()
    print("Salvataggio dei frame selezionati...")

    for index, frame in enumerate(selected_frames):

        file_path = (
            frames_output_path
            / f"frame_{index:03d}.png"
        )

        cv2.imwrite(
            str(file_path),
            frame
        )

        print(
            f"Salvato: {file_path}"
        )


    print()
    print("Caricamento del background remover...")

    remover = BackgroundRemover()


    print()
    print("Rimozione degli sfondi...")

    for index, frame in enumerate(selected_frames):

        image = Image.fromarray(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )

        result = remover.remove_background(
            image
        )

        file_path = (
            segmented_output_path
            / f"frame_{index:03d}.png"
        )

        result.save(file_path)

        print(
            f"Salvato: {file_path}"
        )


    print()
    print("Normalizzazione delle immagini...")

    for index in range(len(selected_frames)):

        file_path = (
            segmented_output_path
            / f"frame_{index:03d}.png"
        )

        image = Image.open(file_path)

        normalized = normalize_image(
            image
        )

        output_file_path = (
            tiles_output_path
            / f"tile_{index:03d}.png"
        )

        normalized.save(
            output_file_path
        )

        print(
            f"Tile salvato: {output_file_path}"
        )

else:

    print("Preprocessing saltato.")
    print("Uso le tile gia presenti in output/tiles.")


print()
print("Preprocessing completato.")


# Target grid

print()
print("Caricamento del target...")

target = load_target(
    target_path
)

print(
    f"Target caricato: {target.size}"
)


print()
print("Calcolo della griglia...")

columns = 40
rows = 40

cells = calculate_grid(
    target,
    columns,
    rows
)

print(
    f"Celle totali: {len(cells)}"
)


subject_cells = [
    cell
    for cell in cells
    if cell["subject_ratio"] > 0
]

print(
    f"Celle con soggetto: {len(subject_cells)}"
)

print(
    f"Celle trasparenti: "
    f"{len(cells) - len(subject_cells)}"
)


print()
print("Prime celle con soggetto:")

for cell in subject_cells[:10]:

    print(
        f"Riga {cell['row']}, "
        f"colonna {cell['column']} -> "
        f"soggetto {cell['subject_ratio']:.1%}, "
        f"colore {cell['color']}, "
        f"orientamento "
        f"{cell['orientation']:.1f} gradi"
    )


# Matching

print()
print("Caricamento delle tile...")

tiles = load_tiles(
    tiles_output_path
)


print()
print("Orientamento delle tile:")

for index, tile in enumerate(tiles):

    angle = calculate_orientation(
        tile
    )

    print(
        f"Tile {index:03d}: "
        f"{angle:.1f} gradi"
    )


print(
    f"Tile caricate: {len(tiles)}"
)


print()
print("Generazione della photomosaic...")

mosaic = render_mosaic(
    target,
    cells,
    tiles
)


# Composizione finale

print()
print("Sovrapposizione della mosaic al target originale...")

original_target = Image.open(
    target_path
).convert("RGBA")

final_image = original_target.copy()

final_image.alpha_composite(
    mosaic
)


final_output_path = (
    mosaic_output_path
    / "mosaic_final.png"
)

final_image.save(
    final_output_path
)

print(
    f"Immagine finale salvata: {final_output_path}"
)