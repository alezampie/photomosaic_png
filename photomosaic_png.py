from pathlib import Path

import cv2
from PIL import Image

from video.sampler import sample_video, select_sharpest_frames
from tiles.normalize import normalize_image
from segmentation.background_removal import BackgroundRemover


RUN_PREPROCESSING = False


video_path = Path("input/video/video.mp4")

frames_output_path = Path("output/frames")
segmented_output_path = Path("output/segmented")
tiles_output_path = Path("output/tiles")


frames_output_path.mkdir(parents=True, exist_ok=True)
segmented_output_path.mkdir(parents=True, exist_ok=True)
tiles_output_path.mkdir(parents=True, exist_ok=True)


if RUN_PREPROCESSING:

    print("Campionamento del video...")

    frames = sample_video(video_path, 30)

    selected_frames = select_sharpest_frames(frames, 10)

    print(f"Frame campionati: {len(frames)}")
    print(f"Frame selezionati: {len(selected_frames)}")


    print("Salvataggio dei frame selezionati...")

    for index, frame in enumerate(selected_frames):
        file_path = frames_output_path / f"frame_{index:03d}.png"

        cv2.imwrite(str(file_path), frame)

        print(f"Salvato: {file_path}")


    print("Caricamento del background remover...")

    remover = BackgroundRemover()


    print("Rimozione degli sfondi...")

    for index, frame in enumerate(selected_frames):
        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        result = remover.remove_background(image)

        file_path = segmented_output_path / f"frame_{index:03d}.png"

        result.save(file_path)

        print(f"Salvato: {file_path}")


    print("Normalizzazione delle immagini...")

    for index in range(len(selected_frames)):
        file_path = segmented_output_path / f"frame_{index:03d}.png"

        image = Image.open(file_path)

        normalized = normalize_image(image)

        output_file_path = tiles_output_path / f"tile_{index:03d}.png"

        normalized.save(output_file_path)

        print(f"Tile salvato: {output_file_path}")

else:

    print("Preprocessing saltato.")
    print("Uso le tile gia presenti in output/tiles.")


print("Preprocessing completato.")