import cv2

from .sharpness import calculate_sharpness


def sample_video(video_path, number_of_frames):
    """
    Estrae un numero prestabilito di frame distribuiti lungo il video.

    Restituisce una lista di immagini OpenCV, una per ogni punto
    campionato nel video.
    """

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError(f"Impossibile aprire il video: {video_path}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        video.release()
        raise ValueError("Il video non contiene frame.")

    number_of_frames = min(number_of_frames, total_frames)

    if number_of_frames == 1:
        frame_indices = [0]
    else:
        frame_indices = [
            int(i * (total_frames - 1) / (number_of_frames - 1))
            for i in range(number_of_frames)
        ]

    frames = []

    for frame_index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = video.read()

        if not success:
            continue

        frames.append(frame)

    video.release()

    return frames


def select_sharpest_frames(frames, number_of_frames):
    """
    Divide i frame in gruppi e prende il frame piu nitido
    da ogni gruppo.

    In questo modo i frame selezionati rimangono distribuiti
    lungo tutto il video.
    """

    if number_of_frames >= len(frames):
        return frames

    group_size = len(frames) / number_of_frames

    selected_frames = []

    for i in range(number_of_frames):
        start = int(i * group_size)
        end = int((i + 1) * group_size)

        group = frames[start:end]

        sharpest_frame = max(
            group,
            key=calculate_sharpness
        )

        selected_frames.append(sharpest_frame)

    return selected_frames