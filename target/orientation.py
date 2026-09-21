import numpy as np
from PIL import Image
import cv2


def normalize_angle(angle):
    """
    Normalizza un angolo tra -90 e +90 gradi.
    """

    while angle > 90:
        angle -= 180

    while angle < -90:
        angle += 180

    return angle


def calculate_orientation(
    image,
    min_pixels=50,
    reference_point=None
):
    """
    Calcola l'orientamento principale del soggetto
    usando una PCA pesata sul canale alpha.
    """

    image = image.convert("RGBA")

    array = np.array(image)

    alpha = array[:, :, 3]

    points = np.column_stack(
        np.where(alpha > 0)
    )

    if len(points) < min_pixels:
        return None

    if reference_point is None:

        center = points.mean(axis=0)

        weights = np.ones(
            len(points)
        )

    else:

        reference_y, reference_x = reference_point

        distances = np.sqrt(
            (points[:, 0] - reference_y) ** 2
            + (points[:, 1] - reference_x) ** 2
        )

        sigma = max(
            image.size
        ) / 3

        weights = np.exp(
            -(distances ** 2)
            / (2 * sigma ** 2)
        )

        if weights.sum() == 0:
            return None

        center = np.average(
            points,
            axis=0,
            weights=weights
        )

    centered = points - center

    covariance = np.cov(
        centered,
        rowvar=False,
        aweights=weights
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    principal_axis = eigenvectors[
        :, np.argmax(eigenvalues)
    ]

    y, x = principal_axis

    angle = np.degrees(
        np.arctan2(
            y,
            x
        )
    )

    return normalize_angle(angle)


def calculate_contour_points(image):
    """
    Estrae i punti del contorno esterno del soggetto.

    Per ogni punto restituisce:
    - posizione x
    - posizione y
    - orientamento della tangente
    """

    image = image.convert("RGBA")

    array = np.array(image)

    alpha = array[:, :, 3]

    mask = np.where(
        alpha > 0,
        255,
        0
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    points = []

    for contour in contours:

        contour = contour[:, 0, :]

        contour_length = len(
            contour
        )

        if contour_length < 10:
            continue

        offset = max(
            5,
            min(
                15,
                contour_length // 10
            )
        )

        for index, point in enumerate(contour):

            point_before = contour[
                (index - offset)
                % contour_length
            ]

            point_after = contour[
                (index + offset)
                % contour_length
            ]

            dx = (
                point_after[0]
                - point_before[0]
            )

            dy = (
                point_after[1]
                - point_before[1]
            )

            if dx == 0 and dy == 0:
                continue

            angle = np.degrees(
                np.arctan2(
                    dy,
                    dx
                )
            )

            angle = normalize_angle(
                angle
            )

            points.append({
                "x": float(point[0]),
                "y": float(point[1]),
                "angle": float(angle)
            })

    return points


def calculate_local_orientation(
    contour_points,
    center_x,
    center_y,
    max_distance=150
):
    """
    Calcola l'orientamento locale usando i punti
    del contorno piu vicini alla cella.

    Gli angoli vengono mediati come assi,
    quindi 0 e 180 gradi sono equivalenti.
    """

    if not contour_points:
        return None

    candidates = []

    for point in contour_points:

        dx = point["x"] - center_x
        dy = point["y"] - center_y

        distance = np.sqrt(
            dx * dx
            + dy * dy
        )

        if distance <= max_distance:
            candidates.append(
                (
                    distance,
                    point["angle"]
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda value: value[0]
    )

    candidates = candidates[:20]

    angles = np.radians([
        angle
        for _, angle in candidates
    ])

    weights = np.array([
        1 / (distance + 1)
        for distance, _ in candidates
    ])

    sin_sum = np.sum(
        weights * np.sin(2 * angles)
    )

    cos_sum = np.sum(
        weights * np.cos(2 * angles)
    )

    if (
        abs(sin_sum) < 1e-8
        and abs(cos_sum) < 1e-8
    ):
        return None

    angle = 0.5 * np.arctan2(
        sin_sum,
        cos_sum
    )

    return normalize_angle(
        np.degrees(angle)
    )


def smooth_orientation_field(
    orientation_field,
    columns,
    rows,
    radius=1,
    iterations=1
):
    """
    Smussa leggermente il campo di orientamento.

    Gli orientamenti sono assi, quindi vengono
    mediati usando il doppio dell'angolo.
    """

    if radius <= 0 or iterations <= 0:
        return orientation_field

    field = np.array(
        orientation_field,
        dtype=object
    ).reshape(
        rows,
        columns
    )

    for _ in range(iterations):

        new_field = field.copy()

        for row in range(rows):

            for column in range(columns):

                if field[row, column] is None:
                    continue

                angles = []

                for dy in range(
                    -radius,
                    radius + 1
                ):

                    for dx in range(
                        -radius,
                        radius + 1
                    ):

                        y = row + dy
                        x = column + dx

                        if (
                            y < 0
                            or y >= rows
                            or x < 0
                            or x >= columns
                        ):
                            continue

                        angle = field[
                            y,
                            x
                        ]

                        if angle is None:
                            continue

                        angles.append(
                            angle
                        )

                if not angles:
                    continue

                angles = np.radians(
                    angles
                )

                sin_sum = np.mean(
                    np.sin(2 * angles)
                )

                cos_sum = np.mean(
                    np.cos(2 * angles)
                )

                new_angle = (
                    0.5
                    * np.arctan2(
                        sin_sum,
                        cos_sum
                    )
                )

                new_field[
                    row,
                    column
                ] = normalize_angle(
                    np.degrees(
                        new_angle
                    )
                )

        field = new_field

    return field.flatten().tolist()


def calculate_orientation_field(
    image,
    cells,
    columns,
    rows,
    window_size=45,
    contour_distance=30,
    smoothing_radius=1,
    smoothing_iterations=1
):
    """
    Calcola il campo di orientamento locale
    dell'intero soggetto.

    L'orientamento viene ricavato dal contorno
    del soggetto e non da una PCA indipendente
    per ogni cella.
    """

    contour_points = calculate_contour_points(
        image
    )

    orientation_field = []

    for cell in cells:

        if cell["subject_ratio"] == 0:

            orientation_field.append(
                None
            )

            continue

        center_x = (
            cell["x"]
            + cell["width"] / 2
        )

        center_y = (
            cell["y"]
            + cell["height"] / 2
        )

        orientation = calculate_local_orientation(
            contour_points,
            center_x,
            center_y,
            max_distance=150
        )

        orientation_field.append(
            orientation
        )

    orientation_field = smooth_orientation_field(
        orientation_field,
        columns,
        rows,
        radius=smoothing_radius,
        iterations=smoothing_iterations
    )

    return orientation_field