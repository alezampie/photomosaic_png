from PIL import Image
import numpy as np
import colorsys


def get_tile_color(image):
    """
    Calcola il colore medio dei pixel visibili della tile.
    """

    image = image.convert("RGBA")

    array = np.array(image)

    rgb = array[:, :, :3]
    alpha = array[:, :, 3]

    visible_pixels = rgb[alpha > 0]

    if len(visible_pixels) == 0:
        return (0, 0, 0)

    mean_color = visible_pixels.mean(axis=0)

    return tuple(
        int(value)
        for value in mean_color
    )


def recolor_tile(image, target_color):
    """
    Ricolora la tile verso il colore target.

    Mantiene le variazioni di luminosita e i dettagli
    della tile originale.
    """

    image = image.convert("RGBA")

    array = np.array(image).astype(np.float32)

    rgb = array[:, :, :3] / 255.0
    alpha = array[:, :, 3]

    target_rgb = (
        np.array(
            target_color,
            dtype=np.float32
        ) / 255.0
    )

    target_h, target_s, target_v = colorsys.rgb_to_hsv(
        *target_rgb
    )

    # Calcolo della luminosita originale della tile.

    value = np.max(
        rgb,
        axis=2
    )

    # Calcolo della saturazione originale.

    min_rgb = np.min(
        rgb,
        axis=2
    )

    delta = value - min_rgb

    saturation = np.zeros_like(
        value
    )

    non_zero_value = value > 0

    saturation[non_zero_value] = (
        delta[non_zero_value]
        / value[non_zero_value]
    )

    # Calcolo della luminosita media
    # dei pixel visibili.

    visible_value = value[
        alpha > 0
    ]

    if len(visible_value) == 0:
        return image

    mean_value = visible_value.mean()

    # Evita divisioni per zero.

    if mean_value > 0:

        brightness_scale = (
            target_v
            / mean_value
        )

    else:

        brightness_scale = 0.0

    # Applica la nuova luminosita mantenendo
    # le variazioni relative della tile.

    new_value = (
        value
        * brightness_scale
    )

    new_value = np.clip(
        new_value,
        0,
        1
    )

    # Manteniamo una parte della saturazione originale
    # e una parte di quella del target.

    new_saturation = (
        saturation * 0.35
        + target_s * 0.65
    )

    new_saturation = np.clip(
        new_saturation,
        0,
        1
    )

    # Il colore target determina la tonalita.

    hue = np.full_like(
        value,
        target_h
    )

    # Conversione HSV -> RGB.

    c = (
        new_value
        * new_saturation
    )

    x = c * (
        1
        - np.abs(
            ((hue * 6) % 2) - 1
        )
    )

    m = new_value - c

    output = np.zeros_like(
        rgb
    )

    h6 = hue * 6

    region = (
        (h6 >= 0)
        & (h6 < 1)
    )

    output[region] = np.stack(
        [
            c[region],
            x[region],
            np.zeros_like(
                c[region]
            )
        ],
        axis=1
    )

    region = (
        (h6 >= 1)
        & (h6 < 2)
    )

    output[region] = np.stack(
        [
            x[region],
            c[region],
            np.zeros_like(
                c[region]
            )
        ],
        axis=1
    )

    region = (
        (h6 >= 2)
        & (h6 < 3)
    )

    output[region] = np.stack(
        [
            np.zeros_like(
                c[region]
            ),
            c[region],
            x[region]
        ],
        axis=1
    )

    region = (
        (h6 >= 3)
        & (h6 < 4)
    )

    output[region] = np.stack(
        [
            np.zeros_like(
                c[region]
            ),
            x[region],
            c[region]
        ],
        axis=1
    )

    region = (
        (h6 >= 4)
        & (h6 < 5)
    )

    output[region] = np.stack(
        [
            x[region],
            np.zeros_like(
                c[region]
            ),
            c[region]
        ],
        axis=1
    )

    region = (
        (h6 >= 5)
        & (h6 < 6)
    )

    output[region] = np.stack(
        [
            c[region],
            np.zeros_like(
                c[region]
            ),
            x[region]
        ],
        axis=1
    )

    output += m[:, :, None]

    result = np.zeros_like(
        array
    )

    result[:, :, :3] = (
        output * 255
    )

    result[:, :, 3] = alpha

    return Image.fromarray(
        result.astype(np.uint8),
        "RGBA"
    )