from PIL import Image


def normalize_image(image, size=512, margin=0.05):
    """
    Porta il soggetto dentro un canvas quadrato.

    Il soggetto viene individuato tramite il canale alpha,
    ritagliato dalla trasparenza circostante e ingrandito
    fino a occupare quasi tutto il canvas.

    Le proporzioni e l'inclinazione originale vengono mantenute.
    """

    image = image.convert("RGBA")

    alpha = image.getchannel("A")

    bbox = alpha.getbbox()

    if bbox is None:
        return Image.new(
            "RGBA",
            (size, size),
            (0, 0, 0, 0)
        )

    image = image.crop(bbox)

    width, height = image.size

    available_size = int(size * (1 - margin * 2))

    scale = min(
        available_size / width,
        available_size / height
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGBA",
        (size, size),
        (0, 0, 0, 0)
    )

    x = (size - new_width) // 2
    y = (size - new_height) // 2

    canvas.paste(image, (x, y))

    return canvas