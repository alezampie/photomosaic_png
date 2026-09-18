from PIL import Image


def normalize_image(image, size=512):
    """
    Porta un'immagine dentro un canvas quadrato mantenendo
    le proporzioni originali.

    L'immagine viene ridimensionata fino a stare completamente
    dentro il canvas e poi viene centrata.
    """

    image = image.convert("RGBA")

    width, height = image.size

    scale = min(size / width, size / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

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