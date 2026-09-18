import cv2


def calculate_sharpness(image):
    """
    Calcola quanto e nitida un'immagine.

    Usa la varianza del Laplaciano: immagini con piu dettagli
    e bordi definiti tendono ad avere un valore piu alto.

    Restituisce un numero che possiamo usare per confrontare
    la nitidezza di diversi frame.
    """

    # Il Laplaciano viene calcolato su un'immagine in scala di grigi.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # La varianza del Laplaciano viene usata come misura della nitidezza.
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return sharpness