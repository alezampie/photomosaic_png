from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BackgroundRemover:
    """
    Gestisce RMBG-2.0 per rimuovere lo sfondo dalle immagini.
    """

    def __init__(self):
        print("Caricamento RMBG-2.0...")

        self.model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0",
            trust_remote_code=True
        )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        print("RMBG-2.0 caricato.")

    def remove_background(self, image):
        """
        Rimuove lo sfondo e restituisce un'immagine RGBA.
        """

        image = image.convert("RGB")

        original_size = image.size

        input_image = self.transform(image).unsqueeze(0)

        print("Rimozione dello sfondo...")

        with torch.no_grad():
            prediction = self.model(input_image)[-1].sigmoid().cpu()

        mask = prediction[0].squeeze()

        mask = transforms.ToPILImage()(mask)

        mask = mask.resize(original_size)

        image.putalpha(mask)

        return image