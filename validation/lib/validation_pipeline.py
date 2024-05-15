import numpy as np
import tqdm
from transformers import CLIPProcessor, CLIPModel


class Validator:
    def __init__(self):
        self._model = None
        self._processor = None
        self._false_neg_thres = 0.4

        self._negative_prompts = [
            "empty",
            "nothing",
            "false",
            "wrong",
            "negative",
            "not quite right",
        ]

    def validate(self, images: list, prompt: str):
        print("[INFO] Starting validation ...")
        dists = []
        self._negative_prompts.append(prompt)
        prompts = self._negative_prompts
        for img, _ in zip(images, tqdm.trange(len(images))):
            inputs = self._processor(text=prompts, images=[img], return_tensors="pt", padding=True)
            results = self._model(**inputs)
            logits_per_image = results["logits_per_image"]  # this is the image-text similarity score
            probs = (
                logits_per_image.softmax(dim=1).cpu().detach().numpy()
            )  # we can take the softmax to get the label probabilities
            dists.append(probs[0][-1])

        dists = np.sort(dists)
        count_false_detection = np.sum(dists < self._false_neg_thres)
        if count_false_detection < len(dists):
            dists = dists[dists > self._false_neg_thres]

        print("[INFO] Done.")

        return np.mean(dists)

    def preload_scoring_model(self, scoring_model: str = "openai/clip-vit-base-patch16", dev="cuda"):
        print("[INFO] Preloading CLIP model for validation.")

        model = CLIPModel.from_pretrained(scoring_model)
        self._model = model.to(dev)
        self._processor = CLIPProcessor.from_pretrained(scoring_model)

        print("[INFO] Done.")
