import numpy as np
import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.ensemble import IsolationForest


class Validator:
    def __init__(self, debug: bool = False):
        self._model = None
        self._processor = None
        self._debug = debug

        self._negative_prompts = [
            "empty",
            "nothing",
            "false",
            "wrong",
            "negative",
            "not quite right",
        ]

    def validate(self, images: list, prompt: str):
        if not self._debug:
            print("[INFO] Starting validation ...")
        dists = []
        prompts = self._negative_prompts + [prompt,]
        for img, _ in zip(images, tqdm.trange(len(images), disable=True)):
            inputs = self._processor(text=prompts, images=[img], return_tensors="pt", padding=True)
            results = self._model(**inputs)
            logits_per_image = results["logits_per_image"]  # this is the image-text similarity score
            probs = (
                logits_per_image.softmax(dim=1).cpu().detach().numpy()
            )  # we can take the softmax to get the label probabilities
            dists.append(probs[0][-1])

        dists = np.sort(dists)

        dists = dists.reshape(-1, 1)
        clf = IsolationForest(contamination=0.1)  # Set contamination to expected proportion of outliers
        clf.fit(dists)
        preds = clf.predict(dists)
        outliers = np.where(preds == -1)[0]
        filtered_dists = np.delete(dists, outliers)
        score = np.median(filtered_dists)

        if self._debug:
            print("data: ", dists.T)
            print("outliers: ", dists[outliers].T)
            print("score: ", score)

        if not self._debug:
            print("[INFO] Done.")

        return score

    def preload_scoring_model(self, scoring_model: str = "openai/clip-vit-base-patch16", dev="cuda"):
        print("[INFO] Preloading CLIP model for validation.")

        model = CLIPModel.from_pretrained(scoring_model)
        self._model = model.to(dev)
        self._processor = CLIPProcessor.from_pretrained(scoring_model)

        print("[INFO] Done.")
