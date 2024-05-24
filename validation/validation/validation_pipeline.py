from typing import List

import tqdm
import torch
import numpy as np
from PIL import Image
from loguru import logger
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from sklearn.ensemble import IsolationForest


class ValidationPipeline:
    """ Class with implementation of the validation algorithm """
    def __init__(self, debug: bool = False):
        """ Constructor

        Parameters
        ----------
        debug: enable/disable debugging
        """

        self._model = None
        self._processor = None
        self._debug = debug
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)

        self._negative_prompts = [
            "empty",
            "nothing",
            "false",
            "wrong",
            "negative",
            "not quite right",
        ]

    def validate(self, images: List[Image.Image], prompt: str):
        """ Function for validating the input data

        Parameters
        ----------
        images: a list with rendered images defined as PIL.image
        prompt: a string with input prompt that was used for generating the 3D model

        Returns
        -------
        a float evaluation score
        """
        if not self._debug:
            logger.info(" Validating input data.")

        dists = []
        prompts = self._negative_prompts + [prompt,]
        for img, _ in zip(images, tqdm.trange(len(images), disable=True)):
            inputs = self._processor(text=prompts, images=[img], return_tensors="pt", padding=True)
            inputs.to(self._device)
            results = self._model(**inputs)

            # this is the image-text similarity score
            logits_per_image = results["logits_per_image"]

            # we can take the softmax to get the label probabilities
            probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()

            dists.append(probs[0][-1])

        dists = np.sort(dists)
        dists = dists.reshape(-1, 1)

        # searching for the anomalies
        # Set contamination to expected proportion of outliers
        clf = IsolationForest(contamination=0.1)
        clf.fit(dists)
        preds = clf.predict(dists)

        # removing found outliers from the input dists array
        outliers = np.where(preds == -1)[0]
        filtered_dists = np.delete(dists, outliers)
        score = np.median(filtered_dists)

        if self._debug:
            logger.debug(f" data: {dists.T}")
            logger.debug(f" outliers: {dists[outliers].T}")
            logger.debug(f" score: {score}")

        if not self._debug:
            logger.info(" Done.")

        return score

    def preload_scoring_model(self, scoring_model: str = "facebook/metaclip-b16-fullcc2.5b"):
        """ Function for preloading the MetaClip model

        Parameters
        ----------
        scoring_model: the MetaClip model to preload and use during the validation

        """
        logger.info(" Preloading MetaClip model for validation.")

        self._processor = AutoProcessor.from_pretrained(scoring_model)
        self._model = AutoModelForZeroShotImageClassification.from_pretrained(scoring_model).to(self._device)

        logger.info("[INFO] Done.")
