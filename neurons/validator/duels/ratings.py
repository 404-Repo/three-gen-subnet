from pathlib import Path

import bittensor as bt
import glicko2
from pydantic import BaseModel, Field


NEURONS_LIMIT = 256

MIN_GLICKO_RATING = 0.0
MAX_GLICKO_RATING = 3000.0
MIN_GLICKO_RD = 30.0
MAX_GLICKO_RD = 350.0
MIN_GLICKO_VOL = 0.03
MAX_GLICKO_VOL = 0.2


class Glicko2Rating(BaseModel):
    rating: float = 1500.0
    rd: float = 350.0
    vol: float = 0.5


class Rating(BaseModel):
    glicko: Glicko2Rating = Field(default_factory=Glicko2Rating)


class State(BaseModel):
    ratings: list[Rating]


class DuelRatings:
    def __init__(self) -> None:
        self._ratings = [Rating() for _ in range(NEURONS_LIMIT)]

    def get_miner_rating(self, miner_uid: int) -> Rating:
        return self._ratings[miner_uid]

    def get_miner_reward_rating(self, miner_uid: int) -> float:
        """Returns miner rating used to calculate the reward."""
        return self._ratings[miner_uid].glicko.rating

    def get_reward_ratings(self) -> list[float]:
        """Returns a list of ratings used to calculate the rewards."""
        return [r.glicko.rating for r in self._ratings]

    def reset_rating(self, miner_uid: int) -> None:
        self._ratings[miner_uid] = Rating()

    def load_ratings(self, *, full_path: Path) -> None:
        path = full_path / "ratings.json"
        if not path.exists():
            bt.logging.warning("No saved ratings found")

        try:
            with path.open("r") as f:
                content = f.read()
            self._ratings = State.model_validate_json(content).ratings
            bt.logging.info("Miners ratings loaded.")
        except Exception as e:
            bt.logging.exception(f"Failed to load the miners ratings: {e}")

    def save_ratings(self, *, full_path: Path) -> None:
        try:
            path = full_path / "ratings.json"
            with path.open("w") as f:
                f.write(State(ratings=self._ratings).model_dump_json())
        except Exception as e:
            bt.logging.exception(f"Validator state saving failed with {e}")


def update_ratings(left_rating: Rating, right_rating: Rating, winner: int) -> None:
    """
    In-place rating update required for de-registered miner handling.

    Ratings are references that may belong to recently de-registered miners.
    De-registered miners' ratings are used in calculations but not saved to state.
    In-place updates ensure we modify the actual referenced objects without
    needing to track which miners are still active during the update process.
    """
    update_glicko2(left_rating.glicko, right_rating.glicko, winner)


def update_glicko2(left: Glicko2Rating, right: Glicko2Rating, winner: int) -> None:
    if winner == 1:
        result = 1.0
    elif winner == 2:
        result = 0.0
    elif winner == 0:
        result = 0.5
    else:
        bt.logging.error(f"Unexpected duel result: winner = {winner}")
        return

    left_player = glicko2.Player(rating=left.rating, rd=left.rd, vol=left.vol)
    right_player = glicko2.Player(rating=right.rating, rd=right.rd, vol=right.vol)

    left_player.update_player([right.rating], [right.rd], [result])
    right_player.update_player([left.rating], [left.rd], [1 - result])

    # Copy back the updated values
    left.rating = left_player.rating
    left.rd = left_player.rd
    left.vol = left_player.vol
    right.rating = right_player.rating
    right.rd = right_player.rd
    right.vol = right_player.vol

    # Safeguards:
    left.rating = max(MIN_GLICKO_RATING, min(MAX_GLICKO_RATING, left.rating))
    right.rating = max(MIN_GLICKO_RATING, min(MAX_GLICKO_RATING, right.rating))
    left.rd = max(MIN_GLICKO_RD, min(MAX_GLICKO_RD, left.rd))
    right.rd = max(MIN_GLICKO_RD, min(MAX_GLICKO_RD, right.rd))
    left.vol = max(MIN_GLICKO_VOL, min(MAX_GLICKO_VOL, left.vol))
    right.vol = max(MIN_GLICKO_VOL, min(MAX_GLICKO_VOL, right.vol))


duel_ratings = DuelRatings()
