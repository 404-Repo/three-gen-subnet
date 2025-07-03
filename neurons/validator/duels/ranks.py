from pathlib import Path

import bittensor as bt
import glicko2
import trueskill
from pydantic import BaseModel, Field


NEURONS_LIMIT = 256
MIN_ELO_RANK = 100.0
MAX_ELO_RANK = 3000.0

MIN_GLICKO_RATING = 0.0
MAX_GLICKO_RATING = 3000.0
MIN_GLICKO_RD = 30.0
MAX_GLICKO_RD = 350.0
MIN_GLICKO_VOL = 0.03
MAX_GLICKO_VOL = 0.2

MIN_TRUESKILL_SIGMA = 0.1
MAX_TRUESKILL_SIGMA = 15.0
MIN_TRUESKILL_MU = -50.0
MAX_TRUESKILL_MU = 100.0


class EloRank(BaseModel):
    rank: float = 1600.0


class Glicko2Rank(BaseModel):
    rank: float = 1500.0
    rd: float = 350.0
    vol: float = 0.5


class TrueSkillRank(BaseModel):
    mu: float = 25.0
    sigma: float = 8.33


class Rank(BaseModel):
    elo: EloRank = Field(default_factory=EloRank)
    glicko: Glicko2Rank = Field(default_factory=Glicko2Rank)
    trueskill: TrueSkillRank = Field(default_factory=TrueSkillRank)


class State(BaseModel):
    ranks: list[Rank]


class DuelRanks:
    def __init__(self) -> None:
        self._ranks = [Rank() for _ in range(NEURONS_LIMIT)]

    def get_miner_rank(self, miner_uid: int) -> Rank:
        return self._ranks[miner_uid]

    def reset_rank(self, miner_uid: int) -> None:
        self._ranks[miner_uid] = Rank()

    def load_ranks(self, *, full_path: Path) -> None:
        path = full_path / "ranks.json"
        if not path.exists():
            bt.logging.warning("No saved ranks found")

        try:
            with path.open("r") as f:
                content = f.read()
            self._ranks = State.model_validate_json(content).ranks
            bt.logging.info("Miners ranks loaded.")
        except Exception as e:
            bt.logging.exception(f"Failed to load the miners ranks: {e}")

    def save_ranks(self, *, full_path: Path) -> None:
        try:
            path = full_path / "ranks.json"
            with path.open("w") as f:
                f.write(State(ranks=self._ranks).model_dump_json())
        except Exception as e:
            bt.logging.exception(f"Validator state saving failed with {e}")


def update_ranks(left_rank: Rank, right_rank: Rank, winner: int) -> None:
    """
    In-place rank update required for de-registered miner handling.

    Ranks are references that may belong to recently de-registered miners.
    De-registered miners' ranks are used in calculations but not saved to state.
    In-place updates ensure we modify the actual referenced objects without
    needing to track which miners are still active during the update process.
    """
    update_elo(left_rank.elo, right_rank.elo, winner)
    update_glicko2(left_rank.glicko, right_rank.glicko, winner)
    update_trueskill(left_rank.trueskill, right_rank.trueskill, winner)


def update_elo(left: EloRank, right: EloRank, winner: int) -> None:
    if winner == 1:
        result = 1.0
    elif winner == 2:
        result = 0.0
    elif winner == 0:
        result = 0.5
    else:
        bt.logging.error(f"Unexpected duel result: winner = {winner}")
        return

    expected_left = 1.0 / (1.0 + 10 ** ((right.rank - left.rank) / 400))
    left.rank += 32 * (result - expected_left)
    right.rank += 32 * ((1 - result) - (1 - expected_left))

    # Safeguards:
    left.rank = max(MIN_ELO_RANK, min(MAX_ELO_RANK, left.rank))
    right.rank = max(MIN_ELO_RANK, min(MAX_ELO_RANK, right.rank))


def update_glicko2(left: Glicko2Rank, right: Glicko2Rank, winner: int) -> None:
    if winner == 1:
        result = 1.0
    elif winner == 2:
        result = 0.0
    elif winner == 0:
        result = 0.5
    else:
        bt.logging.error(f"Unexpected duel result: winner = {winner}")
        return

    left_player = glicko2.Player(rating=left.rank, rd=left.rd, vol=left.vol)
    right_player = glicko2.Player(rating=right.rank, rd=right.rd, vol=right.vol)

    left_player.update_player([right.rank], [right.rd], [result])
    right_player.update_player([left.rank], [left.rd], [1 - result])

    # Copy back the updated values
    left.rank = left_player.rating
    left.rd = left_player.rd
    left.vol = left_player.vol
    right.rank = right_player.rating
    right.rd = right_player.rd
    right.vol = right_player.vol

    # Safeguards:
    left.rank = max(MIN_GLICKO_RATING, min(MAX_GLICKO_RATING, left.rank))
    right.rank = max(MIN_GLICKO_RATING, min(MAX_GLICKO_RATING, right.rank))
    left.rd = max(MIN_GLICKO_RD, min(MAX_GLICKO_RD, left.rd))
    right.rd = max(MIN_GLICKO_RD, min(MAX_GLICKO_RD, right.rd))
    left.vol = max(MIN_GLICKO_VOL, min(MAX_GLICKO_VOL, left.vol))
    right.vol = max(MIN_GLICKO_VOL, min(MAX_GLICKO_VOL, right.vol))


def update_trueskill(left: TrueSkillRank, right: TrueSkillRank, winner: int) -> None:
    left_rating = trueskill.Rating(mu=left.mu, sigma=left.sigma)
    right_rating = trueskill.Rating(mu=right.mu, sigma=right.sigma)

    if winner == 1:
        updated_left, updated_right = trueskill.rate_1vs1(left_rating, right_rating)
    elif winner == 2:
        updated_right, updated_left = trueskill.rate_1vs1(right_rating, left_rating)
    elif winner == 0:
        updated_left, updated_right = trueskill.rate_1vs1(left_rating, right_rating, drawn=True)
    else:
        bt.logging.error(f"Unexpected duel result: winner = {winner}")
        return

    left.mu = max(MIN_TRUESKILL_MU, min(MAX_TRUESKILL_MU, updated_left.mu))
    right.mu = max(MIN_TRUESKILL_MU, min(MAX_TRUESKILL_MU, updated_right.mu))
    left.sigma = max(MIN_TRUESKILL_SIGMA, min(MAX_TRUESKILL_SIGMA, updated_left.sigma))
    right.sigma = max(MIN_TRUESKILL_SIGMA, min(MAX_TRUESKILL_SIGMA, updated_right.sigma))


duel_ranks = DuelRanks()
