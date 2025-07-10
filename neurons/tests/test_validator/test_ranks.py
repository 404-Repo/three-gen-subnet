import trueskill
import glicko2

from validator.duels.ranks import (
    EloRank,
    update_elo,
    MAX_ELO_RANK,
    MIN_ELO_RANK,
    Glicko2Rank,
    update_glicko2,
    MAX_GLICKO_RATING,
    MIN_GLICKO_RATING,
    MIN_GLICKO_RD,
    MAX_GLICKO_RD,
    MIN_GLICKO_VOL,
    MAX_GLICKO_VOL,
    TrueSkillRank,
    update_trueskill,
    update_ranks,
    Rank,
)


class TestDuelRanks:

    def test_elo_update(self) -> None:
        left, right = EloRank(rank=1600.0), EloRank(rank=1600.0)

        update_elo(left, right, winner=0)
        assert left.rank == 1600.0
        assert right.rank == 1600.0

        update_elo(left, right, winner=1)
        assert left.rank + right.rank == 3200.0
        assert left.rank > right.rank

        update_elo(left, right, winner=2)
        assert left.rank + right.rank == 3200.0
        assert left.rank < right.rank

    def test_elo_update_precise(self) -> None:
        left, right = EloRank(rank=1616.17), EloRank(rank=1610.43)
        update_elo(left, right, winner=1)

        expected_left = 1.0 / (1.0 + 10 ** ((1610.43 - 1616.17) / 400))
        assert left.rank == 1616.17 + 32 * (1 - expected_left)
        assert right.rank == 1610.43 + 32 * (expected_left - 1)

    def test_elo_safeguards(self) -> None:
        left, right = EloRank(rank=MAX_ELO_RANK - 1), EloRank(rank=MAX_ELO_RANK)
        update_elo(left, right, winner=1)
        assert left.rank == MAX_ELO_RANK

        left, right = EloRank(rank=MIN_ELO_RANK + 1), EloRank(rank=MIN_ELO_RANK)
        update_elo(left, right, winner=2)
        assert left.rank == MIN_ELO_RANK

        left, right = EloRank(rank=MAX_ELO_RANK), EloRank(rank=MAX_ELO_RANK - 1)
        update_elo(left, right, winner=2)
        assert right.rank == MAX_ELO_RANK

        left, right = EloRank(rank=MIN_ELO_RANK), EloRank(rank=MIN_ELO_RANK + 1)
        update_elo(left, right, winner=1)
        assert right.rank == MIN_ELO_RANK

    def test_glicko_update_precise(self) -> None:
        left = Glicko2Rank(rank=1616.17, rd=150.0, vol=0.06)
        right = Glicko2Rank(rank=1616.17, rd=160.0, vol=0.03)

        update_glicko2(left, right, winner=1)

        player1 = glicko2.Player(rating=1616.17, rd=150.0, vol=0.06)
        player2 = glicko2.Player(rating=1616.17, rd=160.0, vol=0.03)

        player1.update_player(rating_list=[player2.rating], RD_list=[player2.rd], outcome_list=[True])
        assert left.rank == player1.rating
        assert left.rd == player1.rd
        assert left.vol == player1.vol

    def test_glicko_update_safeguards(self) -> None:
        left = Glicko2Rank(rank=MAX_GLICKO_RATING - 1)
        right = Glicko2Rank(rank=MAX_GLICKO_RATING)
        update_glicko2(left, right, winner=1)
        assert left.rank == MAX_GLICKO_RATING

        left = Glicko2Rank(rank=MAX_GLICKO_RATING)
        right = Glicko2Rank(rank=MAX_GLICKO_RATING - 1)
        update_glicko2(left, right, winner=2)
        assert right.rank == MAX_GLICKO_RATING

        left = Glicko2Rank(rank=MIN_GLICKO_RATING + 1)
        right = Glicko2Rank(rank=MIN_GLICKO_RATING)
        update_glicko2(left, right, winner=2)
        assert left.rank == MIN_GLICKO_RATING

        left = Glicko2Rank(rank=MIN_GLICKO_RATING)
        right = Glicko2Rank(rank=MIN_GLICKO_RATING + 1)
        update_glicko2(left, right, winner=1)
        assert right.rank == MIN_GLICKO_RATING

        left = Glicko2Rank(rank=2100.0, vol=MIN_GLICKO_VOL - 0.01)
        right = Glicko2Rank(rank=1500.0, vol=MIN_GLICKO_VOL)
        update_glicko2(left, right, winner=1)
        assert left.vol == MIN_GLICKO_VOL

        left = Glicko2Rank(rank=1500.0, vol=MIN_GLICKO_VOL)
        right = Glicko2Rank(rank=1600.0, vol=MIN_GLICKO_VOL - 0.01)
        update_glicko2(left, right, winner=2)
        assert right.vol == MIN_GLICKO_VOL

        left = Glicko2Rank(rank=1600.0, rd=200.0, vol=MAX_GLICKO_VOL + 0.01)
        right = Glicko2Rank(rank=1500.0, rd=200.0, vol=0.06)
        update_glicko2(left, right, winner=2)
        assert left.vol == MAX_GLICKO_VOL

        left = Glicko2Rank(rank=1500.0, vol=0.06)
        right = Glicko2Rank(rank=1600.0, vol=MAX_GLICKO_VOL + 0.01)
        update_glicko2(left, right, winner=1)
        assert right.vol == MAX_GLICKO_VOL

    def test_trueskill_precise(self) -> None:
        left, right = TrueSkillRank(mu=26.0, sigma=8), TrueSkillRank(mu=22.0, sigma=11)
        update_trueskill(left, right, winner=1)

        left_rating = trueskill.Rating(mu=26.0, sigma=8)
        right_rating = trueskill.Rating(mu=22.0, sigma=11)

        updated_left, updated_right = trueskill.rate_1vs1(left_rating, right_rating)

        assert left.mu == updated_left.mu
        assert left.sigma == updated_left.sigma
        assert right.mu == updated_right.mu
        assert right.sigma == updated_right.sigma

        left, right = TrueSkillRank(mu=22.0, sigma=11), TrueSkillRank(mu=26.0, sigma=8)
        update_trueskill(left, right, winner=2)

        assert left.mu == updated_right.mu
        assert left.sigma == updated_right.sigma
        assert right.mu == updated_left.mu
        assert right.sigma == updated_left.sigma

    def test_wrong_winner_value(self) -> None:
        left, right = Rank(), Rank()
        update_ranks(left, right, winner=3)

        assert left == Rank()
        assert right == Rank()
