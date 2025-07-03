import glicko2

from validator.duels.ratings import (
    Glicko2Rating,
    update_glicko2,
    MAX_GLICKO_RATING,
    MIN_GLICKO_RATING,
    MIN_GLICKO_VOL,
    MAX_GLICKO_VOL,
    update_ratings,
    Rating,
)


class TestDuelRatings:

    def test_glicko_update_precise(self) -> None:
        left = Glicko2Rating(rating=1616.17, rd=150.0, vol=0.06)
        right = Glicko2Rating(rating=1616.17, rd=160.0, vol=0.03)

        update_glicko2(left, right, winner=1)

        player1 = glicko2.Player(rating=1616.17, rd=150.0, vol=0.06)
        player2 = glicko2.Player(rating=1616.17, rd=160.0, vol=0.03)

        player1.update_player(rating_list=[player2.rating], RD_list=[player2.rd], outcome_list=[True])
        assert left.rating == player1.rating
        assert left.rd == player1.rd
        assert left.vol == player1.vol

    def test_glicko_update_safeguards(self) -> None:
        left = Glicko2Rating(rating=MAX_GLICKO_RATING - 1)
        right = Glicko2Rating(rating=MAX_GLICKO_RATING)
        update_glicko2(left, right, winner=1)
        assert left.rating == MAX_GLICKO_RATING

        left = Glicko2Rating(rating=MAX_GLICKO_RATING)
        right = Glicko2Rating(rating=MAX_GLICKO_RATING - 1)
        update_glicko2(left, right, winner=2)
        assert right.rating == MAX_GLICKO_RATING

        left = Glicko2Rating(rating=MIN_GLICKO_RATING + 1)
        right = Glicko2Rating(rating=MIN_GLICKO_RATING)
        update_glicko2(left, right, winner=2)
        assert left.rating == MIN_GLICKO_RATING

        left = Glicko2Rating(rating=MIN_GLICKO_RATING)
        right = Glicko2Rating(rating=MIN_GLICKO_RATING + 1)
        update_glicko2(left, right, winner=1)
        assert right.rating == MIN_GLICKO_RATING

        left = Glicko2Rating(rating=2100.0, vol=MIN_GLICKO_VOL - 0.01)
        right = Glicko2Rating(rating=1500.0, vol=MIN_GLICKO_VOL)
        update_glicko2(left, right, winner=1)
        assert left.vol == MIN_GLICKO_VOL

        left = Glicko2Rating(rating=1500.0, vol=MIN_GLICKO_VOL)
        right = Glicko2Rating(rating=1600.0, vol=MIN_GLICKO_VOL - 0.01)
        update_glicko2(left, right, winner=2)
        assert right.vol == MIN_GLICKO_VOL

        left = Glicko2Rating(rating=1600.0, rd=200.0, vol=MAX_GLICKO_VOL + 0.01)
        right = Glicko2Rating(rating=1500.0, rd=200.0, vol=0.06)
        update_glicko2(left, right, winner=2)
        assert left.vol == MAX_GLICKO_VOL

        left = Glicko2Rating(rating=1500.0, vol=0.06)
        right = Glicko2Rating(rating=1600.0, vol=MAX_GLICKO_VOL + 0.01)
        update_glicko2(left, right, winner=1)
        assert right.vol == MAX_GLICKO_VOL

    def test_wrong_winner_value(self) -> None:
        left, right = Rating(), Rating()
        update_ratings(left, right, winner=3)

        assert left == Rating()
        assert right == Rating()
