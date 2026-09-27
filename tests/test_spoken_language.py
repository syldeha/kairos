"""Kairos prepares its thoughts in the language the room speaks now, not the meeting's default."""

from kairos.agents.common import spoken_language


def test_a_short_line_keeps_the_language_the_room_speaks() -> None:
    lines = ["Let's say we're planning a trip to Groningen for my sister's defense.", "We're going with my boys."]
    assert spoken_language(lines, "French") == "English"


def test_a_switch_back_is_followed_as_soon_as_it_can_be_told() -> None:
    lines = ["What about our trip?", "Bon, on en reparle demain, c'est plus simple pour tout le monde."]
    assert spoken_language(lines, "English") == "French"


def test_without_any_telling_line_the_default_stays() -> None:
    assert spoken_language(["OK.", "Groningen."], "French") == "French"


def test_kairos_misheard_as_qui_ose_is_still_a_call_but_not_inside_a_sentence() -> None:
    from kairos.addressing import NAME
    assert NAME.search("Qui ose ?") and NAME.search("qui ose, tu aurais des idées ?") and NAME.search("Kéros, tu es là ?")
    assert not NAME.search("Qui ose dire que c'est trop cher ?")
