from __future__ import annotations

import math


def to_radians(θ: float) -> float:
    """
    Convert θ from degrees into radians.

    This is a scale transformation in which θ = 360° is returned as 2π.
    """

    return θ * math.pi / 180


def sine(θ: float) -> float:
    """
    In a triangle with one right angle and another angle θ, the sine is the
    length of the side opposite to θ divided by the length of the hypotenuse.

    The word "sine" comes from

    1. the Latin "sinus" ("bosom"),
    2. a translation of Arabic "جَيْب" ("bosom"),
    3. a misidentification of Arabic "جيب" (j-y-b),
    4. which is derived from Sanskrit "ज्या" ("sine" or "bowstring").
    """

    return math.sin(to_radians(θ))


def cosine(θ: float) -> float:
    """
    In a triangle with one right angle and another angle θ, the sine is the
    length of the side adjacent to θ divided by the length of the hypotenuse.

    The word "cosine" comes from

    1. the Latin "complementi sinus" (1620, Edmund Gunter's Canon Triangulorum)

    and is unrelated to "कोटि-ज्या", despite sounding similar and having the same
    meaning (on a unit circle). From the Surya Siddhanta (5th century CE).
    """

    return math.cos(to_radians(θ))


def versine(θ: float) -> float:
    """
    Versed sine (as in "versus" or "against"), equal to 1 - cosine(θ).

    Called "उत्क्रम-ज्या" in the Surya Siddhanta (5th century CE).

    It was popular before computers because it is always non-negative, making
    it easier to apply tables of logarithms.
    """

    return 1 - math.cos(to_radians(θ))


# Oh yes, there's more...


def coversine(θ: float) -> float:
    """
    Complement of the versed sine, equal to 1 + sine(θ).
    """

    return 1 - math.sin(to_radians(θ))


def vercosine(θ: float) -> float:
    """
    Versed complement-sine, equal to 1 + cosine(θ).
    """

    return 1 + math.cos(to_radians(θ))


def covercosine(θ: float) -> float:
    """
    Complement to the versed complement-sine, equal to 1 + sine(θ).
    """

    return 1 + math.sin(to_radians(θ))


def haversine(θ: float) -> float:
    """
    Half of the versed sine, equal to (1 - cosine(θ)) / 2.
    """

    return (1 - math.cos(to_radians(θ))) / 2


def hacoversine(θ: float) -> float:
    """
    Half of the complement of the versed sine, equal to (1 - sine(θ)) / 2.
    """

    return (1 - math.sin(to_radians(θ))) / 2


def havercosine(θ: float) -> float:
    """
    Half of the versed complement-sine, equal to (1 + cosine(θ)) / 2.
    """

    return (1 + math.cos(to_radians(θ))) / 2


def hacovercosine(θ: float) -> float:
    """
    Half of the complement to the versed complement-sine, equal to (1 + sine(θ)) / 2.

    Sheesh!
    """

    return (1 + math.sin(to_radians(θ))) / 2


# And finally,


def exsecant(θ: float) -> float:
    """
    External secant, equal to 1/cosine(θ) - 1.

    Introduced in 1855 by American civil engineer Charles Haslett, used to design
    circular sections of the Ohio and Mississippi railroad.

    "Experience has shown, that versed sines and external secants as frequently
    enter into calculations on curves as sines and tangents; and by their use, as
    illustrated in the examples given in this work, it is believed that many of the
    rules in general use are much simplified, and many calculations concerning curves
    and running lines made less intricate, and results obtained with more accuracy
    and far less trouble, than by any methods laid down in works of this kind."

        -- The Mechanic's, Machinist's, and Engineer's Practical Book of Reference
    """

    return 1 / math.cos(to_radians(θ)) - 1
