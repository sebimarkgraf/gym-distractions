class GymDistractionsError(Exception):
    """Error thrown by the gym-distraction package.

    Catch this error if you want to automatically catch all errors
    created by the package.
    """

    pass


class GymDistractionsTypeError(GymDistractionsError):
    """A type given to a package method is wrong."""

    pass
