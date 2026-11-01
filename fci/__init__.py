"""TNO-FCI: one package, one command. See `fci --help`."""


class StepError(Exception):
    """A step stopped, with a message for the person running it."""
