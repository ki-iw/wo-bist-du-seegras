def convert_msec(msec):
    """Convert milliseconds to hours, minutes, seconds, and milliseconds."""
    milliseconds = int((msec % 1000) / 100)
    seconds = int(msec / 1000) % 60
    minutes = int(msec / (1000 * 60)) % 60
    hours = int(msec / (1000 * 60 * 60)) % 24
    return hours, minutes, seconds, milliseconds
