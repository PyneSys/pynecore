__all__ = [
    'isfirst',
    'islast',
    'isconfirmed',
    'ishistory',
    'islastconfirmedhistory',
    'isnew',
    'isrealtime'
]

isfirst = True
""" Returns true if current bar is first bar in barset, false otherwise."""

islast = False
""" Returns true if current bar is the last bar in barset, false otherwise. """

isconfirmed = True
""" Returns true if the script is calculating the last (closing) update of the current bar. """

ishistory = True
""" Returns true if script is calculating on historical bars, false otherwise. """

islastconfirmedhistory = False
""" Returns true on the last historical bar before real-time bars begin. """

isnew = False
""" Returns true if script is currently calculating on new bar, false otherwise. """

isrealtime = False
""" Returns true if script is calculating on real-time bars, false otherwise. """
