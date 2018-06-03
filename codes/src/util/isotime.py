# -*- coding: utf-8 -*-
"""@package isotime
Creates a string containing the current local time in ISO 8601 basic format

@author: Chris Mirabito (mirabito@mit.edu)
"""
from datetime import datetime
#from matplotlib.dates import SEC_PER_DAY
SEC_PER_DAY = 86400


def isotime():
    """Current local time in ISO 8601 basic format
    @return String containing the current local time in ISO 8601 basic format
    """
    local_time = datetime.now()
    utc_time = datetime.utcnow()
    time_diff = local_time - utc_time

    hours, minutes = divmod(
        (time_diff.days * SEC_PER_DAY + time_diff.seconds + 30) // 60, 60)
    return ('{}{:+03d}{:02d}'
            .format(local_time.strftime('%Y%m%dT%H%M%S'),
                    int(hours), int(minutes)))
