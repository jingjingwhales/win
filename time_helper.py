from datetime import datetime, timezone, timedelta

def convert_to_milli_time(timestamp):
    return timestamp.replace(tzinfo=timezone.utc).timestamp() * 1000

def last_4_hours_milli_time():
    return convert_to_milli_time(datetime.now() - timedelta(hours=4))

def last_30_mins_milli_time():
    return convert_to_milli_time(datetime.now() - timedelta(hours=0.5))