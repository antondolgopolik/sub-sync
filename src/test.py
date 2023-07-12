import datetime
import time

t = datetime.datetime.fromtimestamp(0.012, tz=datetime.timezone.utc)
print(t.strftime("%H:%M:%S,%f")[:-3])
