import pandas as pd
import os

from datetime import datetime as dt

all_dates = []

with open('sunspot_data/sunspot_1874.txt', 'r') as f:
    for line in f.readlines():
        date = line[:8]
        
        date_l = [d for d in date if d != ' ']
        print(date_l)
        date = "".join(date_l)
        date_obj = dt.strptime(date, "%Y%m%d")
        print(date_obj)
        all_dates.append(date_obj)