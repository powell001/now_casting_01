import requests
import json
import pandas as pd

strictBefore = "2024-01-01"
after = "2023-12-01"

url = "https://api.netanders.io/v1/utilizations?point=0&type=18&granularity=6&granularitytimezone=1&classification=2&activity=1&validfrom[strictly_before]="+strictBefore+"&validfrom[after]="+after

headers = {
  'X-AUTH-TOKEN': '26c87ac7bd620faf82bd7038dde61f5dd21e4868cb12ae5ddaafaae22c5d2c88',
  'accept': 'application/ld+json' }

payload = {}
response = requests.request("GET", url, headers=headers, data=payload, allow_redirects=False, verify=False)
dic = response.json()

df1 = dic['hydra:member']
print(pd.DataFrame.from_dict(df1))