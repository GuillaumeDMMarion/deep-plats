"""
"""
import requests


def get_data():
    root = "https://raw.githubusercontent.com/GuillaumeDMMarion/deep-plats/master/data/{}.csv"
    f = requests.get(url)
    return f.text
