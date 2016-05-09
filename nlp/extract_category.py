import sys
from pymongo import MongoClient


def get_comments_in_category(category):
    client = MongoClient()
    blic = client.scrape.blic

    return blic.find({'link': {'$regex': category}}, {'comment': 1, '_id': 0})
     

