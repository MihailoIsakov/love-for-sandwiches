#! /usr/bin/env python

import sys
from pymongo import MongoClient


def get_labeled_comment_dict(): 
    client = MongoClient()
    blic = client.blic
    b92 = client.b92
    n1 = client.n1
    
    blic_comments = list(blic.comments.find())
    b92_comments = list(b92.comments.find())
    n1_comments = list(n1.comments.find())
    
    blic_reactions = list(blic.reactions.find())
    b92_reactions = list(b92.reactions.find())
    n1_reactions = list(n1.reactions.find())
    
    #all_comments = blic_comments + b92_comments + n1_comments
    all_comments = blic_comments + n1_comments
    all_reactions = blic_reactions + n1_reactions
    
    # Comment dictionary with comment_id's as indices
    comments_dict = {}
    for comment in all_comments:
        # FIXME: For some reason multiple same comments pop up. 
        comment['reactions'] = []
        comments_dict[comment['comment_id']] = comment
    
    # Add the reactions to each comment
    for reaction in all_reactions:
        if reaction['comment_id'] not in comments_dict:
            #         print("Missing comment with ID: %s" % reaction['comment_id'])
            pass
        else:
            comments_dict[reaction['comment_id']]['reactions'].append(reaction)     

    return comments_dict


def save_comment_dict(comments):

    comment_output = open("lns_comments.txt", 'w')
    label_output = open("lns_labels.txt", 'w')

    for comment in comments.values():
        # bot/not ratio
        bt = 0; nt = 0;
        for reaction in comment['reactions']:
            if reaction['bot'] == 'false':
                nt += 1
            else:
                bt += 1
        
        if comment['comment'] is not None:
            text = comment['comment'].replace('\n', ' ') + '\n'
            comment_output.write(text.encode('utf8'))
            label_output.write(str(bt / (bt + nt + 0.0)) + "\n")

    comment_output.close()
    label_output.close()


def main():
    comments = get_labeled_comment_dict()
    save_comment_dict(comments)


if __name__ == "__main__":
    main()

    
