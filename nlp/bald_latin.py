# -*- coding: utf-8 -*-
cyrillic = set(u"АаБбВвГгДдЂђЕеЖжЗзИиЈјКкЛлЉљМмНнЊњОоПпРрСсТтЋћУуФфХхЦцЧчЏџШш")


def remove_cyrillic_comments(comments, labels, print_perc=True):
    cyrillic_count = 0.0; all_count = len(comments)
    
    clean_coms = []
    clean_labels = []
    
    for comment,label in zip(comments, labels):
        if not bool(set(comment.decode('utf8')).intersection(cyrillic)):
            clean_coms.append(comment)
            clean_labels.append(label)
        else:
            cyrillic_count += 1
        
    if print_perc:
        print "Cyrillic comments make up %s percent" % (cyrillic_count / all_count * 100)
        
    return clean_coms, clean_labels


def remove_serbian_accents(comments):
    bald = []
    for comment in comments:
        bald_comment = (comment.decode('utf8')) \
            .replace(u"ć", u"c") \
            .replace(u"č", u"c") \
            .replace(u"š", u"s") \
            .replace(u"đ", u"dj") \
            .replace(u"ž", u"z")

        bald.append(bald_comment)
        
    return bald
