# -*-coding:utf-8-*-

import os


def append_omega(input_filename, *args): 
    """
    Appends and omega to the end of every line.
    """
    if len(args) == 0:
        output_filename = input_filename.replace('.', '_omega.')
    else: 
        output_filename = args[0]

    os.system("cat %s(input) | sed -e 's/$/ Ω' > %s(output)" % {'input': input_filename, 'output': output_filename})


def stem(input_filename, *args):
    """
    Runs the croatian stemmer on the input_filename.
    If the args contains an output filename, stores the result in there.
    Otherwise creates a new file with the _stemmed suffix.
    """
    if len(args) == 0:
        output_filename = input_filename.replace('.', '_stemmed.')
    else: 
        output_filename = args[0]

    os.system('python cro_stemmer/Croatian_stemmer.py %s %s' % (input_filename, output_filename))


def parse(input_filename, output_filename):
    """
    Takes the result from the croatian parser and converts it back to a comments file
    """

    input_file = open(input_filename, 'r')
    output_file = open(output_filename, 'w')

    comment = "" 
    flush_count = 0

    while True:
        line = input_file.readline()
        if not line:
            break

        if u"ω".encode('utf8') in line:
            output_file.write(comment + "\n")
            comment = ""
            continue

        flush_count += 1
        if flush_count % 10000 == 0:
            output_file.flush()
            flush_count = 0

        first_word = line.split()[1]
        comment += first_word + " "

    input_file.close()
    output_file.close()


def find_char_set(comments_file):
    """
    Returns a set of characters used in the comments_file
    Useful for figuring out if the comments file contains cyrillic/symbols.
    """
    input_file = open(comments_file, 'r')
    
    chars = set()

    while True:
        line = input_file.readline()
        if not line: 
            break

        chars.update(line.decode('utf8'))

    return chars


#def comment_is_strange(comment, chars=('):
    #"""
    #Returns true if the comment contains 
