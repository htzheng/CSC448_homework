#!/usr/bin/python

import collections
from math import *
from sys import *

def parse_line(line):
    """Parse a single sentence line into its parts

    Example:
        "3 Britain NNP is VBZ big JJ" ->
        
        (["Britain", "is", "big"], ["NNP", "VBZ", "JJ"])
    """
    parts = line.strip().split(" ")
    words = parts[1::2]
    speech_parts = parts[2::2]

    return (words, speech_parts)

def read_file(filename):
    """Read all the sentences from a file"""
    sentences = []
    with open(filename) as f:
        for line in f.readlines():
            sentences.append(parse_line(line))
    return sentences

def get_pos_to_pos_probabilities(data):
    """Generate a table of probabilities of going from one part of speech to
    another from the given data
    """
    counts = collections.defaultdict(lambda: collections.defaultdict(float))

    for _, parts in data:
        for curr, next in zip(parts, parts[1:]):
            counts[curr][next] += 1

    for nextcounts in counts.itervalues():
        total = sum(nextcounts.values())
        for next in nextcounts:
            nextcounts[next] /= total

    for curr in counts.iterkeys():
        for next in counts[curr].iterkeys():
            print "T_"+curr+"_"+next, log(counts[curr][next])
            
    return counts

def get_pos_to_word_probabilities(data):
    """Generate a table of probabilities of generating a part of speech from a given
    word to another from the given data
    """
    counts = collections.defaultdict(lambda: collections.defaultdict(float))

    for words, parts in data:
        for word, part in zip(words, parts):
            counts[part][word] += 1

    for partcounts in counts.itervalues():
        total = sum(partcounts.values())
        for part in partcounts:
            partcounts[part] /= total

    for curr in counts.iterkeys():
        for next in counts[curr].iterkeys():
            print "E_"+curr+"_"+next, log(counts[curr][next])
            
    return counts

def get_file_data(filename):
    """Get the raw data and the two tables of probabilities from a given file"""
    data = read_file(filename)

    return (data,
            get_pos_to_word_probabilities(data),
            get_pos_to_pos_probabilities(data))

get_file_data(argv[1])
