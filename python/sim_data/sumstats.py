import numpy as np
import pkgutil
import logging
import os
import sys
import json
import csv
import collections
import glob
from random import sample


def get_diff_loc(s1, s2):
    """Gets index locations of differences between s1 and s2"""
    diffs = [
        i for i in range(len(s1)) if s1[i] != s2[i] and s1[i] != "N" and s2[i] != "N"
    ]
    return diffs


def get_pairwise_dists(locs, length):
    """Given sequence pair difference locations and sequence total length, 
    calculates the pairwise distances between all of the differences and divides by total number of pairs at that distance"""
    arr = np.zeros(length - 1)
    for i in range(len(locs)):
        for j in range(len(locs) - i - 1):
            arr[np.abs(locs[i] - locs[i + j + 1]) - 1] = (
                arr[np.abs(locs[i] - locs[i + j + 1]) - 1] + 1
            )
    if np.sum(arr[0:100]) == 0:
        return arr[0:100]
    else:
        arr_norm = arr / np.sum(arr)
        range_deno = np.linspace(length - 1, 1, length - 1)
        range_deno = range_deno / np.sum(range_deno)
        arr_norm = arr_norm / range_deno
    return arr_norm[0:100]


def colocal_vector(parent_seqs, child_seqs):
    """"Calculate the colacalization from distance 1 to distance 100 for parent_seqs and child_seqs"""
    diff_list = []
    for i in range(len(parent_seqs)):
        diff_list.append(get_diff_loc(parent_seqs[i], child_seqs[i]))
    length_list = [len(i) for i in parent_seqs]
    arr = np.zeros(100)
    comps = 0
    for i in range(len(diff_list)):
        t_arr = get_pairwise_dists(diff_list[i], length_list[i])
        if len(diff_list[i]) < 2:
            arr = arr + t_arr
        else:
            arr = arr + t_arr
            comps = comps + 1
    return arr / comps


def get_cg_summ(seq, germline):
    """"Get number of c mutations and number of g mutations from a template-seq pair"""
    c_mut_count = np.sum(
        [germline[i] != seq[i] and germline[i] == "C" for i in range(len(germline))]
    )
    g_mut_count = np.sum(
        [germline[i] != seq[i] and germline[i] == "G" for i in range(len(germline))]
    )
    return c_mut_count, g_mut_count


def get_cg_summ_sample(seqs, germlines):
    """Get total c mut count and g mut count over sample of pairs"""
    c_counts = []
    g_counts = []
    for i in range(len(seqs)):
        t_c, t_g = get_cg_summ(seqs[i], germlines[i])
        c_counts.append(t_c)
        g_counts.append(t_g)
    return c_counts, g_counts


def base_prob(seqs, germlines):
    """Get base mutation freq over a sample"""
    num = 0
    deno = 0
    for i in range(len(seqs)):
        deno = deno + len(seqs[i])
        num = num + np.sum([seqs[i][j] != germlines[i][j] for j in range(len(seqs[i]))])
    return num / deno


def at_frac(seqs, germlines):
    """Get fraction of total mutations that are at a/t sites over a sample"""
    num = 0
    deno = 0
    for i in range(len(seqs)):
        deno = deno + np.sum(
            [seqs[i][j] != germlines[i][j] for j in range(len(seqs[i]))]
        )
        num = num + np.sum(
            [
                (seqs[i][j] != germlines[i][j]) and (germlines[i][j] in ["A", "T"])
                for j in range(len(seqs[i]))
            ]
        )
    return num / deno


def get_exo_summ(seq,germline):
    mut_ind = np.where([seq[i] != germline[i] and (germline[i] == 'A' or germline[i] == 'T') for i in range(len(germline))])
    c_dist = shortestDistance(germline, 'C')
    g_dist = shortestDistance(germline, 'G')
    min_dist = np.minimum(c_dist,g_dist)
    return min_dist[mut_ind]

def get_pairwise_at(seq,germline):
    mut_ind = np.where([seq[i] != germline[i] and (germline[i] == 'A' or germline[i] == 'T') for i in range(len(germline))])[0]
    dists = []
    for i in range(len(mut_ind)):
        for j in range(i):
            dists.append(mut_ind[i]-mut_ind[j])
    return(dists)

def get_pairwise(seq,germline):
    mut_ind = np.where([seq[i] != germline[i] for i in range(len(germline))])[0]
    dists = []
    for i in range(len(mut_ind)):
        for j in range(i):
            dists.append(mut_ind[i]-mut_ind[j])
    return(dists)

def shortestDistance(S, X):
 
    # Find distance from occurrences of X
    # appearing before current character.
    inf = float('inf')
    prev = inf
    ans = []
    for i,j in enumerate(S):
        if S[i] == X:
            prev = i
        if (prev == inf) :
            ans.append(inf)
        else :    
            ans.append(i - prev)
 
 
    # Find distance from occurrences of X
    # appearing after current character and
    # compare this distance with earlier.
    prev = inf
    for i in range(len(S) - 1, -1, -1):
        if S[i] == X:
            prev = i
        if (X != inf):   
            ans[i] = min(ans[i], prev - i)
 
    # return array of distance
    return ans