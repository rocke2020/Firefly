import json
import logging
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from icecream import ic

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
sys.path.append(os.path.abspath('.'))

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
punc_at_ends = re.compile(r'^\W+|\W+$')


def read_chem_x_gene():
    """  """
    file = Path('/mnt/nas1/corpus-bio-nlp/NER/PGx_CTD_chem_x_gene.csv')
    df_data = pd.read_csv(file)
    ic(df_data.columns.to_list())
    ic(df_data.entity_type.unique())
    unique_entity_names = df_data.entity_name.unique().tolist()
    ic(len(unique_entity_names))
    ic('genes' in df_data.entity_name.unique().tolist())
    ic('proteins' in df_data.entity_name.unique().tolist())
    ic('chemicals' in df_data.entity_name.unique().tolist())

    # check short general entities
    possible_general_entities = []
    short_general_entities = ['gene', 'protein', 'chemical', 'drug']
    for short_entity in short_general_entities:
        for unique_entity in unique_entity_names:
            if short_entity in unique_entity:
                possible_general_entities.append(unique_entity)
    possible_general_entities = sorted(possible_general_entities, key=lambda x: len(x.split()))
    possible_general_entities_file = 'possible_general_entities_file.log'
    with open(possible_general_entities_file, 'w') as f:
        f.write('\n'.join(possible_general_entities))

    # filter general entities, not filter drup
    # general_entities = ['genes', 'proteins', 'multidrug', 'drug']
    general_entities = ['genes', 'proteins']
    df_data = df_data[~df_data.entity_name.isin(general_entities)]
    df_data.rename(columns={"id": "uid"}, inplace=True)

    # save and check uid contains pgx_
    # _df_data = df_data[df_data["uid"].str.contains("pgx_")]
    # ic(len(_df_data))
    # file = 'PGx_CTD_chem_x_gene_pgx-starts.csv'
    # _df_data.to_csv(file, index=False)

    ic(len(df_data), df_data.sentence.nunique())

    file = 'PGx_CTD_chem_x_gene_sentence_same_sorted.csv'
    df_data = df_data.sort_values(by=['sentence'])
    df_data.to_csv(file, index=False)
    return df_data


def check_indexes(sorted_ents: list):
    """  """
    ## 1 same type NER are splitted as 2 parts; 2 NER types with overlap in sentence
    former_ent = None
    for entity in sorted_ents:
        if former_ent:
            former_ini = former_ent[0]
            former_end = former_ent[1]
            former_type = former_ent[3]
            current_ini = entity[0]
            current_end = entity[1]
            current_type = entity[3]
            if current_ini < former_end:
                # ic(former_ent, entity)
                # Allows overlap for different NER types.
                # former_ent: (61, 67, 'MEK1/2', 'Gene')
                # entity: (61, 77, 'MEK1/2 inhibitor', 'Chemical')
                if former_type == current_type:
                    ic(former_ent, entity)
                    ## for high recall, not remove.
                    # former_ent: (80, 109, 'in combination with cetuximab', 'Chemical')
                    # entity: (100, 109, 'cetuximab', 'Chemical')
                    # former_ent: (141, 158, 'chemotherapy with', 'Chemical')
                    # entity: (141, 170, 'chemotherapy with carboplatin', 'Chemical')
                    # if current_ini == former_ini:
                        # former_ent: (20, 31, 'trastuzumab', 'Chemical')
                        # entity: (20, 48, 'trastuzumab and capecitabine', 'Chemical')
                        # former_ent: (20, 48, 'trastuzumab and capecitabine', 'Chemical')
                        # entity: (36, 48, 'capecitabine', 'Chemical')
                        # sorted_ents.remove(former_ent)
                    # elif current_end < former_end:
                        # former_ent: (102, 153, 'peginterferon ( PEG-IFN ) and ribavirin combination', 'Chemical')
                        # entity: (132, 141, 'ribavirin', 'Chemical')
                        # sorted_ents.remove(entity)
        former_ent = entity


def split_with_span_index(sentence: str):
    words = sentence.split()
    search_start = 0
    out = []
    for word in words:
        start = sentence.find(word, search_start)
        end = start + len(word)
        out.append((word, start, end))
        search_start = end
    return out

def add_context_words(sorted_ents, sent: str):
    """
    special cases:
        entity_name: ATRA
        sent: (2) After treatment with ATRA, the fusion protein disappeared and PML protein resumed in NB4 cells, while in HL-60 and K562 cells there was no difference from control cells.

        induced CYP1A1-dependent
        MMP2/TIMP2 mRNA ratio
        that PI-3K/Akt-mediated cyclin

        entity: (13, 29, 'Arsenic trioxide', 'Chemical')
        sent: CONCLUSIONS: Arsenic trioxide-induced renal
    """
    extended_ents = []
    all_words = split_with_span_index(sent)
    for entity in sorted_ents:
        entity_ini, entity_end, entity_name, entity_type = entity
        leading_word = None
        trailing_word = None
        for word_i, (word, start, end) in enumerate(all_words):
            # induced CYP1A1-dependent, CYP1A1 is gene
            if (start in (entity_ini, entity_ini-1)):
                if word_i > 0:
                    leading_word = all_words[word_i-1][0]
                if entity_end + 1 < end:
                    trailing_word = punc_at_ends.sub('', word[len(entity_name):])
            # MMP2/TIMP2 mRNA ratio
            # with ATRA, the fusion
            if not trailing_word and (end in (entity_end, entity_end+1, entity_end+2)):
                if word_i < len(all_words)-1:
                    trailing_word = all_words[word_i+1][0]
                if not leading_word and start + 1 < entity_ini:
                    leading_word = punc_at_ends.sub('', word[:entity_ini-start])
            # that PI-3K/Akt-mediated cyclin， Akt at the middle
            if start + 1 < entity_ini and entity_end + 1 < end:
                if not leading_word:
                    leading_word = punc_at_ends.sub('', word[:entity_ini-start])
                if not trailing_word:
                    ent_end_index_in_word = word.index(entity_name) + len(entity_name)
                    trailing_word = punc_at_ends.sub('', word[ent_end_index_in_word:])
            # entity: (13, 29, 'Arsenic trioxide', 'Chemical')
            # sent: CONCLUSIONS: Arsenic trioxide-induced renal
            if entity_end + 1 < end and not trailing_word and start < entity_end:
                trailing_start_i = entity_end - start
                trailing_word = punc_at_ends.sub('', word[trailing_start_i:])
            # entity: (52, 77, 'cyclin-dependent kinase 4', 'Gene')
            # sent: decreased cyclin-dependent kinase 4 kinase
            if not trailing_word and start in (entity_end + 1, entity_end + 2):
                trailing_word = word
            # sent: cyclin D1/cyclin-dependent kinase 4 complexes
            if not leading_word and start + 1 < entity_ini and entity_ini < end:
                leading_end_i = len(word) - (end - entity_ini)
                leading_word = punc_at_ends.sub('', word[:leading_end_i])

        # the entity_ini at the start of sentence, or the entity_end at the end of sentence.
        if ((not leading_word and entity_ini <= 1)
            or (not trailing_word and (entity_end + 2 >= len(sent)))
        ):
            pass
        elif not leading_word or not trailing_word:
            ic(entity, sent)

        if not leading_word:
            leading_word = 'At the start of sentence'
        if not trailing_word:
            trailing_word = 'At the end of sentence'
        extended_ents.append((leading_word, entity_name, trailing_word, entity_type))
    return extended_ents

def merge_ner(x):
    sent = x.sentence.tolist()[0]
    lst_ents = set()
    for ini, end, entity_name, entity_type in zip(x.entity_ini.tolist(), x.entity_end.tolist(), x.entity_name.tolist(), x.entity_type.tolist()):
        if pd.isna(ini):
            continue
        entity_in_sent = sent[ini:end]
        if entity_in_sent != entity_name:
            # ic(entity_in_sent, entity_name, ini, end, sent)
            if sent[ini: ini+len(entity_name)] == entity_name:
                end = ini+len(entity_name)
                lst_ents.add((ini, end, entity_name, entity_type))
        else:
            lst_ents.add((ini, end, entity_name, entity_type))
    sorted_ents = sorted(lst_ents, key=lambda x: (x[0], x[1]))

    # check_indexes(sorted_ents)
    sorted_ents = add_context_words(sorted_ents, sent)
    return pd.Series([sorted_ents])


df_data = read_chem_x_gene()
df_data_ner = df_data.groupby(by=["sentence"]).apply(lambda x: merge_ner(x)).reset_index().rename(columns={0: "entity_info"})
ic(df_data_ner.columns.to_list())
df_data_ner = df_data_ner.merge(df_data[["uid", "sentence"]].drop_duplicates(), on=["sentence"])
df_data_ner.drop_duplicates(subset=["sentence"])
ic(len(df_data_ner))
merged_ner_file = 'PGx_CTD_chem_x_gene_merged_ner.csv'
df_data_ner.to_csv(merged_ner_file, index=False, sep=',')