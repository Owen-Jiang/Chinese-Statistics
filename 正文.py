import pandas as pd
import json
import re
import csv
import unicodedata
from collections import Counter

import typing

## Output files: don't change these
RESULT_CSV = "result/Stats.csv"

## Based on the Unihan data file format specs (you probably don't need to change these)
DATAFILE_PATH = "data/Unihan_%s.txt"
DATAFILE_DELIM = '\t'
DATAFILE_COMMENT = '#'
DATAFILE_BASENAMES = ['codepoint', 'prop', 'val'] # The values will get renamed according to PROPS_NAMES
DATA_DIR = "data/%s"

# Unihan data files we need (and what properties they provide of interest to us)
SOURCES = {
    "DictionaryLikeData": ["kPhonetic"],
    "Readings": ["kMandarin"],
    "Variants": ["kSimplifiedVariant"],
}
# What to rename each of the properties desired above [OPTIONAL]
PROPS_NAMES = {
    "kMandarin": "read",
    "kPhonetic": "write",
    "kSimplifiedVariant": "simplified",
}

# Will get populated by loadData()
DATAFRAMES = {}

def loadData():
    global DATAFRAMES

    for source in SOURCES:
        sourceDF = pd.read_csv(DATAFILE_PATH % source, delimiter=DATAFILE_DELIM, comment=DATAFILE_COMMENT,
                               header=None, names=DATAFILE_BASENAMES)
        for prop in SOURCES[source]:
            print(f"[*]\t {prop} [{source}]")
            DATAFRAMES[prop] = sourceDF[sourceDF['prop'] == prop] \
                .drop('prop', axis=1) \
                .rename(columns={'val': PROPS_NAMES.get(prop, prop)}) \
                .reset_index(drop=True)

## Stage 1: Data loading
print("[*] Loading data:")

# DataFrame Reading
loadData()
pinyin_df = DATAFRAMES['kMandarin']
radical_df = DATAFRAMES['kPhonetic']

t2s_table = DATAFRAMES['kSimplifiedVariant']
def t2s_convert(codepoint: str):
    conversion = t2s_table[t2s_table['codepoint'] == codepoint]
    if not len(conversion):
        return codepoint
    return conversion.iloc[0]['simplified'].split(' ')[0] # If there are multiple possible conversions, return the first
    # (most likely) one

def strip(s):
    s = s.translate(str.maketrans("ǖǘǚǜ", "vvvv"))
    normalized = unicodedata.normalize('NFD', s)
    return ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')

# Format pinyin_df
pinyin_df["read"] = pinyin_df["read"].apply(strip)
pinyin_df["read"] = pinyin_df["read"].astype(str).str.replace("v", "ü")
# Duplicate readings
pinyin_df["read"] = pinyin_df["read"].astype(str).str.strip()
pinyin_df = (pinyin_df.assign(read=pinyin_df["read"].str.split(r"[\s,]+")).explode("read").reset_index(drop=True))

# Format radical_df
radical_df["write"] = radical_df["write"].astype(str).str.replace("*", "", regex=False)
radical_df["write"] = radical_df["write"].astype(str).apply(lambda s: [w for w in s.split() if 'x' not in w]) # each radical
# may belong to multiple phonetic groups (e.g. it contains one phonetic component which is a subcomponent of another in it).
# we want to keep these group IDs as lists, so we can group each character multiple times below.
radical_df = radical_df.reset_index(drop=True)

# Make the phonetic dictionary
radical_df['char'] = radical_df['codepoint'].apply(t2s_convert).apply(lambda x: chr(int(x[2:], 16)))
phonetic_dict = radical_df.explode('write').groupby('write')['char'].apply(list).to_dict()

print("[+] Data loaded!")

# Merge and save to 
df = pd.merge(pinyin_df, radical_df, on="codepoint")

## Stage 2: Analysis

stats = (df.groupby('read')['write'].agg(lambda x: x.value_counts().idxmax()).reset_index())

print("[*] Analyzing...")

def mapper(ks):
    chars_grps = []
    for key in ks:
        key = key.strip()
        if key in phonetic_dict:
            chars_grps.append(phonetic_dict[key])
    return " | ".join([" ".join(chars) for chars in chars_grps])

stats["map"] = stats.iloc[:, 1].apply(mapper)

stats.to_csv(RESULT_CSV, index=False)

# Above only takes most common by total number of characters, below scales by frequency

freq_df = pd.read_csv(DATA_DIR % "SUBTLEX-CH.csv", sep=",")
freq_df["Word"] = freq_df["Word"].astype(str)
freq_df["WCount"] = pd.to_numeric(freq_df["WCount"], errors="coerce").fillna(0)

# Split multi-character words and divide frequency equally
char_freq = {}
for _, row in freq_df.iterrows():
    word = row["Word"].strip()
    freq = row["WCount"]
    if not word:
        continue
    share = freq / len(word)
    for ch in word:
        char_freq[ch] = char_freq.get(ch, 0) + share

def phonetic_sorter(row):
    read = row["read"]
    keys = row["write"]
    info = []

    for k in keys:
        if k not in phonetic_dict:
            print("[WARN]", k, "not in phonetic_dict!")
            continue

        chars = list(dict.fromkeys(phonetic_dict[k]))  # deduplicate
        same, different = [], []

        for ch in chars:
            freq = char_freq.get(ch, 0.0)
            # Find Mandarin readings for this character
            reads = pinyin_df.loc[pinyin_df["codepoint"] == f"U+{ord(ch):04X}", "read"].tolist()
            if read in reads:
                same.append((ch, freq))
            else:
                different.append((ch, freq))

        # Sort by descending frequency
        same.sort(key=lambda x: x[1], reverse=True)
        different.sort(key=lambda x: x[1], reverse=True)

        info.append({
            k: {
                "same": [c for c, _ in same],
                "different": [c for c, _ in different]
            }
        })

    return info
print("[*]\t Phonetically sorting...")
stats["correspondences"] = stats.apply(phonetic_sorter, axis=1)

# Drop the temporary frequency key
for dlist in stats["correspondences"]:
    for d in dlist:
        for v in d.values():
            v.pop("freq", None)

# Duplication removal
def dedup_string(s):
    """Remove duplicate characters from a string while preserving order."""
    seen = set()
    return ''.join(ch for ch in s if not (ch in seen or seen.add(ch)))

def dedup_phonetic_info(info_list):
    """Remove duplicate characters in 'same' and 'different' lists within the phonetic_info structure."""
    if not isinstance(info_list, list):
        return info_list
    cleaned = []
    for entry in info_list:
        if not isinstance(entry, dict):
            continue
        key = list(entry.keys())[0]
        val = entry[key]
        same = val.get("same", [])
        diff = val.get("different", [])
        # Preserve order
        val["same"] = list(dict.fromkeys(same))
        val["different"] = list(dict.fromkeys(diff))
        cleaned.append({key: val})
    return cleaned

# Apply deduplication
stats["map"] = stats["map"].astype(str).apply(dedup_string)
stats["correspondences"] = stats["correspondences"].apply(dedup_phonetic_info)

# Save final result
stats.to_csv(RESULT_CSV, index=False)

# Summarize below

def extract_summary(info_list):
    if not info_list or not isinstance(info_list, list):
        return pd.Series({
            "1st_write": None,
            "1st_same": [],
            "1st_different": [],
            "1st_same_len": 0,
            "1st_different_len": 0,
            "1st_congruency": 0,
            "most_congruency_write": None,
            "most_congruency_same": [],
            "most_congruency_different": [],
            "most_congruency_same_len": 0,
            "most_congruency_different_len": 0,
            "most_congruency_congruency": 0
        })

    # --- First (most common) phonetic index ---
    first_entry = info_list[0]
    first_key = list(first_entry.keys())[0]
    first_val = first_entry[first_key]
    same = first_val.get("same", [])
    diff = first_val.get("different", [])
    same_len = len(same)
    diff_len = len(diff)
    first_congruency = same_len / (same_len + diff_len) if (same_len + diff_len) > 0 else 0

    # --- Most congruent phonetic index ---
    best_key = None
    best_val = None
    best_congruency = -1
    for entry in info_list:
        k = list(entry.keys())[0]
        v = entry[k]
        s_len = len(v.get("same", []))
        d_len = len(v.get("different", []))
        c = s_len / (s_len + d_len) if (s_len + d_len) > 0 else 0
        if c > best_congruency:
            best_congruency = c
            best_key = k
            best_val = v

    best_same = best_val.get("same", []) if best_val else []
    best_diff = best_val.get("different", []) if best_val else []

    return pd.Series({
        "1st_write": first_key,
        "1st_same": ' '.join(same),
        "1st_different": ' '.join(diff),
        "1st_same_len": same_len,
        "1st_different_len": diff_len,
        "1st_congruency": first_congruency,
        "most_congruency_write": best_key,
        "most_congruency_same": ' '.join(best_same),
        "most_congruency_different": ' '.join(best_diff),
        "most_congruency_same_len": len(best_same),
        "most_congruency_different_len": len(best_diff),
        "most_congruency_congruency": best_congruency
    })

print("[*]\t Extracting summary...")
summary_df = stats["correspondences"].apply(extract_summary)
stats = pd.concat([stats, summary_df], axis=1)

stats.to_csv(RESULT_CSV, index=False)

print("[*] Analysis complete; saved to:", RESULT_CSV)
