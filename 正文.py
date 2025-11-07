import pandas as pd
import json
import re
import csv
import unicodedata
from opencc import OpenCC
from collections import Counter

cc = OpenCC('t2s')

def txtTocsv(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", newline="", encoding="utf-8") as outfile:
        
        writer = csv.writer(outfile)
        writer.writerow(["codepoint", "field", "read" if output_file == "Readings.csv" else "write"])
        
        for line in infile:
            line = line.strip()
            print(line)
            if not line or line.startswith("#"):
                continue  # skip comments and blank lines
            
            parts = line.split("\t")
            if len(parts) == 3 and (parts[1] == "kPhonetic" or parts[1] == "kMandarin"):
                writer.writerow(parts)

    print(f"{input_file} → {output_file}")

def strip(s):
    s = s.translate(str.maketrans("ǖǘǚǜ", "vvvv"))
    normalized = unicodedata.normalize('NFD', s)
    return ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')

# Unihan TXT to CSV
txtTocsv("Unihan_Readings.txt", "Readings.csv")
txtTocsv("Unihan_DictionaryLikeData.txt", "Radicals.csv")

# DataFrame Reading
pinyin_df = pd.read_csv("Readings.csv")
radical_df = pd.read_csv("Radicals.csv")

# Format pinyin_df
pinyin_df["read"] = pinyin_df["read"].apply(strip)
pinyin_df["read"] = pinyin_df["read"].astype(str).str.replace("v", "ü")
# Duplicate readings
pinyin_df["read"] = pinyin_df["read"].astype(str).str.strip()
pinyin_df = (pinyin_df.assign(read=pinyin_df["read"].str.split(r"[\s,]+")).explode("read").reset_index(drop=True))

# Format radical_df
radical_df["write"] = radical_df["write"].astype(str).str.replace("*", "", regex=False)
radical_df["write"] = radical_df["write"].astype(str).apply(lambda s: " ".join([w for w in s.split() if 'x' not in w]))
radical_df = radical_df.reset_index(drop=True)

# Make the phonetic dictionary
radical_df['char'] = radical_df['codepoint'].apply(lambda x: chr(int(x[2:], 16))).apply(cc.convert)
phonetic_dict = radical_df.groupby('write')['char'].apply(list).to_dict()

# Merge and save to 
df = pd.merge(pinyin_df, radical_df, on="codepoint")

stats = (df.groupby('read')['write'].agg(lambda x: x.value_counts().idxmax()).reset_index())

def mapper(ks):
    chars = []
    for key in str(ks).split():
        key = key.strip()
        if key in phonetic_dict:
            chars.extend(phonetic_dict[key])
    return " ".join(chars)

stats["map"] = stats.iloc[:, 1].apply(mapper)

stats.to_csv("Stats.csv", index=False)

# Above only takes most common by total number of characters, below scales by frequency

freq_df = pd.read_csv("SUBTLEX-CH.csv", sep=",")
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
    keys = str(row["write"]).split()
    info = []

    for k in keys:
        if k not in phonetic_dict:
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
stats.to_csv("Stats.csv", index=False)

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
        "1st_same": same,
        "1st_different": diff,
        "1st_same_len": same_len,
        "1st_different_len": diff_len,
        "1st_congruency": first_congruency,
        "most_congruency_write": best_key,
        "most_congruency_same": best_same,
        "most_congruency_different": best_diff,
        "most_congruency_same_len": len(best_same),
        "most_congruency_different_len": len(best_diff),
        "most_congruency_congruency": best_congruency
    })

summary_df = stats["correspondences"].apply(extract_summary)
stats = pd.concat([stats, summary_df], axis=1)

stats.to_csv("Stats.csv", index=False)
