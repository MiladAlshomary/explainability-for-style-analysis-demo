import csv
import re

def load_original_map_and_extract_morph(path="human_readable.txt"):
    human_to_code = {}
    morph_entries = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line or line.startswith("#"):
                continue
            key, val = [p.strip() for p in line.split(":", 1)]

            # If key looks like Aspect=Perf, it's a morphological tag
            if "=" in key:
                morph_entries.append((val, key))  # human:code
            else:
                human_to_code[val] = key  # human:code for POS/etc.

    return human_to_code, morph_entries

def extract_bigrams_from_csv(csv_path="../datasets/gram2vec_feats.csv"):
    bigrams = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat = row["gram2vec_feats"]
            if feat.startswith("Part-of-Speech Bigram:"):
                human_bigram = feat.split(":", 1)[1].strip()
                if "followed by" in human_bigram:
                    bigrams.add(human_bigram)
    return bigrams

def generate_bigram_code_map(human_to_code, bigrams):
    pattern = re.compile(r"(.+?) followed by (.+)")
    code_map = {}

    for bigram in bigrams:
        match = pattern.match(bigram)
        if match:
            x = match.group(1).strip()
            y = match.group(2).strip()
            code_x = human_to_code.get(x)
            code_y = human_to_code.get(y)
            if code_x and code_y:
                code_map[bigram] = f"{code_x} {code_y}"
            else:
                print(f"Could not map: {bigram} → {code_x}, {code_y}")
        else:
            print(f"Not matched: {bigram}")
    return code_map

def write_augmented_map(pos_bigram_map, morph_entries, original_path="human_readable.txt", output_path="augmented_human_readable.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        # Flip original lines: write human-readable:code instead of code:human
        with open(original_path, "r", encoding="utf-8") as orig:
            for line in orig:
                line = line.strip()
                if not line or line.startswith("#"):
                    f.write(line + "\n")
                    continue
                if ":" not in line:
                    continue
                key, val = [p.strip() for p in line.split(":", 1)]
                flipped_line = f"{val}:{key}\n"
                f.write(flipped_line)


        # Add new section for POS bigrams
        f.write("\n")
        for human, code in sorted(pos_bigram_map.items()):
            f.write(f"{human}:{code}\n")

        # Re-add morph tag mappings
        f.write("\n")
        for human, code in sorted(morph_entries):
            f.write(f"{human}:{code}\n")

    print(f"Augmented map written to {output_path}")

# Run all
human_to_code, morph_entries = load_original_map_and_extract_morph()
bigrams = extract_bigrams_from_csv()
pos_bigram_map = generate_bigram_code_map(human_to_code, bigrams)
write_augmented_map(pos_bigram_map, morph_entries)
