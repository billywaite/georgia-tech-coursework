
import re
def normalize_string(s):
    assert type(s) is str
    s = s.lower()
    s= re.sub("[^a-zA-Z\s\s*]", "", s)
    return s

def get_normalized_words(s):
    assert type (s) is str
    # Keep whitespace & alpha chars
    s = normalize_string(s)
    # Split into list of words
    words = s.split()
    return(words)

#########
# Exercise 3
#########

def make_itemsets(words):
    itemsets = []
    [itemsets.append(set(x)) for x in words]
    return(itemsets)

# Exercise 4
from collections import defaultdict
from itertools import combinations # Hint!

def update_pair_counts (pair_counts, itemset):
    """
    Updates a dictionary of pair counts for
    all pairs of items in a given itemset.
    """
    # Create list of letter combinations from itemset
    for a, b in combinations(itemset, 2):
        pair_counts[a, b] += 1
        pair_counts[b, a] += 1

##############
#  Exercise 5
############
        
def update_item_counts(item_counts, itemset):
    #
    for x in itemset:
        item_counts[x] += 1
        
##################
# Exerciser 6
#################

def filter_rules_by_conf (pair_counts, item_counts, threshold):
    rules = {} # (item_a, item_b) -> conf (item_a => item_b)
            
    for key in pair_counts:
        
        # Sum occurences of each item in item count, pulled from the pair
        ic = 0
        for i in item_counts:
            if i in key:
                ic += item_counts[i]
        
        # Calculate and test confidence before pushing to rules dictionary
        confidence = float(pair_counts[key]) / ic
        if confidence >= threshold:
            rules[key] = pair_counts[key]
    
    return(rules)



##################
# Exerciser 7
#################


def update_pair_counts(pair_counts, itemset):

    # Create list of letter combinations from r
    for a, b in combinations(itemset, 2):
        pair_counts[a, b] += 1
        pair_counts[b, a] += 1

def update_item_counts(item_counts, itemset):
    for i in itemset:
        item_counts[i] += 1


def filter_rules_by_conf (pair_counts, item_counts, threshold):
    rules = {} # (item_a, item_b) -> conf (item_a => item_b)
            
    for (a, b) in pair_counts:
        # Calculate and test confidence before pushing to rules dictionary
        confidence = pair_counts[(a, b)] / item_counts[a]
        if confidence >= threshold:
            rules[(a, b)] = confidence
    
    return rules


def find_assoc_rules(receipts, threshold):
    
    # Create default dictionary for counts
    pair_counts = defaultdict(int)
    item_counts = defaultdict(int)
        
    for r in receipts:
        
        # Update pair counts
        update_pair_counts(pair_counts, r)
        
        # Update item counts
        update_item_counts(item_counts, r)
        
    # Test confidence
    rules = filter_rules_by_conf(pair_counts, item_counts, threshold)
            
    return(rules)

receipts = [set('abbc'), set('ac'), set('a')]
rules = find_assoc_rules(receipts, 0.6)


##################
# Exerciser 8
#################

# Normalize latin text
latin_text = get_normalized_words(latin_text)
latin_text = make_itemsets(latin_text)
latin_rules = find_assoc_rules(latin_text, 0.75)

##################
# Exerciser 9
#################

def intersect_keys(d1, d2):
    assert type(d1) is dict or type(d1) is defaultdict
    assert type(d2) is dict or type(d2) is defaultdict
    
    # Dictionary compehension to locate key intersection in dictionaries
    intersections = {key:value for key,value in d1.items() if key in d2}
    return(intersections)

# For testing purposes
P = dict(zip('a b c d'.split(),[1,2,3,4]))
E = dict(zip('a b e f'.split(),[6,7,8,9]))
print(intersect_keys(P, E))

##################
# Exerciser 10
#################
# Create english rules. Latin rules created prior to this block
english_text = get_normalized_words(english_text)
english_text = make_itemsets(english_text)
english_rules = find_assoc_rules(english_text, 0.75)

# Execute function created in exercise 9, which already filters by high confidence level
common_high_conf_rules = intersect_keys(latin_rules, english_rules)

print("High-confidence rules common to _lorem ipsum_ in Latin and English:")
print_rules(common_high_conf_rules)

##################
# Exerciser 11
#################


# Code below gets data
def on_vocareum():
    import os
    return os.path.exists('.voc')

def download(file, local_dir="", url_base=None, checksum=None):
    import os, requests, hashlib, io
    local_file = "{}{}".format(local_dir, file)
    if not os.path.exists(local_file):
        if url_base is None:
            url_base = "https://cse6040.gatech.edu/datasets/"
        url = "{}{}".format(url_base, file)
        print("Downloading: {} ...".format(url))
        r = requests.get(url)
        with open(local_file, 'wb') as f:
            f.write(r.content)            
    if checksum is not None:
        with io.open(local_file, 'rb') as f:
            body = f.read()
            body_checksum = hashlib.md5(body).hexdigest()
            assert body_checksum == checksum, \
                "Downloaded file '{}' has incorrect checksum: '{}' instead of '{}'".format(local_file,
                                                                                           body_checksum,
                                                                                           checksum)
    print("'{}' is ready!".format(file))
    
if on_vocareum():
    DATA_PATH = "../resource/asnlib/publicdata/"
else:
    DATA_PATH = ""
datasets = {'groceries.csv': '0a3d21c692be5c8ce55c93e59543dcbe'}

for filename, checksum in datasets.items():
    download(filename, local_dir=DATA_PATH, checksum=checksum)

with open('{}{}'.format(DATA_PATH, 'groceries.csv')) as fp:
    groceries_file = fp.read()
print (groceries_file[0:250] + "...\n... (etc.) ...") # Prints the first 250 characters only
print("\n(All data appears to be ready.)")
############################################
# Setup
# Confidence threshold
THRESHOLD = 0.5

# Only consider rules for items appearing at least `MIN_COUNT` times.
MIN_COUNT = 10
##################33
# df: groceries_file

# Split groceries file by new line
s = groceries_file.split('/n')

# Create sets of receipts
receipts = [list(set(receipt.split(','))) for receipt in s]

# Find confidence rules above confidence threshold
high_conf_groceries = find_assoc_rules(receipts, THRESHOLD)

# Filter by min_count
High_freq_item = list([elem for elem in receipts if receipts.count(elem) >= MIN_COUNT])




