# import pandas, spacy and the downloaded en_core_web_sm pre-trained model
import pandas as pd
import spacy
import en_core_web_sm
from spacy import displacy

# import the negex negation algortihm from the negspacy library
from negspacy.negation import Negex
# Negation (neg) - import Span library
from spacy.tokens import Span

# import the spacy pattern matcher@
from spacy.matcher import Matcher

# Load the pre-trained model as a language model into a variable called nlp
nlp = en_core_web_sm.load()

# Add the negation detection pipeline ("negex") to the spaCy NLP object. 
# The config parameter specifies which entity types to consider for negation 
# detection, in this case, "PERSON" and "ORG" (organization).

# Take Negex out for now to see if it speeds up processing
# nlp.add_pipe("negex")#, config={"ent_types":["PERSON","ORG"]})

# 4/2/25 - NEED TO MAKE A CUSTOM ENTITY TYPE FOR MY MATCHED SPANS TO POINT NEGEX AT
# NEXT RULE to try - "Oxygen administered", or e.g. "o2 given"

filtered_cols = [
    'CareEpisodeID',  # Added a capital E in Episode to match input file column 28/2/25
    'impressionPlan',
    'injuryIllnessDetails'
    # 'Oxygen Administered' # This is the button-press field; need to add to input csv
    ]
df = pd.read_csv("Oxygen and 12 Lead Free Text Training Model v0.1 - 20250227.csv" #"nlp_input.csv"
, usecols = filtered_cols
, nrows = 100
, encoding_errors='ignore'
)

# TEST ROW - Remove later
# df = df[3:4] # Row contains a great positive/negative 12 lead example 
# force in some text to check how Spacy handles various entries
# df.iloc[0,1] = "I'm looking for no gave oxygen, was taken, and no 12 lead here."
# df.iloc[0,2] = "...and air taken here"

# Initialize the Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)

# Create a pattern to match 12 lead, including text and punctuation:
twelve_lead_pattern_1 = [{"ORTH" : "12"}, {"IS_PUNCT": True, "OP": "?"},
{"LOWER" : "lead"}]
twelve_lead_pattern_2 = [{"LOWER": "twelve"}, {"IS_PUNCT": True, "OP": "?"},
{"LOWER": "lead"}]

# Oxygen pattern match rule
# ISSUE - too many possible patterns, e.g. these all miss "oxygen was given"
# CROSS-CHECK WTIH SUE'S LABELLED DATA AND CREW BUTTON-PRESS COLUMNS
o2_pattern_1 = [{"LOWER": "oxygen"}, {"LOWER" : "admin"}]
o2_pattern_2 = [{"LEMMA" : "oxygen"}, {"POS" : "VERB"}]
o2_pattern_3 = [{"POS" : "VERB"}, {"LEMMA" : "oxygen"}]
o2_pattern_4 = [{"LOWER": "o2"}, {"LOWER" : "admin"}]
o2_pattern_5 = [{"LEMMA" : "o2"}, {"POS" : "VERB"}]
o2_pattern_6 = [{"POS" : "VERB"}, {"LEMMA" : "o2"}]
o2_pattern_7 = [{"LOWER": "oxy"}, {"LOWER" : "admin"}]
o2_pattern_8 = [{"LEMMA" : "oxy"}, {"POS" : "VERB"}]
o2_pattern_9 = [{"POS" : "VERB"}, {"LEMMA" : "oxy"}]

# Add the pattern(s) to the Matcher
# MAKE SEPARATE PATTERNS TO LABEL "TWELVE LEAD", "OXYGEN", ETC FOR NEURAL NET
matcher.add("12_lead_ecg", \
    [
    twelve_lead_pattern_1, 
    twelve_lead_pattern_2
    ])

matcher.add("oxygen",\
    [
    o2_pattern_1, 
    o2_pattern_2, 
    o2_pattern_3,
    o2_pattern_4,
    o2_pattern_5,
    o2_pattern_6,
    o2_pattern_7,
    o2_pattern_8,
    o2_pattern_9
    ])

# Create a new column to store all matches
df['matched_spans'] = ''

# Apply the matcher to each row of the dataframe:
for index, row in df.iterrows():
    # Combine text in multiple columsn into a single string and pass to nlp
    doc = nlp(str(row['impressionPlan'])
    + ' '
    + str(row['injuryIllnessDetails'])
    )
    matches = matcher(doc)
    
    # List to store all matches for this row
    row_matches = []
    
    # Process all matches
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]
        row_matches.append(span.text)
        # print(f"Row {index}, Matched span: {span.text}")
        print(f"Row {index}, Matched span: {span.text}, String_ID : {string_id}") # , Negation: {span._.negex}")
    
    # Join all matches for this row into a single string
    df.at[index, 'matched_spans'] = '; '.join(row_matches)

# Print summary
print(f"\nTotal rows with matches: {df['matched_spans'].astype(bool).sum()}")
print(f"Total rows in dataframe: {len(df)}")

# Create results df for rows with non-blank matches
results = df.loc[df['matched_spans'].notna() & 
    (df['matched_spans'].str.strip() != ''), :]
results