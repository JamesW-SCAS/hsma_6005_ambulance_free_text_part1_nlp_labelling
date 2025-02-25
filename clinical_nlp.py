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
nlp.add_pipe("negex")#, config={"ent_types":["PERSON","ORG"]})

# 4/2/25 - NEED TO MAKE A CUSTOM ENTITY TYPE FOR MY MATCHED SPANS TO POINT NEGEX AT
# NEXT RULE to try - "Oxygen administered", or e.g. "o2 given"

filtered_cols = [
    'CareepisodeID',
    'impressionPlan',
    'injuryIllnessDetails'
    ]
df = pd.read_csv("nlp_input.csv"
, usecols = filtered_cols
, nrows=100
, encoding_errors='ignore'
)

# Initialize the Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)
# Create a pattern to match 12 lead, including text and punctuation:
twelve_lead_pattern_1 = [{"ORTH" : "12"}, {"IS_PUNCT": True, "OP": "?"},
{"LOWER" : "lead"}]
twelve_lead_pattern_2 = [{"LOWER": "twelve"}, {"IS_PUNCT": True, "OP": "?"},
{"LOWER": "lead"}]

# Add the pattern(s) to the Matcher
matcher.add("lead_pattern", [twelve_lead_pattern_1, twelve_lead_pattern_2])

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
        span = doc[start:end]
        row_matches.append(span.text)
        # print(f"Row {index}, Matched span: {span.text}")
        print(f"Row {index}, Matched span: {span.text}, Negation: {span._.negex}")
    
    # Join all matches for this row into a single string
    df.at[index, 'matched_spans'] = '; '.join(row_matches)

# Print summary
print(f"\nTotal rows with matches: {df['matched_spans'].astype(bool).sum()}")
print(f"Total rows in dataframe: {len(df)}")

# Create results df for rows with non-blank matches
results = df.loc[df['matched_spans'].notna() & 
    (df['matched_spans'].str.strip() != ''), :]
results