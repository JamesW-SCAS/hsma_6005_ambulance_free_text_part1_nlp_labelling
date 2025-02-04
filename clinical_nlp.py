# import pandas, spacy and the downloaded en_core_web_sm pre-trained model
import pandas as pd
import spacy
import en_core_web_sm
from spacy import displacy

# import the spacy pattern matcher@
from spacy.matcher import Matcher

# Load the pre-trained model as a language model into a variable called nlp
nlp = en_core_web_sm.load()

# Read the clinical data into a dataframe, selecting only certain columns
# NEXT - Look for NEGATION, e.g. "12 lead not done"
# ALSO LOOK IN THE ZOLL FIELD OF EPR FOR 12 LEAD
# Compare the matched span column with "12 lead taken" boolean field
    # Look in BI009062 Supporting the review of Clinical Records DATA MASTER v0.2 - 20241003
        # see tab "OHCAO data set", column "12 lead taken"
            # Test for rows where we have added value
# NEXT RULE to try - "Oxygen administered", or e.g. "o2 given"

filtered_cols = [
    'CareepisodeID',
    'impressionPlan',
    'injuryIllnessDetails'
    ]
df = pd.read_csv("nlp_input.csv"
, usecols = filtered_cols
, nrows=20
, encoding_errors='ignore'
)
# Drop any rows with null data
# df.dropna(
#     subset='impressionPlan',
#  inplace=True)

# # TEST ROW FOR 12 LEAD ECG - Remove later
# df = df[3:4] # Row contains a great positive/negative 12 lead example 
# # force in some text to check how Spacy handles various entries
# df.iloc[0,1] = "12      lead"

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
        print(f"Row {index}, Matched span: {span.text}")
    
    # Join all matches for this row into a single string
    df.at[index, 'matched_spans'] = '; '.join(row_matches)

# Print summary
print(f"\nTotal rows with matches: {df['matched_spans'].astype(bool).sum()}")
print(f"Total rows in dataframe: {len(df)}")

# Create results df for rows with non-blank matches
results = df.loc[df['matched_spans'].notna() & 
    (df['matched_spans'].str.strip() != ''), :]
results