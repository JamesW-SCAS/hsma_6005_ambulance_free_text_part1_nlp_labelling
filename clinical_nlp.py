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
# NEXT - TRY WITH MORE THAN ONE FREE TEXT COLUMN
# ALSO LOOK IN THE ZOLL FIELD OF EPR FOR 12 LEAD
filtered_cols = [
    'CareepisodeID',
    'impressionPlan'
    ]
df = pd.read_csv("nlp_input.csv"
, usecols = filtered_cols
, nrows=100
)
# Drop any rows with null data
df.dropna(
    subset='impressionPlan',
 inplace=True)

# # TEST ROW FOR 12 LEAD ECG - Remove later
# df = df[3:4] # Row contains a great positive/negative 12 lead example 
# # force in some text to check how Spacy handles various entries
# df.iloc[0,1] = "patient had a twelve_lead, , TWELVE LEAD, a 12 lead,\
#  a 12-lead and a 3 lead"

# Initialize the Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)
# Create a pattern to match 12 lead, including text and punctuation:
twelve_lead_pattern_1 = [{"ORTH" : "12"}, {"IS_PUNCT": True, "OP": "?"},
 {"LOWER" : "lead"}]
twelve_lead_pattern_2 = [{"LOWER": "twelve"}, {"IS_PUNCT": True, "OP": "?"},
 {"LOWER": "lead"}]
# Add rules to account for underscores (which Spacy uses to split tokens)
# THESE DON'T SEEM TO WORK!
twelve_lead_pattern_3 = [{"LOWER": "12"}, {"ORTH": "_"}, {"LOWER": "lead"}]
twelve_lead_pattern_4 = [{"LOWER": "twelve"}, {"ORTH": "_"}, {"LOWER": "lead"}]
# Add the pattern(s) to the Matcher
matcher.add("lead_pattern", [twelve_lead_pattern_1, twelve_lead_pattern_2,
twelve_lead_pattern_3, twelve_lead_pattern_4])

# Apply the matcher to each row of the dataframe:
for index, row in df.iterrows():
    doc = nlp(row['impressionPlan'])
    matches = matcher(doc)
    # Print the matches
    for match_id, start, end in matches:
        span = doc[start:end]
        print(f"Matched span: {span.text}")