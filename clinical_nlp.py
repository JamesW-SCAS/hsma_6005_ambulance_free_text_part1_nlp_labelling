# import pandas, spacy and the downloaded en_core_web_sm pre-trained model
import pandas as pd
import spacy
import en_core_web_sm
from negspacy.negation import Negex
from spacy.tokens import Span
from spacy.matcher import Matcher

# Load the pre-trained model as a language model into a variable called nlp
nlp = en_core_web_sm.load()

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
df = df[3:4] # Row contains a great positive/negative 12 lead example 
# force in some text to check how Spacy handles various entries
df.iloc[0,1] = "12 lead ecg was done but no oxygen given" #"I'm looking for no oxygen not given, was taken, and no 12 lead here. More oxygen given. And another twelve lead"
# df.iloc[0,2] = "...and air taken here"

# Initialize the Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)

# Create a pattern to match 12 lead, including text and punctuation:
twelve_lead_pattern_1 = [{"ORTH" : "12"}, {"IS_PUNCT": True, "OP": "?"},
{"LOWER" : "lead"}]
twelve_lead_pattern_2 = [{"LOWER": "twelve"}, {"IS_PUNCT": True, "OP": "?"},
{"LOWER": "lead"}]

# Oxygen pattern match rule
o2_pattern_1 = [{"LOWER": "oxygen"}] 

# Add the pattern(s) to the Matcher
matcher.add("12_lead_ecg_label", \
    [
    twelve_lead_pattern_1, 
    twelve_lead_pattern_2
    ])
matcher.add("oxygen_label",\
    [
    o2_pattern_1, 
    ])

# # Create a new column to store all matches
# df['matched_spans'] = ''
# # Create new column to store all labels of matched spans
# df['train_labels'] = ''

# Add the custom language component here:
@spacy.Language.component("custom_ents_component")
def custom_ents_component(doc):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label=match_id)
     for match_id, start, end in matches]
    # print(spans)
    doc.ents = spans
    return doc

# Try adding components to the pipeline in order, as per Sammi's notebook
nlp.add_pipe("custom_ents_component", after="ner")
nlp.add_pipe("negex", last=True, config={
    "ent_types":["12_lead_ecg_label", "oxygen_label"]
    })

# Add columns to receive resulting entity and negation labels, default 0
df['12_lead_label_found'] = 0
df['12_lead_label_negated'] = 0
df['oxygen_label_found'] = 0
df['oxygen_label_negated'] = 0

# Apply the matcher to each row of the dataframe:
for index, row in df.iterrows():
    # Combine text in multiple columns into a single string and pass to nlp
    doc = nlp(str(row['impressionPlan'])
    + ' '
    + str(row['injuryIllnessDetails'])
    )
        
    print(f"Row {index}:")
    for e in doc.ents:
        print(f"Entity: {e.text}, Label: {e.label_}, Negation: {e._.negex}")
        # Check if entity label matches a custom entity, and set flag
        # 12-lead label
        if e.label_ == '12_lead_ecg_label':
            df.at[index, "12_lead_label_found"] = 1
            # Check if the label was negated, and set flag
            if e._.negex == True:
                df.at[index, "12_lead_label_negated"] = 1
        # Oxygen label
        if e.label_ == 'oxygen_label':
            df.at[index, "oxygen_label_found"] = 1
            # Check if the label was negated, and set flag
            if e._.negex == True:
                df.at[index, "oxygen_label_negated"] = 1

df