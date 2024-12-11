# import pandas, spacy and the downloaded en_core_web_sm pre-trained model
import pandas as pd
import spacy
import en_core_web_sm
from spacy import displacy

# Load the pre-trained model as a language model into a variable called nlp
nlp = en_core_web_sm.load()

# Read the clinical data into a dataframe, selecting only certain columns
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

# TEST ROW FOR 12 LEAD ECG - Remove later
# df = df[3:4] # Row contains a great positive/negative 12 lead example 
# force in some text to check how Spacy handles various entries
# df.iloc[0,1] = "12 lead, 12lead, twelve lead, twelvelead"

# Function to extract named entities
def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Apply the function to the 'text' column
df['named_entities'] = df['impressionPlan']\
    .apply(extract_named_entities)

# Print three example lines from the df head
print(df.head(3))

# Visualize the entities in displacy
for index, row in df.iterrows():
    doc = nlp(row['impressionPlan'])
    # displacy.render(doc, style="ent", jupyter=True)
    # New loop to ID 12 lead ecg
    for token in doc:
        # Check if token is like a number
        if token.like_num:
            # Get the next token
            next_token = doc[token.i + 1]
            # Check if next token is "lead"
            if next_token.text == "lead":
                print("12-lead ECG found:", token.text, next_token.text)