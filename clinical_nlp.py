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
    'presentingConditionChiefComplaintHistoryFreeText'
    ]
df = pd.read_csv("nlp_input.csv"
, usecols = filtered_cols
, nrows=10
)
# Drop any rows with null data
df.dropna(
    subset='presentingConditionChiefComplaintHistoryFreeText',
 inplace=True)

# Function to extract named entities
def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Apply the function to the 'text' column
df['named_entities'] = df['presentingConditionChiefComplaintHistoryFreeText']\
    .apply(extract_named_entities)

# Print three example lines from the df head
print(df.head(3))

# Visualize the entities in displacy
for index, row in df.iterrows():
    doc = nlp(row['presentingConditionChiefComplaintHistoryFreeText'])
    displacy.render(doc, style="ent", jupyter=True)