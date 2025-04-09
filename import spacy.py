import spacy
from spacy.matcher import Matcher
from negspacy.negation import Negex
from spacy.tokens import Span
import pandas as pd

# Create sample DataFrame
df = pd.DataFrame({'text_column': [
    "12 lead ecg was done but no oxygen given",
    "oxygen saturation levels were normal",
    "12 lead ecg showed abnormalities but no oxygen therapy"
]})

# Load spaCy model and define matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define patterns for matching
pattern = [{"LOWER": "12"}, {"LOWER": "lead"}, {"LOWER": "ecg"}]
oxy_pattern = [{"LOWER": "oxygen"}]
matcher.add("ECG_PATTERN", [pattern])
matcher.add("O2_PATTERN", [oxy_pattern])

# Custom component to add entities based on matcher
@spacy.Language.component("custom_ents_component")
def custom_ents_component(doc):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label=match_id) for match_id, start, end in matches]
    doc.ents = spans  # Override doc.ents with matched spans
    return doc

# Add components to the pipeline
nlp.add_pipe("custom_ents_component", after="ner")
nlp.add_pipe("negex", last=True, config={
    "ent_types": ["ECG_PATTERN", "O2_PATTERN"]
})

# Iterate through each row in the DataFrame and process text
for index, row in df.iterrows():
    doc = nlp(row['text_column'])  # Process each row's text with spaCy pipeline
    print(f"Row {index}:")
    for e in doc.ents:
        print(f"Entity: {e.text}, Label: {e.label_}, Negation: {e._.negex}")
