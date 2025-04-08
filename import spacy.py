import spacy
from spacy.matcher import Matcher
from negspacy.negation import Negex
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("negex", last=True, config={"ent_types":["ECG_PATTERN"]})

matcher = Matcher(nlp.vocab)
pattern = [{"LOWER": "12"}, {"LOWER": "lead"}, {"LOWER": "ecg"}]
oxy_pattern = [{"LOWER" : "oxygen"}]
matcher.add("ECG_PATTERN", [pattern])
matcher.add("O2_PATTERN", [oxy_pattern])

@spacy.Language.component("custom_ents_component")
def custom_ents_component(doc):
    # docs = []
    matches = matcher(doc)
    spans = [Span(doc, start, end, label=match_id)
     for match_id, start, end in matches]
    print(spans)
    doc.ents = spans
    # docs.append(doc)
    return doc

# Try adding components to the pipeline in order, as per Sammi's notebook
nlp.add_pipe("custom_ents_component", after="ner")
nlp.add_pipe("negex", last=True, config={
    "ent_types":["ECG_PATTERN", "O2_PATTERN"]
    })

doc = nlp("12 lead ecg was done but no oxygen given")

# Below code produces outputs
for e in doc.ents:
    print(e.text, e.label_, e._.negex)