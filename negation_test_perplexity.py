import spacy
from spacy.matcher import Matcher

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Create a Matcher
matcher = Matcher(nlp.vocab)

# Define patterns for "12 lead"
twelve_lead_pattern = [{"TEXT": "12"}, {"LOWER": "lead"}]
matcher.add("TWELVE_LEAD", [twelve_lead_pattern])

# Define negation lemmas
negation_lemmas = ["no", "not", "never", "neither", "nor", "none", "nobody", "nowhere", "nothing", "without", "absent", "negative", "deny", "refuse", "decline", "reject"]

def is_negation(token):
    return token.lemma_.lower() in negation_lemmas

def analyze_text(text):
    doc = nlp(text)
    
    twelve_lead_matches = matcher(doc)
    
    results = []
    for match_id, start, end in twelve_lead_matches:
        span = doc[start:end]
        
        # Get the sentence containing the "12 lead" mention
        sent = span.sent
        
        # Check for negations within the same sentence
        nearby_negation = any(
            is_negation(token) and abs(token.i - start) <= 6
            for token in sent
        )

        # Check for negations in the previous sentence
        prev_sent_end = sent[0].i
        if prev_sent_end > 0:
            prev_sent = doc[doc[prev_sent_end - 1].sent.start:prev_sent_end]
            nearby_negation = nearby_negation or any(
                is_negation(token) and abs(token.i - start) <= 6
                for token in prev_sent
            )

        # Check for negations in the next sentence
        next_sent_start = sent[-1].i + 1
        if next_sent_start < len(doc):
            next_sent = doc[next_sent_start:doc[next_sent_start].sent.end]
            nearby_negation = nearby_negation or any(
                is_negation(token) and abs(token.i - start) <= 6
                for token in next_sent
            )
        
        context = (
            (doc[doc[prev_sent_end - 1].sent.start:prev_sent_end].text + " " if prev_sent_end > 0 else "") +
            sent.text +
            (" " + doc[next_sent_start:doc[next_sent_start].sent.end].text if next_sent_start < len(doc) else "")
        )
        
        result = {
            "span": span.text,
            "negated": nearby_negation,
            "context": context
        }
        results.append(result)
    
    return results

# Test the function
test_texts = [
    "A 12 lead ECG was performed.",
    "No 12 lead ECG was done.",
    "The patient refused a 12 lead test.",
    "12 lead ECG was not performed due to patient condition.", # Wrong - 3 token limit?
    "The doctor ordered a 12 lead, but it was not completed.", # NOTE! It gets this one wrong because the negation is more than 3 tokens away from the "12 lead" span.
    "I did a 12 lead. Not again.", # Gets this one wrong too; don't need it to look into previous and next sentences?
    "He never agreed to a 12 lead. It wasn't necessary.",
    "The nurse prepared for a 12 lead. The patient declined." # Gets this one wrong too; I think because of the 3 token search window.
]

for text in test_texts:
    print(f"Text: {text}")
    results = analyze_text(text)
    for result in results:
        print(f"  Span: {result['span']}")
        print(f"  Negated: {result['negated']}")
        print(f"  Context: {result['context']}")
    print()


'''
Key changes in this updated version:
We've replaced the PhraseMatcher with a list of negation lemmas and a function is_negation() that checks if a token's lemma is in this list.
The negation check now uses the is_negation() function instead of matching against a fixed list of words:
python
nearby_negation = any(
    is_negation(token) and abs(token.i - start) <= 3
    for token in sent
)
We've expanded the list of negation lemmas to include more variations and related concepts.
The code now checks for negations in the previous, current, and next sentences using the lemma-based approach.
The context provided in the result now includes the previous sentence (if it exists) as well as the current and next sentences.
This implementation provides a more robust negation detection system that can identify various forms of negation words across multiple sentences. It should effectively handle cases where negations are expressed using different word forms or in adjacent sentences.
'''