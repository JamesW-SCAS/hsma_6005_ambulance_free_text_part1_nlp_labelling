# Readme for HSMA project
# 6005_improving_ambulance_care_fast_feedback_quality_care_indicators
# Part 1 - Natural Language Processing element

This project aims to speed up feedback to Ambulance crews following 
on from Cardiac Arrest incidents.  The code looks for key phrases that we would
hope to find in clinical notes from such incidents, and tests whether the crew
performed the action or not.

This project has two repos:
1) NLP repo that labels the free-text data, and creates training data for 
a neural network

The main code is in the file "clinical_nlp.py"

2) Neural Net repo that uses the output from repo 1 and uses it when training
a custom neural network designed to look for negation in free-text, e.g. "
I did not administer Oxygen"