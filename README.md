# Fake Job Posting Detector

An NLP-based machine learning tool that detects potentially fraudulent job postings, helping job seekers avoid scams.

## Problem
Fake job postings are a growing scam that targets job seekers, especially freshers, often to extract money or personal information. This project builds a model to flag suspicious postings automatically.

## Dataset
"Real or Fake Job Posting" dataset — 17,880 job postings, 866 confirmed fraudulent (about 4.8%).

## Key Insights (from EDA)
- Fake postings are over 4x more likely to have no company profile at all (68% vs 16% for real postings).
- Real postings are far more likely to include a company logo (82% vs 33% for fake postings).

## Approach
1. Combined text fields (title, company profile, description, requirements, benefits) into one text field
2. Converted text into numeric features using TF-IDF, combined with structured features (has_company_logo, telecommuting, has_questions)
3. Trained a Logistic Regression model with class weighting to handle severe class imbalance (~95% real vs ~5% fake)
4. Built a Streamlit app with two modes: single posting check, and bulk CSV upload for checking many postings at once
5. Added a transparent "reasoning" layer explaining why a posting was flagged (e.g., missing company profile, no logo, short description)

## Results
- Accuracy: 96%
- Recall for fraudulent postings: 90%
- Precision for fraudulent postings: 55%

Recall was prioritized over precision, since missing an actual scam is more costly than occasionally flagging a real posting for review.

## Limitations
This model is a text-pattern classifier — it has no ability to verify company legitimacy, check external links, or access real-time information. Like any fraud detection system, it is vulnerable to adversarial evasion (a sufficiently novel scam pattern could go undetected) and would ideally be paired with non-text verification signals, human review, and periodic retraining on newly confirmed fraud cases in a production setting.

## Live App
[Try the live app here](https://fake-job-posting-detector-r3t3avlltdcizqgfkocf9e.streamlit.app)

## Tech Stack
Python, pandas, scikit-learn, TF-IDF, Streamlit, joblib
