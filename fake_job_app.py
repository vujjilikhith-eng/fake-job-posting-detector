import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack

# Load the saved model and vectorizer
model = joblib.load('fake_job_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

st.title("Fake Job Posting Detector")

mode = st.radio(
    "Choose mode:",
    ["Check a single posting", "Bulk check (upload a CSV of many postings)"]
)

# --- Shared helper: build features for one row of data ---
def build_features(title, company_profile, description, requirements, benefits,
                    telecommuting, has_company_logo, has_questions):
    full_text = f"{title} {company_profile} {description} {requirements} {benefits}"
    text_features = tfidf.transform([full_text])
    extra_features = np.array([[telecommuting, has_company_logo, has_questions]])
    return hstack([text_features, extra_features])

def get_reasons(company_profile, description, requirements, benefits, has_company_logo):
    reasons = []
    if not str(company_profile).strip():
        reasons.append("No company profile provided")
    if has_company_logo == 0:
        reasons.append("No company logo included")
    if len(str(description).strip()) < 80:
        reasons.append("Job description is unusually short")
    if not str(requirements).strip():
        reasons.append("No requirements listed")
    if not str(benefits).strip():
        reasons.append("No benefits mentioned")
    return "; ".join(reasons) if reasons else "No obvious red flags"


# ============================================================
# MODE 1: Single posting check
# ============================================================
if mode == "Check a single posting":
    st.write("Paste a job posting below to check whether it looks genuine or potentially fraudulent.")

    title = st.text_input("Job Title")
    company_profile = st.text_area("Company Profile / About the Company", height=100)
    description = st.text_area("Job Description", height=150)
    requirements = st.text_area("Requirements", height=100)
    benefits = st.text_area("Benefits", height=80)

    st.write("Additional details:")
    telecommuting = st.selectbox("Is this a remote/telecommuting job?", ["No", "Yes"])
    has_company_logo = st.selectbox("Does the posting include a company logo?", ["No", "Yes"])
    has_questions = st.selectbox("Does the posting include screening questions?", ["No", "Yes"])

    if st.button("Check This Posting"):
        combined_features = build_features(
            title, company_profile, description, requirements, benefits,
            1 if telecommuting == "Yes" else 0,
            1 if has_company_logo == "Yes" else 0,
            1 if has_questions == "Yes" else 0
        )

        prediction = model.predict(combined_features)[0]
        probability = model.predict_proba(combined_features)[0][1]

        st.subheader("Result")
        if prediction == 1:
            st.error(f"This posting looks potentially FAKE. (Fraud probability: {probability:.2%})")
        else:
            st.success(f"This posting looks REAL. (Fraud probability: {probability:.2%})")

        st.subheader("Why this result?")
        reasons_text = get_reasons(company_profile, description, requirements, benefits,
                                    1 if has_company_logo == "Yes" else 0)
        for r in reasons_text.split("; "):
            st.write(f"- {r}")

        st.caption("Note: This tool provides an automated estimate based on patterns in historical data. It is not a guarantee — always verify a job posting independently.")


# ============================================================
# MODE 2: Bulk CSV check
# ============================================================
else:
    st.write("Upload a CSV file containing multiple job postings to check them all at once.")
    st.write("Your file should have these columns (case-sensitive): **title, company_profile, description, requirements, benefits, telecommuting, has_company_logo, has_questions**")
    st.caption("Tip: if your file doesn't have telecommuting / has_company_logo / has_questions columns, leave them out — they'll default to 0 (No) for every row.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)

        # Fill in any missing expected columns with safe defaults
        text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        for col in text_cols:
            if col not in bulk_df.columns:
                bulk_df[col] = ""
        bulk_df[text_cols] = bulk_df[text_cols].fillna("")

        for col in ['telecommuting', 'has_company_logo', 'has_questions']:
            if col not in bulk_df.columns:
                bulk_df[col] = 0
            bulk_df[col] = bulk_df[col].fillna(0).astype(int)

        if st.button("Run Bulk Check"):
            full_text = (
                bulk_df['title'] + ' ' + bulk_df['company_profile'] + ' ' +
                bulk_df['description'] + ' ' + bulk_df['requirements'] + ' ' + bulk_df['benefits']
            )
            text_features = tfidf.transform(full_text)
            extra_features = bulk_df[['telecommuting', 'has_company_logo', 'has_questions']].values
            combined_features = hstack([text_features, extra_features])

            predictions = model.predict(combined_features)
            probabilities = model.predict_proba(combined_features)[:, 1]

            bulk_df['prediction'] = np.where(predictions == 1, "Fake", "Real")
            bulk_df['fraud_probability'] = (probabilities * 100).round(2)
            bulk_df['reasons'] = bulk_df.apply(
                lambda row: get_reasons(
                    row['company_profile'], row['description'], row['requirements'],
                    row['benefits'], row['has_company_logo']
                ), axis=1
            )

            total = len(bulk_df)
            fake_count = (bulk_df['prediction'] == "Fake").sum()
            real_count = total - fake_count

            st.subheader("Summary")
            st.write(f"Total postings checked: **{total}**")
            st.write(f"Flagged as Real: **{real_count}**")
            st.write(f"Flagged as Fake: **{fake_count}**")

            st.subheader("Postings flagged as Fake")
            fake_postings = bulk_df[bulk_df['prediction'] == "Fake"][
                ['title', 'fraud_probability', 'reasons']
            ].sort_values('fraud_probability', ascending=False)
            st.dataframe(fake_postings, use_container_width=True)

            st.subheader("Full results (all postings)")
            st.dataframe(bulk_df[['title', 'prediction', 'fraud_probability', 'reasons']], use_container_width=True)

            # Let the user download the full results as a CSV
            csv_output = bulk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download full results as CSV",
                data=csv_output,
                file_name="job_postings_results.csv",
                mime="text/csv"
            )

            st.caption("Note: This tool provides an automated estimate based on patterns in historical data. It is not a guarantee — always verify flagged postings independently.")
