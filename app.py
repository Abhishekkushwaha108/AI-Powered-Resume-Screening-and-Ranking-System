import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text_content = [page.extract_text() for page in reader.pages if page.extract_text()]
    return " ".join(text_content)

# Function to calculate similarity scores
def calculate_resume_similarity(job_description, resume_contents):
    corpus = [job_description] + resume_contents
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(corpus).toarray()
    similarity_scores = cosine_similarity([tfidf_vectors[0]], tfidf_vectors[1:]).flatten()
    return similarity_scores

# Streamlit UI Setup
st.title("AI-Based Resume Screening & Ranking System")

st.header("Enter Job Description")
job_desc_input = st.text_area("Provide the job description here")

st.header("Upload Resume Files (PDF Only)")
uploaded_resumes = st.file_uploader("Select multiple resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_resumes and job_desc_input:
    extracted_texts = [extract_text_from_pdf(resume) for resume in uploaded_resumes]
    similarity_results = calculate_resume_similarity(job_desc_input, extracted_texts)

    # Create and display ranking table
    ranking_results = pd.DataFrame({
        "Resume Name": [resume.name for resume in uploaded_resumes],
        "Relevance Score": similarity_results
    })
    ranking_results = ranking_results.sort_values(by="Relevance Score", ascending=False)

    st.subheader("Ranked Resumes")
    st.dataframe(ranking_results)
