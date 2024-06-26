import re
import spacy
import pickle
import streamlit as st 

nlp = spacy.load('en_core_web_sm')

#Loading models
clf = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))



def preprocess_tokenize(text):
    doc = nlp(text)

    # Remove stop words, punctuation, URLs, email addresses, and extra spaces
    cleaned_tokens = [
        token.text for token in doc
        if not token.is_stop            # Remove stop words
        and not token.is_punct          # Remove punctuation
        and not token.like_url          # Remove URLs
        and not token.like_email        # Remove email addresses
        and not token.is_space          # Remove newlines and extra spaces
    ]
    cleaned_text = ' '.join(cleaned_tokens)
    cleaned_text = re.sub('#[\w-]+', '', cleaned_text) #Remove # words
    cleaned_text = re.sub('@\w+', '', cleaned_text) #Remove @ words
    cleaned_text = re.sub(r'[^\x00-\x7f]', '', cleaned_text) #Remove non-ascii


    return cleaned_text.strip()

#Web app
def main():
    st.title('Resume Screening App')
    uploaded_file = st.file_uploader('Upload Resume', type = ['txt', 'pdf'])
    
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
            
        clean_resume = preprocess_tokenize(resume_text)
        clean_resume = tfidf.transform([clean_resume])
        prediction_id = clf.predict(clean_resume)[0]
        st.write(prediction_id)
        
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
        
        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name) 
        
    
#Python Main
if __name__ == "__main__":
    main() 
