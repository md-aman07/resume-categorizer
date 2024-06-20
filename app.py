import numpy as np
import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')


clf = pickle.load(open('H:\Resume_Category_classifier\models\model.pkl','rb'))
tfidf = pickle.load(open('H:\Resume_Category_classifier\models\idf.pkl','rb'))
le = pickle.load(open('H:\Resume_Category_classifier\models\label.pkl','rb'))

def cleanResume(text):
  text = re.sub('http\S+\s',' ',text)
  text = re.sub('RT|cc',' ',text)
  text = re.sub('#\S+',' ',text)
  text = re.sub('@\S+',' ',text)
  text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
  text = re.sub(r'[^\x00-\x7f]',r' ',text)
  text = re.sub('\s+', ' ', text)

  return text

def main():
    st.title("Resume Category Classifier")
    uploaded_file=st.file_uploader('Upload Resume',type=['txt','pdf'])
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume=cleanResume(resume_text)
        vectored_resume = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(vectored_resume)[0]
        predicted_label = le.inverse_transform([prediction_id])[0]  # Corrected here
        st.subheader("Your Resume Category is :")
        st.write(predicted_label)




        





if __name__== "__main__":
    main()

