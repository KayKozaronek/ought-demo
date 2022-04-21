import streamlit as st
import pandas as pd
from io import StringIO
from transformers import pipeline
import seaborn as sns

"""
# Ought ML Special Ops - Demo

## Task 9: Generate claims from papers

**Description:**
- Using the full-text of a paper, classify whether each sentence is a ‘claim’. 
- Then return all the claims of a paper that a user can search through and verify using Elicit.
"""

@st.cache(allow_output_mutation=True)
def instantiate_pipeline():
    zero_shot = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')
    return zero_shot

cm = sns.light_palette("#31333F", as_cmap=True)

'### a) Classifying a single sentence'

model = instantiate_pipeline()

# fixed_candidate_labels = ['claim','question', 'conclusion', 'citation']
candidate_labels = st.multiselect('Select Options',
                              options=['claim', 'question', 'background information', 'conclusion', 'citation'] 
                              )

data = st.text_input('Input some text')
with st.expander('Results'):
        
    if data != '' and len(candidate_labels) != 0:   
        res = model(data, candidate_labels)
        st.write(res)
        

"""### b) Classifying a document
"""


paper = st.file_uploader(label='Upload a paper here',
                 type = ['txt','csv'],
                 accept_multiple_files=False)


if paper is not None and len(candidate_labels) != 0:
    # To convert to a string based IO:
    stringio = StringIO(paper.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read().split('\n')

    with st.expander('See raw text'):
        st.write(string_data)
    
    df = pd.DataFrame(columns=['sentence', *candidate_labels])
    for i, sentence in enumerate(string_data):
        res = model(sentence, candidate_labels)
        classification = {k :v for k,v in zip(res['labels'],res['scores'])} 
        df.loc[i] = {'sentence':sentence, **classification}

    st.dataframe(df.style.background_gradient(cmap=cm, axis=1))
    # st.dataframe(df.style.highlight_max(subset=candidate_labels, axis=1))

'''
### c) Verify Claims 
'''

check_text_button = st.button('Verify Claims')

if check_text_button:
    st.write("""You've reached the end of this demo. 
            Want to see more? 
            """)
    st.write('**Well how about hiring me?**')
    st.balloons()

    
