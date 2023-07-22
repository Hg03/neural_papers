import streamlit as st
from pandasai import PandasAI
import pandas as pd
from markdownlit import mdlit
from pandasai.llm.falcon import Falcon

st.set_page_config(layout='wide')

# Create a context manager to run an event loop

@st.cache_data
def load_data():
    authors = pd.read_csv('data/authors.csv')
    papers = pd.read_csv('data/papers.csv')
    authors.drop(['institution'],axis=1,inplace=True)
    authors.dropna(inplace=True)
    #papers = pd.read_csv('/kaggle/input/nips-papers-1987-2019-updated/papers.csv')
    papers['abstract'].fillna("Didn't mentioned",inplace=True)
    papers['full_text'].fillna("Didn't mentioned",inplace=True)
    merged = authors.merge(papers,on='source_id')
    merged['full_name'] = merged['first_name'] + ' ' +  merged['last_name']
    return merged
  
def neural_paper():   
    mdlit('# 🪢🪢 [green]Neural[/green] [yellow]Papers[/yellow] 📝')
    data = load_data()
    st.image('static/intro.png')
    authors = st.selectbox('Choose the Author',list(data['full_name'].unique()))
    df = data.loc[data['full_name'] == authors,['full_name','year','title','abstract']]
    mdlit("## [blue]Author's Description[/blue]")
    st.data_editor(df)
    question = st.text_input('You can ask question regarding these papers as well 🔥')
    if question:
        llm = Falcon(api_token=st.secrets['HUGGINGFACE_API_KEY'])
        chain = PandasAI(llm)
        st.write(chain.run(data,question))

def custom_bot(data):
    st.title('Now you can chat with your data happily')
    question = st.text_input('You can ask question regarding your dataset 🔥')
    if question:
        llm = Falcon(api_token=st.secrets['HUGGINGFACE_API_KEY'])
        chain = PandasAI(llm)
        st.write(chain.run(data,question))
    
def main():
    st.sidebar.title('You can upload your data and chat with your data as well :smile:')
    uploaded_data = st.sidebar.file_uploader("Upload your own dataset",type='csv')
    if uploaded_data is not None:
        custom_bot(uploaded_data)
    else:
        neural_paper()

main()
    


    
