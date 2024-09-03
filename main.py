import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from vectorstore_populate import Portfolio
from utils import clean_text

def create_streamlit_app(llm, portfolio, clean_text):
    st.title('Cold Email Generator')
    url_input = st.text_input('Enter JOB URL:', value='https://jobs.nike.com/job/R-31388')
    submit_button = st.button('Generate')

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_job(data)
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_email(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f'An error occured {e}')

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout='wide', page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)