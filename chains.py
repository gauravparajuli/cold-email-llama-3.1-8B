import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain.schema import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import chromadb
from dotenv import load_dotenv

load_dotenv() # load environment variables

class Chain:
    def __init__(self) -> None:
        self.llm = ChatGroq(temperature=0, model_name='llama-3.1-8b-instant', api_key=os.environ['GROQ_API_KEY'])

    def extract_job(self, cleaned_text):

        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPPED FROM WEBSITE
            {page_data}
            ### INSTRUCTION:
            The scrapped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing
            the following keys: `role`, `experience`, `skills` and `description`.

            only return the valid json.
            ### VALID JSON (NO PREAMBLE)
            """
        )

        chain_extract = prompt_extract | self.llm | StrOutputParser()
        res = chain_extract.invoke({"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res)
        except OutputParserException:
            raise OutputParserException('context too big, unable to parse jobs.')
        
        return res if isinstance(res, list) else [res]

    def write_email(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Ramesh, a business development executive at Everest Dev. Everest Dev is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools.
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability,
            process optimization, cost reduction, and heightened overall efficiency.
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Everest Dev
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Everest Dev's portfolio: {link_list}
            Remember you are Ramesh, BDE at Everest Dev.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE)
            """
        )

        chain_email = prompt_email | self.llm | StrOutputParser()
        res = chain_email.invoke(dict(job_description=str(job), link_list=links))

        return res
    
if __name__ == '__main__':
    print(os.environ['GROQ_API_KEY'])