import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
import requests
from urllib.parse import urlparse
import pickle
import pandas as pd
from PIL import Image
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading
import joblib
import openai
from openai import AzureOpenAI
# from langchain.embeddings import AzureOpenAIEmbeddings
# from langchain.llms import AzureOpenAI
from langchain_community.llms import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from docx import Document
from dotenv import load_dotenv
 
# Initialize text-to-speech engine
engine = pyttsx3.init()
rate = engine.getProperty('rate')   # getting details of current speaking rate
print (rate)                        #printing current voice rate
engine.setProperty('rate',100)
 
 
# Function to convert text to speech
def text_to_speech(text):
    print(text)
    print("hereeee")
    if engine._inLoop:
        engine.endLoop()
    try:
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except RuntimeError as e:
        print("Error:", e)
 
def text_to_speech_thread(text):
    thread = threading.Thread(target=text_to_speech, args=(text,))
    thread.start()
 
vector_store = []
 
st.set_page_config(
    page_title="Chat with Multiple PDFs",
    page_icon=":blue_book:",
)

load_dotenv()
 
# st.set_option('server.allow_dangerous_deserialization', True)
azure_endpoint_url = os.getenv("Azure_Endpoint_Url")
api_key = os.getenv("Api_Key")
deployment_model_GPT = os.getenv("Deployment_Model_GPT")
api_version_GPT = os.getenv("Api_Version_GPT")
 
# Initialize Azure OpenAI
llm = AzureOpenAI(api_key=api_key, deployment_name=deployment_model_GPT, openai_api_base=azure_endpoint_url, openai_api_version=api_version_GPT, temperature=0)
 
embeddings = AzureOpenAIEmbeddings(model = "POC-TXT-EMBED-3-LARGE", api_key=api_key, azure_ad_token=None, azure_ad_token_provider=None, azure_endpoint=azure_endpoint_url)
 
 
# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
 
def generate_summary(text):
    input_chunks = get_text_chunks(text)
    output_chunks = []
    for chunk in input_chunks:
        response = llm.chat.completions.create(
            model=deployment_model_GPT,
            prompt=(f"Please summarize the following text:\n{chunk}\n\nSummary:"),
            temperature=0.5,
            max_tokens=1024,
            n = 1,
            stop=None
        )
        summary = response.choices[0].text.strip()
        output_chunks.append(summary)
    return " ".join(output_chunks)
 
# Function to save the vector store to a file
def save_vector_store(vector_store, filename='vector_store.pkl'):
    vector_store.save_local(filename)
 
# Function to load the vector store from a file
def load_vector_store(filename='vector_store.pkl'):
    if os.path.exists(filename):
        return FAISS.load_local(filename, embeddings, allow_dangerous_deserialization=True)
        # return FAISS.load_local(filename, embeddings)
    else:
        print("Failed to load vector store. File does not exist.")
        return None
 
    # Function to create vector store from text chunks
def get_vector_store(text_chunks):
    print("Length of text_chunks:", len(text_chunks))
    if not text_chunks:
        print("Text chunks is empty. Returning None.")
        return None
 
    embeddings = AzureOpenAIEmbeddings(model = "POC-TXT-EMBED-3-LARGE", api_key=api_key, azure_ad_token=None, azure_ad_token_provider=None, azure_endpoint=azure_endpoint_url)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save vector store to pickle file
    save_vector_store(vector_store)
    print("Vector store created and saved to pickle file.")
    return vector_store
 
# Function to perform similarity search with score
# def similarity_search_with_score(vector_store, query):
#     results_with_scores = vector_store.similarity_search_with_score(query)
#     # results_with_scores_sorted = sorted(results_with_scores, key=lambda x: x[1])  # Sort based on score
#     for idx, (doc, score) in enumerate(results_with_scores):
#         print(score)
#         print(f"Rank: {idx + 1}, Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
 
# Function to serialize the FAISS index to bytes
def serialize_to_bytes(vector_store):
    return vector_store.serialize_to_bytes()
 
# Function to deserialize the FAISS index from bytes
def deserialize_from_bytes(serialized_bytes, embeddings):
    return FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=serialized_bytes, allow_dangerous_deserialization=True)
 
# Function to merge two FAISS vector stores
def merge_vector_stores(vector_store1, vector_store2):
    vector_store1.merge_from(vector_store2)
 
 
prompt_template = """
Please provide a detailed and well-formatted answer. Ensure the following guidelines are met:
 
- **Accuracy**: Provide correct information based on the context.
- **Units**: Include appropriate units (e.g., $, million, billion) for numerical data.
- **Logical Reasoning**: For open-ended questions, provide logical and well-justified answers.
- **Formatting**: Ensure the answer is formatted neatly and clearly. If the user requests bullet points, each point should start on a new line.
- **Date Calculations**: If an end date is asked for and not mentioned, calculate it based on the provided context.
Give accurate responses from the given/specified context only.
\n\n
Context:\n {context}\n
Question:\n{question}\n
 
Answer:
"""
 
# Function to create conversational chain from vector store
def get_conversational_chain(vector_store, prompt_template):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
    )
    return conversation_chain
 
# Global variable to store FAQ data
faq_data = {}
 
 
import fitz  # PyMuPDF
import pytesseract
import io
from PIL import Image
from PyPDF2 import PdfReader
 
# Function to calculate token count
def calculate_token_count(document_text):
    total_characters = len(document_text)
    estimated_token_count = total_characters // 4
    return estimated_token_count
 
 
def process_scanned_pdf(pdf_path):
    # Clear the contents of the "User_Reports" folder
    pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    main_text = ""
    with pdf_path as pdf_file:
        images = extract_images_from_pdf(pdf_file)
        for img in images:
            main_text += pytesseract.image_to_string(img)
       
       
        # Calculate token count for the extracted text
        token_count = calculate_token_count(main_text)
        total_characters = calculate_token_count(main_text)
       
        # Use the token count for your pricing calculation or other purposes
        print("Estimated number of tokens in the document:", token_count)
        print("Estimated number of characters in the document:", total_characters)
 
        text_chunks = get_text_chunks(main_text)
        vector_store = get_vector_store(text_chunks)
        print(text_chunks)
        st.session_state.conversation = get_conversational_chain(vector_store, prompt_template)
        st.session_state.faq_conversation = get_conversational_chain(vector_store, prompt_template)
       
    return f"Processed PDF: {os.path.basename(pdf_path.name)}"
 
def extract_images_from_pdf(pdf_file):
    images = []
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            image_bytes = page.get_pixmap().tobytes()
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return images
 
 
# Function to process uploaded PDFs
def process_pdf(pdf_path):
    # Clear the contents of the "User_Reports" folder
    main_text = ""
    with pdf_path as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            main_text += page.extract_text()
 
        # # Calculate token count for the document
        # token_count = calculate_token_count(main_text)
        # total_characters = calculate_token_count(main_text)
       
        # # Use the token count for your pricing calculation or other purposes
        # print("Estimated number of tokens in the document:", token_count)
        # print("Estimated number of characters in the document:", total_characters)
 
 
        text_chunks = get_text_chunks(main_text)
        vector_store = get_vector_store(text_chunks)
        st.session_state.conversation = get_conversational_chain(vector_store, prompt_template)
        st.session_state.faq_conversation = get_conversational_chain(vector_store, prompt_template)
       
    return f"Processed PDF: {os.path.basename(pdf_path.name)}"
 
def df_to_markdown(df):
    # Extract column names and data
    columns = df.columns
    data = df.values.tolist()
   
    # Create Markdown table string
    markdown_table = "| " + " | ".join(columns) + " |\n"
    markdown_table += "| " + " | ".join(['---'] * len(columns)) + " |\n"
   
    for row in data:
        markdown_table += "| " + " | ".join(map(str, row)) + " |\n"
   
    return markdown_table
 
 
# Function to generate FAQs
def generate_faqs(pdf_name):
    print("HEREEE")
    faq_questions = {
        "FAQs": [
       # "Generate the FAQs and their answers for each uploaded document.",
        # "Give in tabular format, the Type of Agreement, Party name - Clients, Party name - Service Providers (Either of Evolutionary/ Evosys/MST/ Metasoft tech/ Biz Analytica/ Digility/ Trans American/ Indigo blue), Document date (the date on which the document was made or signed), Effective date, End date (the date on which the contract terminates) in tabular format. If end date is not given, calculate it. You should display the dates in the format of 'DD Month YYYY'. It is mentioned. The document date, effective date and the end date is always mentioned so please give it. It is always available in the context. Give all this in tabular format.",
        # "Extract the Party-Client name, Uncapped Liabilities, Warranty, Indemnity, Services/Damage, Jurisdiction/Law, Ability to Exit Condition, Insurance Description and requirements, Mitigation Description in tabular format. These should be the column headings."
        # """Extract the following details from the provided contract and present them in a tabular format.
        #            FOLLOWING IS THE FORMAT:
        #             | Client|  Liability |Uncapped Liabilities |Warranty| Indemnity |Services/Damages|Jurisdiction/Law|Ability to Exit|Insurance|
        #             |---|---|---|---|---|---|---|---|---|
        #             | output | output | output | output | output | output |output (Jurisdiction city mentioned) | output | output|. Each and every thing listed below will be there in the context.
        #             Details to Extract:
        #             Client: Extract the 'Customer's name' or 'Buyer's name' or 'First Party' name, which is name of party signing the contract with mastek.
        #             Liability: Extract this values from 'LIMITATION OF LIABILITY' clause only, Determine if liabilities for the services provided are capped at a specified (amount or percentage given under 'LIMITATION OF LIABILITY' clause only) of the Work Order value, If capped, extract the payable 'value' from 'limitation of liability' clause only and display it as 'Capped at {given value} of {given condition}', If value is not provided then display 'UNCAPPED'.
        #             Uncapped Liabilities: if Liability is 'UNCAPPED' then this field will be 'N/A' else list the types of liabilities that are not capped (e.g., death, personal injury, fraud, breach of law, etc.) Only list them, dont give whole phrases.
        #             Warranty: In the "Warranty" section, if the word "except" is used, extract the details; otherwise, mark it as "Standard". If section is not there then give 'Not available in the contract'.
        #             Indemnity: In the "Indemnity" section, if the word "except" is used, extract the details; otherwise, mark it as "Standard". If section is not there then give 'Not available in the contrsact'.
        #             Services/Damages: If the damages are ongoing, indicate "Ongoing." If there is a cap mentioned (either a percentage or a specific amount), extract that information.
        #             Insurance: Extract types of insurance required for any liability or coverage process and give it in the form 'Insurance type: amount' for all the insurance coverage. Dont give detail for each insaurance type. If insaurance is not mentioned in the context then give 'Not provided' or N/A.
        #             Ability to Exit: Check the "Ability to Exit" or "Termination Clause" sections and extract whether the service provider can terminate the agreement and under which condition. If not provided than give 'Not available in the context'.
        #             Jurisdiction: Extract the Jurisdiction place (country and city) provided in the context.
        #             """
        
            "Effective Date: Effective date (Month Date Year) or Support start date or Proposed start date and such similar phrases and refers to the agreement's/ support's start is the document's effective date. Extract the date in as it is format.",
            "End Date: The document's end date (Month Date, Year) is the same as 'Contract Ending Date' or 'Support Period' or similar phrases and refers to the agreement's end, not the clause's end. If the end date is a duration. For Example: 'x years from effective date' or 'y months from effective date', CALCULATE it as End Date = Effective Date + x years or Effective date + y months. If the end date is in another document, indicate as 'Mentioned in SOW' or name the document.",
            "Client: Client name is the name of the 'Customer' or 'Buyer' or 'First party' mentioned.",
            "Liability: Extract from 'LIMITATION OF LIABILITY' clause only, Determine if liabilities for the services provided are capped at a specified (amount or percentage given under 'LIMITATION OF LIABILITY' clause only) of the Work Order value, If capped, extract the payable 'value' from 'limitation of liability' clause only and display it as 'Capped at {given value} of {given condition}', If value is not provided then display 'UNCAPPED'.",
            "Uncapped Liabilities: if Liability is 'UNCAPPED' then this field will be 'N/A' else list the types of liabilities that are not capped (e.g., death, personal injury, fraud, breach of law, etc.)",
            "Warranty: In the 'Warranty' section, if the word 'except' is used, extract the details; otherwise, mark it as 'Standard'. If section is not there then give 'Not available in the contract'.",
            "Indemnity: In the 'Indemnity' section, if the word 'except' is used, extract the details; otherwise, mark it as 'Standard'. If section is not there then give 'Not available in the contract'.",
            "Services/Damages: If the damages are ongoing, indicate 'Ongoing.' If there is a cap mentioned (either a percentage or a specific amount), extract that information.",
            "Insurance: Extract types of insurance and amount required for any liability or coverage process. If insaurance is not mentioned in the context then give 'Not provided' or N/A.",
            "Termination for Convenience: Check the 'Ability to Exit' or 'Termination Clause' sections and extract whether the service provider can terminate the agreement. If not provided than give 'Not available in the context.",
            "Jurisdiction: Extract the Jurisdiction place (country and city) provided in the context."
        ]
    }
    global faq_data
   
    faq_data[pdf_name] = {}
    pdf_faq = {}
 
    responses=""
 
    # Extract answers using LLM chain
    for category, questions in faq_questions.items():
        for question in questions:
            response = st.session_state.faq_conversation({'question': question})
            if response['chat_history']:
                answer = response['chat_history'][-1].content
                # pdf_faq[question] = answer
                responses += answer + "\n"
            else:
                st.write("A: No answer found")
   
 
   
    # faq_data[pdf_name] = pdf_faq
    print(responses)
   
 
    txt_chunks = get_text_chunks(responses)
    vector_store = get_vector_store(txt_chunks)
    st.session_state.faq_conversation_s = get_conversational_chain(vector_store, prompt_template)
    st.session_state.summary=get_conversational_chain(vector_store, prompt_template)
 
    #create a summary of the above response.
    summary = st.session_state.summary({'question': "create a summary of the whole context within 1200 characters and give each sentence in bullet points."})
    # print("-------HERE is the summary------")
    # print("----------")
    # print(summary)
    # print("")
    # print("")
 
    if summary['chat_history']:
        summary_answer = summary['chat_history'][-1].content
        summary_answer= summary_answer.replace("<|im_end|>",'')
        pdf_faq["The summary of the contract: "] = summary_answer
 
 
    table_question="""You're an advanced AI trained in legal document analysis for service provider contracts.
                    Extract the following fields from the provided contract and present them in a tabular format.
                    Extract as instructed below.
                    FOLLOWING IS THE FORMAT:
                     | Client|  Liability |Uncapped Liabilities |Warranty| Indemnity |Services/Damages|Jurisdiction/Law|Termination for convenience|Insurance|
                     |---|---|---|---|---|---|---|---|---|
                     | output | output | output | output | output | output |output (Jurisdiction city mentioned) |output| output|. Each and every thing listed below will be there in the context.
                     Details to Extract:
                    - Client: Name of the customer or buyer (first party) mentioned in the contract.
                    - Liability: If liability is capped, extract the clause and payable value from the "limitation of liability" clause and display as 'Capped at {/value} of {condition}'. If value is not provided, display 'UNCAPPED'.
                    - Uncapped Liabilities: Is a party’s liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type such as death, personal injury, fraud, breach of law.
                    - Warranty: In the "Warranty" section, if the word "except" is used, extract in single sentence; otherwise, mark it as "Standard". If section is not there then give 'Not available in the contract'.
                    - Indemnity: State "Standard" if this format is provided for indemnification clause: "Party A shall indemnify, defend, and hold harmless Party B from and against any and all claims, liabilities, losses, damages, costs, and expenses (including reasonable attorney’s fees) arising out of or in connection with any third-party claims related to Party A’s breach of this Agreement, negligence, or willful misconduct." If section is not there then give 'Not available in the contract'.
                    - Services/Damages: If the damages are ongoing, indicate "Ongoing." If there is a cap mentioned (either a percentage or a specific amount), extract that information.
                    - Insurance: Extract types of insurance required for any liability or coverage process and give it in the form {Insurance type: amount} for all the insurance coverage. Dont give detail for each insaurance type just list them as 'insurance type: amount' seperated by comma. If insaurance is not mentioned in the context then give 'Not provided' or N/A.
                    - Termination for convenience: Check if supplier can terminate for convenience and provide "Yes ,{clause mentioned}" else summarize the 'Terms and Termination' clause regarding the ability of the service provider or customer to terminate for convenience (max 30 words). If not provided, state 'Not available in the context'.
                    - Jurisdiction: Extract the Jurisdiction place (country and city) provided in the context."""
   
   
    table_response= st.session_state.faq_conversation_s({'question': table_question})
    print("--------------Table---------------")
    if table_response['chat_history']:
        table_answer = table_response['chat_history'][-1].content
        pdf_faq[table_question] = table_answer
        # responses += answer + "\n"
    else:
        st.write("A: No answer found")
 
    faq_data[pdf_name] = pdf_faq
   
 
    # Extracted details
    # client = pdf_faq.get("Client: Extract the 'Customer's name' or 'Buyer's name' or 'First Party' name, which is name of party signing the contract with mastek.", "Not available")
    # liability = pdf_faq.get("Liability: Extract this values from 'LIMITATION OF LIABILITY' clause only, Determine if liabilities for the services provided are capped at a specified (amount or percentage given under 'LIMITATION OF LIABILITY' clause only) of the Work Order value, If capped, extract the payable 'value' from 'limitation of liability' clause only and display it as 'Capped at {given value} of {given condition}', If value is not provided then display 'UNCAPPED'.", "Not available")
    # uncapped_liabilities = pdf_faq.get("Uncapped Liabilities: if Liability is 'UNCAPPED' then this field will be 'N/A' else list the types of liabilities that are not capped (e.g., death, personal injury, fraud, breach of law, etc.) Only list them, dont give whole phrases.", "Not available")
    # warranty = pdf_faq.get("Warranty: In the 'Warranty' section, if the word 'except' is used, extract the details; otherwise, mark it as 'Standard'. If section is not there then give 'Not available in the contract'.", "Not available")
    # indemnity = pdf_faq.get("Indemnity: In the 'Indemnity' section, if the word 'except' is used, extract the details; otherwise, mark it as 'Standard'. If section is not there then give 'Not available in the contract'.", "Not available")
    # services_damages = pdf_faq.get("Services/Damages: If the damages are ongoing, indicate 'Ongoing.' If there is a cap mentioned (either a percentage or a specific amount), extract that information.", "Not available")
    # insurance = pdf_faq.get("Insurance: Extract types of insurance required for any liability or coverage process and give it in the form 'Insurance type: amount' for all the insurance coverage. Dont give detail for each insurance type. If insurance is not mentioned in the context then give 'Not provided' or N/A.", "Not available")
    # ability_to_exit = pdf_faq.get("Ability to Exit: Check the 'Ability to Exit' or 'Termination Clause' sections and extract whether the service provider can terminate the agreement and under which condition. If not provided than give 'Not available in the context'.", "Not available")
    # jurisdiction = pdf_faq.get("Jurisdiction: Extract the Jurisdiction place (country and city) provided in the context.", "Not available")
   
    # # Combine answers into a DataFrame
    # data = {
    #     'Client': [client],
    #     'Liability': [liability],
    #     'Uncapped Liabilities': [uncapped_liabilities],
    #     'Warranty': [warranty],
    #     'Indemnity': [indemnity],
    #     'Services/Damages': [services_damages],
    #     'Insurance': [insurance],
    #     'Ability to Exit': [ability_to_exit],
    #     'Jurisdiction': [jurisdiction]
    # }
   
    # df = pd.DataFrame(data)
    # print(df)
    # pdf_faq = df_to_markdown(df)
 
# Print or use the Markdown output
    # print(pdf_faq)
 
# Store DataFrame in session state
    # faq_data[pdf_name] = pdf_faq
 
    # global faq_data
    # faq_data[pdf_name] = {}
 
    # for category, questions in faq_questions.items():
    #     pdf_faq = {}
    #     for question in questions:
    #         response = st.session_state.faq_conversation({'question': question})
    #         if response['chat_history']:
    #             answer = response['chat_history'][-1].content
    #             pdf_faq[question] = answer
    #         else:
    #             st.write("A: No answer found")
    #     faq_data[pdf_name] = pdf_faq
 
 
# Function to handle user input
def user_input(user_q):
   
    response = st.session_state.conversation({'question': user_q})
    st.session_state.chatHistory = response['chat_history']
    db = load_vector_store()
    answer = response['answer']
    docs_and_scores = db.similarity_search_with_score(answer)
    first_result = docs_and_scores[0]
    print("here7")
    document = first_result[0]
    print("here8")
    similarity_score = first_result[1]
    print("similarity score")
    print(similarity_score)
 
       
    if 'messages' not in st.session_state:
        st.session_state.messages = []
 
    st.session_state.messages.append(answer)
 
 
    st.rerun()
 
def save_processed_data(text_chunks):
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(text_chunks, f)
 
 
# Function to load processed text data
def load_processed_data():
    if os.path.exists('processed_data.pkl'):
        with open('processed_data.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return None
 
   
 
# Function to load and resize images
def load_image(image_path, size=(50,50)):
    image = Image.open(image_path)
    image = image.resize(size)
    return image
 
 
# Load images
human_image = load_image("human_icon.png", size=(100,100))
chatgpt_image = load_image("bot.png", size=(100,100))
 
 
# List of financial terms and their meanings
financial_terms = {
    "consideration": "Something of value exchanged for a promise or performance in a contract.",
    "force majeure clause": "A provision in a contract that excuses one or both parties from performance if certain unforeseen events occur, such as natural disasters or acts of war."
    # Add more terms as needed
}
 
 
# Function to check for financial terms in the response
def underline_financial_terms(response):
    for term, meaning in financial_terms.items():
        if term in response:
            response = response.replace(term, f'<abbr title="{meaning}">{term}</abbr>')
    return response
 
 
# Function to convert DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
 
# Function to save CSV to a file
def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
 
# Function to parse the prompt output into a DataFrame
# Function to parse the prompt output into a DataFrame
def parse_output_to_df(output):
    try:
        unwanted_substrings = ['<|im_end|>']
        for substring in unwanted_substrings:
            output = output.replace(substring, '')
        # Split the input into rows and clean up
        rows = [row.strip().split("|")[1:-1] for row in output.strip().split("\n") if row.strip() and not all(cell.strip() == '---' for cell in row.split("|")[1:-1])]
       
        # Check if rows are empty
        if not rows:
            raise ValueError("No valid data found")
       
        # Ensure all rows have the same number of columns
        max_columns = len(rows[0])
        for row in rows:
            if len(row) < max_columns:
                row.extend([''] * (max_columns - len(row)))
            elif len(row) > max_columns:
                raise ValueError(f"Row has more columns than expected: {row}")
 
        columns = rows[0]
        data = rows[1:]
 
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
 
        # Remove unwanted columns
 
        return df
   
    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
# Main function
def main():
   
    st.header("Contracts Analyzer 📃")
 
    # Check session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if 'chatHistory' not in st.session_state:
        st.session_state.chatHistory = None
    if "faq_data" not in st.session_state:
        st.session_state.faq_data = {}
    if "faq_conversation" not in st.session_state:
        st.session_state.faq_conversation= {}
 
 
 
    # Sidebar for settings
    st.sidebar.title("Settings")
    st.sidebar.subheader("Upload and Process PDFs")
 
    # Upload PDFs
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)
 
    # Process uploaded PDFs
    if pdf_docs and st.sidebar.button("Process PDFs"):
        with st.spinner("Processing PDFs"):
            for pdf in pdf_docs:
                pdf_name = os.path.basename(pdf.name)
                st.write(process_pdf(pdf))
               
                generate_faqs(pdf_name)
                st.session_state.faq_data = faq_data
 
    # Upload Scanned PDFs
    pdf_docs = st.sidebar.file_uploader("Upload Scanned PDF Files", accept_multiple_files=True)
    # Process uploaded PDFs
    if pdf_docs and st.sidebar.button("Process PDFs"):
        with st.spinner("Processing PDFs"):
            for pdf in pdf_docs:
                pdf_name = os.path.basename(pdf.name)
                st.write(process_scanned_pdf(pdf))
               
                generate_faqs(pdf_name)
                st.session_state.faq_data = faq_data
               
 
# Display FAQs if available
    if st.session_state.faq_data:
    # Do something
        st.subheader("Frequently asked questions")
        print("IN faq")
        for pdf_name, pdf_faq in st.session_state.faq_data.items():
            st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
            st.markdown(f"<h4>{pdf_name}</h4>", unsafe_allow_html=True)  # Display PDF name
            for question, answer in pdf_faq.items():
                # if "Q:" in answer:
                #     qa_pairs = answer.split("Q:")
                #     for qa_pair in qa_pairs[1:]:
                #         q_a_pair = qa_pair.split("A:")
                #         if len(q_a_pair) >= 2:
                #             question = q_a_pair[0].strip().replace("**", "")
                #             answer = q_a_pair[1].strip().replace("**", "")
                #             st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; font-weight: bold;">Q: {question}</p>', unsafe_allow_html=True)
                #             st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">A: {answer}</p>', unsafe_allow_html=True)
                           
                # else:
                st.markdown(f"<h6>MetaData</h6>", unsafe_allow_html=True)
                if "|" in answer and "---" in answer:  
                    # Parse the prompt output to DataFrame
                    print(answer)
                    df = parse_output_to_df(answer)
                    print("INSIDEEEE")
                   
                    st.write(df)
                    csv = convert_df_to_csv(df)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name=f'{pdf_name}_quaterly_report.csv',
                        mime='text/csv',
                    )
                else:
                    st.markdown(f"<h6>{question}</h6>", unsafe_allow_html=True)
                    st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;"> {answer.replace("- ", "<br>&nbsp;&nbsp;&nbsp;&nbsp;- ")} </p>', unsafe_allow_html=True)

 
# Handle user input
 
    if st.session_state.chatHistory:
        idx = 0
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                col1, col2 = st.columns([1, 8])
                with col1:
                    st.image(human_image, width=40)
                with col2:
                    st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-align: left;">{message.content}</p>', unsafe_allow_html=True)
            else:
                if "|" in message.content and "---" in message.content:
                    # If the response contains table formatting, format it into a proper table
                    rows = [row.split("|") for row in message.content.split("\n") if row.strip()]
                   
                    # Remove empty first and last columns
                    if rows and len(rows[0]) > 2:
                        rows = [row[1:-1] for row in rows]
 
                    # Remove rows filled with '--- --- ---'
                    character = '-'
 
                    # Filter out rows where all cells contain only the specified character
                    rows = [row for row in rows if not all(cell.strip() == character * len(cell.strip()) for cell in row)]
 
                    if len(rows) > 1 and all(len(row) == len(rows[0]) for row in rows):
                        # Check for empty column names and replace them with a default name
                        columns = [col.strip() if col.strip() else f"Column {i+1}" for i, col in enumerate(rows[0])]
                        df = pd.DataFrame(rows[1:], columns=columns)
                        st.write(df)
                   
                       
                        save_to_csv(df)
                       
                        # idx= idx+1
                        st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
 
                    else:
                        col1, col2 = st.columns([1, 8])
                        with col1:
                            st.image(chatgpt_image)
                        with col2:
                           
                            # st.write(message.content)
                            response_with_underline = underline_financial_terms(message.content)
                            st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{response_with_underline}</p>', unsafe_allow_html=True)
                   
                        st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
 
                else:
                    col1, col2 = st.columns([1, 8])
                    with col1:
                        st.image(chatgpt_image)
                    with col2:
                       
                        response_with_underline = underline_financial_terms(message.content)
                        st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{response_with_underline}</p>', unsafe_allow_html=True)
                   
                    if st.button(f"Text to Speech {i}"):  
                        print("CLICKED")
                        text_to_speech_thread(st.session_state.messages[i//2])
                    st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
                   
    user_question = st.text_input("Ask a Question from the PDF Files", key="pdf_question")
 
    if st.button("Get Response") and user_question :
        user_q =  user_question
        user_input(user_q)
       
 
if __name__ == "__main__":
    main()