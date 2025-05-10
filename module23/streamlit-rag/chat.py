# chat.py
import streamlit as st
import time
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import retrieval

st.set_page_config(page_title="RAG GPT", layout="wide", initial_sidebar_state="collapsed")

if "embed_model" not in st.session_state:
    st.session_state['embed_model'] = SentenceTransformer('BAAI/bge-base-en-v1.5')


st.title("RAG GPT with Savio Saldanha")

# Set a default model
if "openai_model" not in st.session_state:
    # Set OpenAI API key from Streamlit secrets
    st.session_state['openai_client'] = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    st.session_state["openai_model"] = "gpt-4o-mini" #-2024-07-18

with st.spinner("Starting up"):
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.HttpClient(host= "localhost", port = 8000)


    with st.spinner("Loading documents..."):
        try:
            # Get a collection object from an existing collection, by name. Will raise an exception if it's not found.
            collection_retrieved = st.session_state.chroma_client.get_collection(name='contracts')
        except: # when error, then load the docs
            st.error("There was an error in loading the documents. Please reload the page and retry.")
            st.stop()

    # required to show the dataframe on the right side. otherwise retrieved_results errors out as not defined
    if "retrieved_results" not in st.session_state:
        st.session_state.retrieved_results = []


if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, message in enumerate(st.session_state.messages):
    if message['role'] == 'system':
        continue # do not display system messages

    with st.chat_message(message["role"]):
        # check if the message is a dataframe. if so, display it with links. if not done, the links full urls are displayed
        if isinstance(message["content"], pd.DataFrame):
            st.data_editor(
                        message["content"],
                        column_config={
                            "Link": st.column_config.LinkColumn(
                                "Link",
                                help="Go to the page in the PDF",
                                max_chars=100,
                                display_text="Open"
                            )
                        },
                        hide_index=True, use_container_width=True, disabled=True,
                        key=idx)

        else:
            message["content"]

question_selected = st.sidebar.selectbox("Ask a question", options=[
    '',
    'Give me all the commercial terms of the contract with Electrofast',
    'Do we have any joint ventures?',
    'Who were the parties to the contract for our website SEO?'])

if prompt := st.chat_input("Ask me questions") or question_selected:
    # st.session_state.messages.append({"role": "user", "content": prompt})
    if prompt == "" and question_selected != "":
        prompt = question_selected

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Encoding"):
        # embed the user's query
        data = {"text_to_vectorize": [prompt]}
        prompt_vector = st.session_state['embed_model'].encode(prompt).tolist()

    with st.spinner("Retrieving results"):
        top_percentile_retrieved = retrieval.get_similar_documents(collection_retrieved, prompt_vector) # double bracket list [[]]
        print("top_percentile_retrieved: ", top_percentile_retrieved)
        context = f"""Use the context and respond to the user's query:
                        CONTEXT:
                        {". ".join(top_percentile_retrieved["documents"][0])}"""
                        #{st.session_state.retrieved_results['documents'][0][0]}""" # get the top 1 result

    with st.spinner("Generating a response"):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            system_instructions = f"""You are a helpful, respectful, and honest assistant. Always answer as helpfully. Be brief. Only answer with information in the context.
            {context}
            """

            st.session_state.messages.append({"role": "system", "content": system_instructions})
            st.session_state.messages.append({"role": "user", "content": prompt})

            stream = st.session_state.openai_client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if not isinstance(m["content"], pd.DataFrame)
            ],
            )
            full_response = stream.choices[0].message.content
            print(full_response)

            output_printed = ""
            for chunk in full_response.split():
                output_printed += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(output_printed + "▌")
            message_placeholder.markdown(full_response)


        st.session_state.messages.append({"role": "assistant",
                                          "content": full_response})
