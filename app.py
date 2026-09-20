import os
import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# APP CONFIG
# -----------------------------

st.set_page_config(
    page_title="CyberlawGPT",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ CyberlawGPT")
st.caption(
    "AI Legal Assistant based on Pakistan Cyber Laws (PECA)"
)


# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------

st.sidebar.header("⚙️ Response Settings")


technical_level = st.sidebar.selectbox(
    "Technical Level",
    [
        "Beginner",
        "Intermediate",
        "Expert / Legal Professional"
    ]
)


response_length = st.sidebar.selectbox(
    "Response Size",
    [
        "Short",
        "Medium",
        "Detailed"
    ]
)


language = st.sidebar.selectbox(
    "Answer Language",
    [
        "English",
        "Urdu",
        "Roman Urdu"
    ]
)


show_sources = st.sidebar.checkbox(
    "Show Retrieved Legal Sections",
    True
)


# -----------------------------
# GROQ API
# -----------------------------

groq_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)


if not groq_key:
    st.warning(
        "Enter your Groq API key from sidebar."
    )
    st.stop()


client = Groq(
    api_key=groq_key
)


# -----------------------------
# LOAD PDF
# -----------------------------

PDF_FILE = "1470910659_707.pdf"


@st.cache_resource
def load_document():

    doc = fitz.open(PDF_FILE)

    text = ""

    for page in doc:
        text += page.get_text()

    return text



@st.cache_resource
def create_vector_database():

    text = load_document()


    chunks = []

    size = 900

    for i in range(0, len(text), size):

        chunks.append(
            text[i:i+size]
        )


    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )


    dimension = embeddings.shape[1]


    index = faiss.IndexFlatL2(
        dimension
    )


    index.add(
        embeddings
    )


    return index, chunks, model



index, chunks, embed_model = create_vector_database()



# -----------------------------
# RETRIEVAL
# -----------------------------

def retrieve_context(question):

    query_vector = embed_model.encode(
        [question]
    )


    distance, ids = index.search(
        np.array(query_vector),
        5
    )


    results=[]


    for i in ids[0]:

        results.append(
            chunks[i]
        )


    return "\n\n".join(results)



# -----------------------------
# GROQ RESPONSE
# -----------------------------

def generate_answer(question, context):


    prompt=f"""

You are CyberlawGPT.

You are an AI assistant specialized in Pakistan cyber laws.

Answer only according to the provided legal document context.

Do not invent sections, punishments, or legal advice.

If information is unavailable in the document,
clearly say:

"Information not found in provided PECA document."

User question:

{question}


Legal Context:

{context}


Response Requirements:

Technical Level:
{technical_level}

Response Length:
{response_length}

Language:
{language}


Explain clearly.

Mention relevant sections whenever available.

"""



    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role":"system",
                "content":prompt
            }
        ],

        temperature=0.1

    )


    return response.choices[0].message.content



# -----------------------------
# SAMPLE QUESTIONS
# -----------------------------


st.subheader("💡 Sample Questions")


samples=[

"Unauthorized access to someone's account punishment?",

"What is cyber stalking under PECA?",

"What punishment exists for electronic fraud?",

"Can someone share private pictures without permission?",

"What are investigation powers under PECA?"

]


cols=st.columns(2)


for i,q in enumerate(samples):

    if cols[i%2].button(q):

        st.session_state.question=q



# -----------------------------
# USER INPUT
# -----------------------------


question = st.text_input(
    "Ask your Cyber Law Question",
    value=st.session_state.get(
        "question",
        ""
    )
)



if st.button("Generate Answer"):


    if question.strip():

        with st.spinner(
            "Analyzing Cyber Law..."
        ):


            context = retrieve_context(
                question
            )


            answer = generate_answer(
                question,
                context
            )


        st.subheader("⚖️ CyberlawGPT Response")

        st.write(answer)


        if show_sources:

            st.subheader(
                "📚 Retrieved Legal Context"
            )

            st.write(context)


