import os
import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# =====================================
# APP CONFIG
# =====================================

st.set_page_config(
    page_title="CyberlawGPT",
    page_icon="⚖️",
    layout="wide"
)


st.title("⚖️ CyberlawGPT")

st.write(
    "AI Legal Assistant based on Pakistan Cyber Laws (PECA)"
)



# =====================================
# GROQ API KEY
# =====================================


def get_api_key():

    try:
        return st.secrets["GROQ_API_KEY"]

    except Exception:

        return os.getenv(
            "GROQ_API_KEY"
        )



GROQ_API_KEY = get_api_key()



if not GROQ_API_KEY:

    st.error(
        """
        ❌ Groq API Key Missing.

        Add your key in:

        Streamlit Cloud
        → Settings
        → Secrets


        Example:

        GROQ_API_KEY="gsk_xxxxxxxxx"
        """
    )

    st.stop()



client = Groq(
    api_key=GROQ_API_KEY
)



# =====================================
# MODEL SETTINGS
# =====================================


MODEL_NAME = "openai/gpt-oss-120b"



# =====================================
# SIDEBAR SETTINGS
# =====================================


st.sidebar.header(
    "⚙️ Response Settings"
)



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

    "Show Retrieved Legal Text",

    True

)




# =====================================
# PDF LOCATION
# =====================================


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)



PDF_PATH = os.path.join(

    BASE_DIR,

    "1470910659_707.pdf"

)




if not os.path.exists(PDF_PATH):

    st.error(
        f"""
        ❌ PECA PDF file missing.

        Required file:

        {PDF_PATH}


        Upload:

        1470910659_707.pdf

        beside app.py in GitHub.
        """
    )

    st.stop()



# =====================================
# PDF EXTRACTION
# =====================================


@st.cache_resource
def load_pdf_text():

    doc = fitz.open(
        PDF_PATH
    )


    text = ""


    for page in doc:

        text += page.get_text()



    return text




# =====================================
# FAISS VECTOR DATABASE
# =====================================


@st.cache_resource
def create_vector_store():


    text = load_pdf_text()



    chunks=[]


    chunk_size = 1200



    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunks.append(
            text[i:i+chunk_size]
        )



    embedding_model = SentenceTransformer(

        "sentence-transformers/all-MiniLM-L6-v2"

    )



    embeddings = embedding_model.encode(

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



    return index, chunks, embedding_model




index, chunks, embedding_model = create_vector_store()




# =====================================
# RETRIEVE CONTEXT
# =====================================


def retrieve_context(question):


    vector = embedding_model.encode(

        [question]

    )



    distance, ids = index.search(

        np.array(vector),

        5

    )



    context=[]



    for i in ids[0]:

        context.append(
            chunks[i]
        )



    return "\n\n".join(context)




# =====================================
# GROQ GENERATION
# =====================================


def generate_answer(question, context):


    prompt = f"""

You are CyberlawGPT.

You are an AI assistant specialized in Pakistan Cyber Laws.

Answer ONLY from the provided PECA document context.

Never create fake legal sections.

If information is unavailable:

Say:
"Information not found in provided PECA document."


QUESTION:

{question}


LEGAL CONTEXT:

{context}



Answer Settings:

Technical Level:
{technical_level}


Response Size:
{response_length}


Language:
{language}


Answer format:

1. Relevant PECA Section
2. Legal Explanation
3. Punishment (if mentioned)
4. Practical guidance

"""



    try:


        response = client.chat.completions.create(

            model=MODEL_NAME,


            messages=[

                {

                    "role":"system",

                    "content":prompt

                }

            ],


            temperature=0.1,


            max_tokens=1500

        )



        return response.choices[0].message.content



    except Exception as e:


        return f"""

❌ Groq API Error

{str(e)}


Check:

1. Groq API key
2. Model availability
3. Streamlit Secrets

"""




# =====================================
# EXAMPLE QUESTIONS
# =====================================


st.subheader(
    "💡 Sample Questions"
)



examples=[

"Unauthorized access punishment under PECA?",

"What is cyber stalking under PECA?",

"What is electronic fraud?",

"What happens if someone shares private pictures?",

"What are investigation powers under PECA?"

]



cols = st.columns(2)



for i,q in enumerate(examples):

    if cols[i%2].button(q):

        st.session_state.question=q




# =====================================
# USER QUESTION
# =====================================


question = st.text_input(

    "Ask your Cyber Law Question",

    value=st.session_state.get(
        "question",
        ""
    )

)



if st.button(
    "⚖️ Generate Answer"
):


    if question.strip()=="":

        st.warning(
            "Please enter your question."
        )


    else:


        with st.spinner(
            "Searching PECA database..."
        ):


            context = retrieve_context(
                question
            )


            answer = generate_answer(

                question,

                context

            )



        st.subheader(
            "⚖️ CyberlawGPT Answer"
        )


        st.write(answer)



        if show_sources:


            with st.expander(
                "📚 Retrieved PECA Context"
            ):

                st.write(context)
