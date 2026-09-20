import os
import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="CyberlawGPT",
    page_icon="⚖️",
    layout="wide"
)


st.title("⚖️ CyberlawGPT")

st.write(
    "AI Legal Assistant based on Pakistan Cyber Laws (PECA)"
)


# ==========================================
# SECURE GROQ API KEY
# ==========================================

def get_api_key():

    # Streamlit Cloud
    try:
        return st.secrets["GROQ_API_KEY"]

    except Exception:
        pass


    # Local / Colab
    return os.getenv(
        "GROQ_API_KEY"
    )



GROQ_API_KEY = get_api_key()


if not GROQ_API_KEY:

    st.error(
        """
        Groq API Key Missing.

        Streamlit Cloud:
        Settings → Secrets

        Add:

        GROQ_API_KEY="your_key_here"
        """
    )

    st.stop()



client = Groq(
    api_key=GROQ_API_KEY
)



# ==========================================
# SIDEBAR SETTINGS
# ==========================================


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



response_size = st.sidebar.selectbox(
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




# ==========================================
# PDF LOCATION FIX
# ==========================================


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


PDF_FILE = os.path.join(
    BASE_DIR,
    "1470910659_707.pdf"
)




# ==========================================
# LOAD PDF
# ==========================================


@st.cache_resource
def load_pdf_text():


    if not os.path.exists(PDF_FILE):

        st.error(
            f"""
            ❌ PECA PDF file not found.

            Expected location:

            {PDF_FILE}

            Upload:

            1470910659_707.pdf

            in the same folder as app.py
            """
        )

        st.stop()



    doc = fitz.open(
        PDF_FILE
    )


    text = ""


    for page in doc:

        text += page.get_text()



    return text




# ==========================================
# CREATE VECTOR DATABASE
# ==========================================


@st.cache_resource
def build_vector_store():


    text = load_pdf_text()


    chunks=[]


    chunk_size = 1000


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



    vectors = embedding_model.encode(

        chunks,

        convert_to_numpy=True

    )



    dimension = vectors.shape[1]



    index = faiss.IndexFlatL2(
        dimension
    )


    index.add(
        vectors
    )


    return index, chunks, embedding_model





index, chunks, embedding_model = build_vector_store()



# ==========================================
# SEARCH DOCUMENT
# ==========================================


def retrieve_context(question):


    query_vector = embedding_model.encode(

        [question]

    )


    distances, ids = index.search(

        np.array(query_vector),

        5

    )


    context=[]


    for idx in ids[0]:

        context.append(
            chunks[idx]
        )



    return "\n\n".join(context)




# ==========================================
# GROQ GENERATION
# ==========================================


def generate_response(
        question,
        context
):


    prompt=f"""

You are CyberlawGPT.

You answer questions only according to Pakistan cyber law document provided.

Do not invent laws.

If information is not available,
say:

"Information not found in provided PECA document."


Question:

{question}



Legal Context:

{context}



Answer Settings:

Technical Level:
{technical_level}


Response Length:
{response_size}


Language:
{language}



Include:

- Relevant PECA Section
- Explanation
- Punishment (if mentioned)
- Practical guidance

"""


    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[

            {
                "role":"user",
                "content":prompt
            }

        ],

        temperature=0.1

    )



    return response.choices[0].message.content





# ==========================================
# SAMPLE QUESTIONS
# ==========================================


st.subheader(
    "💡 Sample Questions"
)



samples=[

"Unauthorized access punishment under PECA?",

"What is cyber stalking?",

"What is electronic fraud?",

"What happens if someone shares private photos?",

"What are investigation powers under PECA?"

]



cols=st.columns(2)



for i,q in enumerate(samples):

    if cols[i%2].button(q):

        st.session_state.question=q





# ==========================================
# USER INPUT
# ==========================================


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


    if question:


        with st.spinner(
            "Searching PECA law..."
        ):


            context = retrieve_context(
                question
            )


            answer = generate_response(
                question,
                context
            )



        st.subheader(
            "⚖️ CyberlawGPT Answer"
        )


        st.write(answer)



        if show_sources:


            with st.expander(
                "📚 Retrieved Legal Text"
            ):

                st.write(context)



    else:


        st.warning(
            "Please enter a question."
        )
