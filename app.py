import os
import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="CyberlawGPT",
    page_icon="⚖️",
    layout="wide"
)


st.title("⚖️ CyberlawGPT")

st.caption(
    "AI Legal Assistant based on Pakistan Cyber Laws (PECA)"
)



# =====================================
# GROQ API KEY FROM STREAMLIT SECRETS
# =====================================


def get_groq_key():

    try:
        return st.secrets["GROQ_API_KEY"]

    except Exception:

        return os.getenv(
            "GROQ_API_KEY"
        )



GROQ_API_KEY = get_groq_key()



if not GROQ_API_KEY:

    st.error(
        """
        Groq API Key Missing.

        Add it in Streamlit Cloud:

        Settings → Secrets

        Example:

        GROQ_API_KEY="your_key_here"
        """
    )

    st.stop()



client = Groq(
    api_key=GROQ_API_KEY
)



# =====================================
# SIDEBAR SETTINGS
# =====================================


st.sidebar.title(
    "⚙️ Settings"
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
    "Response Length",
    [
        "Short",
        "Medium",
        "Detailed"
    ]
)



language = st.sidebar.selectbox(
    "Language",
    [
        "English",
        "Urdu",
        "Roman Urdu"
    ]
)



show_sources = st.sidebar.checkbox(
    "Show Legal Context",
    value=True
)



# =====================================
# PDF PATH
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
        ❌ PDF not found.

        Searching:

        {PDF_PATH}

        Make sure the PDF is uploaded
        in GitHub beside app.py
        """
    )

    st.stop()




# =====================================
# LOAD PDF
# =====================================


@st.cache_resource
def extract_pdf_text():

    doc = fitz.open(
        PDF_PATH
    )


    text = ""


    for page in doc:

        text += page.get_text()



    return text




# =====================================
# CREATE FAISS VECTOR STORE
# =====================================


@st.cache_resource
def create_vector_database():


    text = extract_pdf_text()



    chunks = []

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



    return (
        index,
        chunks,
        embedding_model
    )




index, chunks, embedding_model = create_vector_database()




# =====================================
# RETRIEVAL
# =====================================


def retrieve_context(question):


    query_embedding = embedding_model.encode(
        [question]
    )



    distances, ids = index.search(
        np.array(query_embedding),
        5
    )



    results=[]


    for idx in ids[0]:

        results.append(
            chunks[idx]
        )


    return "\n\n".join(results)





# =====================================
# GROQ ANSWER
# =====================================


def generate_answer(
        question,
        context
):


    prompt=f"""

You are CyberlawGPT.

You answer questions about Pakistan Cyber Laws.

Use only the provided PECA document context.

Do not create fake legal sections.

If information is unavailable say:

"Information not found in provided PECA document."


Question:

{question}


Legal Context:

{context}



Answer Settings:

Technical Level:
{technical_level}


Response Length:
{response_length}


Language:
{language}



Answer Format:

1. Relevant Section
2. Explanation
3. Punishment (if mentioned)
4. Practical guidance


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





# =====================================
# SAMPLE QUESTIONS
# =====================================


st.subheader(
    "💡 Ask Examples"
)



samples=[

"Unauthorized access punishment under PECA?",

"What is cyber stalking?",

"What is electronic fraud?",

"What happens if someone shares private images?",

"What powers does investigation agency have?"

]



cols = st.columns(2)



for i,q in enumerate(samples):

    if cols[i % 2].button(q):

        st.session_state.question=q




# =====================================
# USER QUESTION
# =====================================


question = st.text_input(

    "Enter your cyber law question",

    value=st.session_state.get(
        "question",
        ""
    )

)




if st.button(
    "⚖️ Generate Answer"
):


    if question.strip():


        with st.spinner(
            "Searching PECA document..."
        ):


            context = retrieve_context(
                question
            )


            answer = generate_answer(
                question,
                context
            )



        st.subheader(
            "⚖️ CyberlawGPT Response"
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
