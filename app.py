import os
import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================
# PAGE SETTINGS
# =========================================

st.set_page_config(
    page_title="CyberlawGPT",
    page_icon="⚖️",
    layout="wide"
)


st.title("⚖️ CyberlawGPT")

st.write(
    "AI Legal Assistant based on Pakistan Cyber Laws (PECA)"
)


# =========================================
# GROQ API KEY FROM SECRETS
# =========================================


def get_api_key():

    try:
        return st.secrets["GROQ_API_KEY"]

    except Exception:
        return os.getenv("GROQ_API_KEY")



api_key = get_api_key()


if not api_key:

    st.error(
        """
        Groq API Key Missing.

        Add it in Streamlit Cloud:

        Settings → Secrets

        Example:

        GROQ_API_KEY="your_api_key"
        """
    )

    st.stop()



client = Groq(
    api_key=api_key
)




# =========================================
# SIDEBAR OPTIONS
# =========================================


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




# =========================================
# PDF LOCATION
# =========================================


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


PDF_FILE = os.path.join(
    BASE_DIR,
    "1470910659_707.pdf"
)



if not os.path.exists(PDF_FILE):

    st.error(
        f"""
        ❌ PECA PDF file not found.

        Looking here:

        {PDF_FILE}

        Upload 1470910659_707.pdf
        in your GitHub repository.
        """
    )

    st.stop()




# =========================================
# READ PDF
# =========================================


@st.cache_resource
def load_pdf():


    document = fitz.open(
        PDF_FILE
    )


    text = ""


    for page in document:

        text += page.get_text()


    return text





# =========================================
# CREATE FAISS DATABASE
# =========================================


@st.cache_resource
def create_database():


    text = load_pdf()



    chunks = []


    chunk_size = 1000


    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunks.append(
            text[i:i+chunk_size]
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




index, chunks, embedding_model = create_database()




# =========================================
# RETRIEVAL
# =========================================


def search_documents(question):


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





# =========================================
# GROQ ANSWER
# =========================================


def generate_answer(question, context):


    prompt=f"""

You are CyberlawGPT.

You answer questions according to Pakistan Cyber Law (PECA).

Use ONLY the provided legal context.

Do not invent sections or punishments.

If information is not available say:

"Information not found in provided PECA document."


QUESTION:

{question}


LEGAL CONTEXT:

{context}



Answer Settings:

Technical Level:
{technical_level}


Response Length:
{response_size}


Language:
{language}



Answer format:

1. Relevant PECA Section
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





# =========================================
# SAMPLE QUESTIONS
# =========================================


st.subheader(
    "💡 Sample Questions"
)



questions=[

"Unauthorized access punishment under PECA?",

"What is cyber stalking?",

"What is electronic fraud?",

"What happens if someone shares private pictures?",

"What powers does investigation agency have?"

]



columns = st.columns(2)



for i,q in enumerate(questions):

    if columns[i%2].button(q):

        st.session_state.question=q




# =========================================
# USER QUERY
# =========================================


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
            "Please enter a question."
        )


    else:


        with st.spinner(
            "Analyzing PECA..."
        ):


            context = search_documents(
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
