import os
import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# ======================================
# APP CONFIGURATION
# ======================================

st.set_page_config(
    page_title="CyberlawGPT",
    page_icon="⚖️",
    layout="wide"
)


st.title("⚖️ CyberlawGPT")

st.markdown(
    """
    ### AI Legal Assistant based on Pakistan Cyber Laws (PECA)
    
    Ask questions related to Pakistan's cyber laws.
    Answers are generated using Retrieval Augmented Generation (RAG).
    """
)



# ======================================
# LOAD GROQ API KEY SECURELY
# ======================================


def get_groq_key():

    # Streamlit Cloud Secret
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]


    # Optional Colab/local testing
    if "GROQ_API_KEY" in os.environ:
        return os.environ["GROQ_API_KEY"]


    return None



groq_key = get_groq_key()


if not groq_key:

    st.error(
        """
        ❌ Groq API Key not found.

        For Streamlit Cloud:
        Add it in:

        App Settings → Secrets

        Example:

        GROQ_API_KEY="your_api_key_here"
        """
    )

    st.stop()



client = Groq(
    api_key=groq_key
)



# ======================================
# SIDEBAR SETTINGS
# ======================================


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

    "Show Retrieved Legal Sections",

    value=True

)




# ======================================
# PDF LOCATION
# ======================================


PDF_FILE = "1470910659_707.pdf"




# ======================================
# READ PDF
# ======================================


@st.cache_resource

def load_pdf():


    if not os.path.exists(PDF_FILE):

        st.error(
            "PECA PDF file not found."
        )

        st.stop()



    document = fitz.open(
        PDF_FILE
    )


    complete_text = ""


    for page in document:

        complete_text += page.get_text()



    return complete_text





# ======================================
# CREATE VECTOR DATABASE
# ======================================


@st.cache_resource

def create_faiss_index():


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



    embedding_model = SentenceTransformer(

        "sentence-transformers/all-MiniLM-L6-v2"

    )



    embeddings = embedding_model.encode(

        chunks,

        show_progress_bar=False,

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





index, chunks, embedding_model = create_faiss_index()




# ======================================
# RETRIEVAL FUNCTION
# ======================================


def retrieve_documents(question):


    question_embedding = embedding_model.encode(

        [question]

    )


    distances, results = index.search(

        np.array(question_embedding),

        5

    )


    retrieved=[]



    for item in results[0]:

        retrieved.append(

            chunks[item]

        )



    return "\n\n".join(retrieved)




# ======================================
# GROQ RESPONSE
# ======================================


def ask_cyberlawgpt(question, context):


    prompt = f"""

You are CyberlawGPT.

You are a Pakistan Cyber Law assistant.

Use ONLY the provided legal context.

Do not create fake sections,
fake punishments, or unsupported legal claims.

If information is unavailable say:

"Information not found in provided PECA document."



User Question:

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



Always explain:

1. Relevant PECA section
2. Legal explanation
3. Punishment (if available)
4. Practical guidance


"""



    completion = client.chat.completions.create(

        model="llama-3.3-70b-versatile",


        messages=[

            {
                "role":"system",
                "content":prompt
            }

        ],


        temperature=0.1

    )


    return completion.choices[0].message.content





# ======================================
# SAMPLE QUESTIONS
# ======================================


st.subheader(
    "💡 Sample Questions"
)


examples=[


"Unauthorized access punishment under PECA?",


"What is cyber stalking?",


"What is electronic fraud?",


"What happens if someone shares private pictures?",


"What powers does investigation agency have?"



]


cols = st.columns(2)



for i,q in enumerate(examples):


    if cols[i%2].button(q):

        st.session_state.question=q




# ======================================
# USER QUERY
# ======================================



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
            "Analyzing PECA law..."
        ):


            context = retrieve_documents(

                question

            )


            answer = ask_cyberlawgpt(

                question,

                context

            )



        st.subheader(
            "⚖️ CyberlawGPT Answer"
        )


        st.write(answer)



        if show_sources:


            with st.expander(
                "📚 Retrieved Legal Sections"
            ):

                st.write(context)
