import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Load PDF and embed
@st.cache_resource
def load_pdf_vectorstore():
    loader = PyPDFLoader("stats.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db

# Load QA chain using vectorstore
@st.cache_resource
def get_qa_chain(_llm):
    db = load_pdf_vectorstore()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(llm=_llm, retriever=retriever, return_source_documents=True)

# External search tools
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
search = DuckDuckGoSearchRun(name="Search")

# Streamlit UI
st.set_page_config(page_title="📘 Educational Chatbot")
st.title("📘 Educational Chatbot with PDF Knowledge & Web Tools")

# Sidebar for API key
st.sidebar.title("🔐 API Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    st.warning("Please enter your Groq API Key to continue.")
    st.stop()
else:
    st.session_state["api_key"] = api_key

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm EduBot! I specialize in Statistics and general knowledge. Ask me anything!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Prompt formatting
persona_template = """You are EduBot, an educational chatbot who gives clear and direct answers.
Focus on accuracy and helpfulness.
User question: {question}
"""
persona_prompt = PromptTemplate(input_variables=["question"], template=persona_template)

# Handle chat input
if prompt := st.chat_input("Ask your educational question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

    with st.chat_message("assistant"):
        # Step 1: Try answering from PDF
        qa_chain = get_qa_chain(_llm=llm)
        pdf_result = qa_chain(prompt)
        pdf_answer = pdf_result["result"]
        pdf_sources = pdf_result.get("source_documents", [])

        # Determine if PDF is usable
        def is_good_answer(text):
            if not text or len(text.split()) < 20:
                return False
            bad_phrases = [
                "i don't know", "not enough information", "could not find", 
                "i cannot answer", "thank you for explaining", "i'm not the one who"
            ]
            return not any(bad in text.lower() for bad in bad_phrases)

        if is_good_answer(pdf_answer):
            final_response = pdf_answer
            source_texts = [doc.page_content for doc in pdf_sources]
            final_source = "📄 Source: Internal PDF knowledge base."
        else:
            # Step 2: Fall back to external tools
            tools = [search, arxiv, wiki]
            llm_chain = LLMChain(llm=llm, prompt=persona_prompt)
            improved_prompt = llm_chain.run(question=prompt)

            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                verbose=False,
                max_iterations=4,
                early_stopping_method="generate"
            )

            try:
                final_response = agent.run(improved_prompt, callbacks=[st_cb])
                final_source = "🌐 Source: Retrieved via external search tools (Wikipedia, Arxiv, DuckDuckGo)."
            except Exception as e:
                final_response = "Sorry, I couldn't find an answer. Please rephrase your question."
                final_source = f"(Error: {e})"

        # Display response + source
        st.session_state.messages.append({"role": "assistant", "content": f"{final_response}\n\n{final_source}"})
        st.write(final_response)
        st.caption(final_source)
