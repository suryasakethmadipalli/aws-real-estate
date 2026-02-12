%%writefile app.py
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA

# PAGE CONFIG
st.set_page_config(page_title="AI Real Estate Agent", page_icon="🏠")
st.title("🏠 AI Real Estate Agent")

# 1. SETUP CREDENTIALS
# We look for secrets in Environment Variables (AWS) first, then Streamlit secrets
try:
    if "GOOGLE_API_KEY" in os.environ:
        pass # AWS handles this automatically
    else:
        # Fallback for local testing if needed
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        os.environ["DB_HOST"] = st.secrets["DB_HOST"]
        os.environ["DB_USER"] = st.secrets["DB_USER"]
        os.environ["DB_PASS"] = st.secrets["DB_PASS"]
        os.environ["DB_NAME"] = st.secrets["DB_NAME"]
except Exception as e:
    st.warning("⚠️ Using Environment Variables for Credentials.")

# 2. DEFINE THE TOOLS
@st.cache_resource
def get_agent_tools():
    tools = []
    
    # --- TOOL 1: SQL DATABASE ---
    try:
        # Get credentials from environment
        db_user = os.environ.get("DB_USER")
        db_pass = os.environ.get("DB_PASS")
        db_host = os.environ.get("DB_HOST")
        db_name = os.environ.get("DB_NAME")
        
        uri = f"postgresql://{db_user}:{db_pass}@{db_host}:5432/{db_name}"
        db = SQLDatabase.from_uri(uri)
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
        
        sql_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
        
        sql_tool = Tool(
            name="Sales Database",
            func=sql_chain.run,
            description="Useful for answering questions about house prices, sales data, agents, or database records."
        )
        tools.append(sql_tool)
    except Exception as e:
        print(f"SQL Tool Error: {e}")

    # --- TOOL 2: PDF SEARCH ---
    pdf_file = "policy.pdf" 
    if os.path.exists(pdf_file):
        try:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(pages, embeddings)
            retriever = vector_store.as_retriever()
            
            rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            
            rag_tool = Tool(
                name="Document Search",
                func=rag_chain.run,
                description="Useful for answering questions about policies, hours, refunds, or contracts."
            )
            tools.append(rag_tool)
        except Exception as e:
            print(f"RAG Tool Error: {e}")
            
    return tools

# 3. INITIALIZE THE BRAIN
llm_agent = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
tools = get_agent_tools()

agent = initialize_agent(
    tools, 
    llm_agent, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True
)

# 4. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about houses or policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Checking..."):
            try:
                response = agent.run(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error("I'm having trouble connecting right now.")
