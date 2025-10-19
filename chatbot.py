import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = "chroma_db"

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="Yalda 2.0 - AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary-color: #6366f1;
        --secondary-color: #ec4899;
        --accent-color: #f97316;
        --dark-bg: #0f172a;
        --light-text: #f1f5f9;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stChatMessageContainer"] {
        background-color: transparent;
    }
    
    .stChatMessage {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    
    /* User message styling */
    [data-testid="stChatMessage"] > div:first-child {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%);
        border-left: 3px solid #6366f1;
    }
    
    /* Assistant message styling */
    [data-testid="stChatMessage"]:last-child > div:first-child {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        border-left: 3px solid #f97316;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    
    .stChatInput > div > div > input {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.3);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .tagline {
        color: #cbd5e1;
        font-size: 0.95rem;
        font-style: italic;
        margin-bottom: 1rem;
    }
    
    .unanswered-box {
        background: rgba(239, 68, 68, 0.1);
        border-left: 3px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .main-title {
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 50%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #475569, transparent);
        margin: 1.5rem 0;
    }
    
    .footer-caption {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

vector_store = Chroma(
    collection_name="recruiter_profile",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

retriever = vector_store.as_retriever(search_kwargs={'k': 5})

if "messages" not in st.session_state:
    st.session_state.messages = []

if "missing" not in st.session_state:
    st.session_state.missing = []

with st.sidebar:   
    st.markdown("<div style='text-align: center; padding: 1rem 0 1.5rem 0;'>", unsafe_allow_html=True)
    st.image('logo.png', width=80)
    st.markdown("<div class='sidebar-header' style='margin-top: 0.5rem;'>Yalda 2.0</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: rgba(99, 102, 241, 0.1); 
                    border-radius: 10px; 
                    padding: 1rem; 
                    margin-bottom: 1.5rem;
                    border: 1px solid rgba(99, 102, 241, 0.2);'>
            <div class='tagline' style='margin: 0; text-align: center;'>
                Skeptical about taking a chance on Yalda? Let me change your mind—or at least make a compelling case!
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.missing = []
        st.rerun()
    
    st.link_button("📧 Contact Yalda", "mailto:ynikooka@uwaterloo.ca", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Unanswered questions section
    if st.session_state.missing:
        st.markdown("""
            <div style='margin-top: 2rem;'>
                <div style='font-size: 1.1rem; 
                            font-weight: 600; 
                            color: #f97316; 
                            margin-bottom: 1rem;
                            display: flex;
                            align-items: center;'>
                    <span style='font-size: 1.3rem; margin-right: 0.5rem;'>❓</span>
                    Unanswered Questions
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        for i, q in enumerate(st.session_state.missing, 1):
            st.markdown(f"""
                <div class='unanswered-box' style='margin-bottom: 0.75rem;'>
                    <div style='color: #f97316; font-weight: 700; margin-bottom: 0.25rem;'>
                        Question {i}
                    </div>
                    <div style='color: #cbd5e1; font-size: 0.9rem;'>
                        {q}
                    </div>
                </div>
            """, unsafe_allow_html=True)
st.markdown("<div class='main-title'>🤖 AI Resume Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='main-subtitle'>Ask me anything about Yalda's experience, skills, and projects</div>", unsafe_allow_html=True)

st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "✨"):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about skills, projects, experience...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    
    try:
        docs = retriever.invoke(user_input)
        knowledge = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        st.error(f"Error retrieving information: {e}")
        st.stop()
    
    if not knowledge.strip():
        st.session_state.missing.append(user_input)
        response_text = (
            "I'm sorry, I don't have enough information to answer that question. "
            "Please reach out to Yalda directly at **ynikooka@uwaterloo.ca**."
        )
    else:
        rag_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant representing Yalda Nikookar. Refer to me as Yalda. Be professional, friendly, and personable. Your goal is to answer recruiter questions about Yalda's background, skills, experience, and projects **using only the information provided**.  

Guidelines for your responses:

1. **Be Specific and Impact-Focused:** Highlight concrete projects, results, or measurable outcomes. Avoid vague praise or filler.  
2. **Avoid Formulaic Structure:** Use natural flow; don't repeat "She has X experience / Her skills include Y / She demonstrated Z".  
3. **Include Context and Impact:** Explain not just what Yalda did, but the outcome, problem solved, or value added.  
4. **Tailor to the Question:** Directly answer the recruiter's query, using relevant examples from Yalda's experience.  
5. **Active, Professional Call-to-Action:** Encourage follow-up by showing relevance to their problem, e.g., "Reach out to ynikooka@uwaterloo.ca to discuss specific projects or challenges."  
6. **Redirect Politely If Needed:** If asked about something unrelated to Yalda's professional background, gently steer back to her professional experience.

Question: {user_input}

Knowledge from Yalda's resume and experience:
{knowledge}

Answer using ONLY the information above. Provide clear examples, measurable impact, and why her experience is relevant. Be concise, professional, and personable. If they ask, encourage them to reach out to me directly via my email ynikooka@uwaterloo.ca 
""")

        chain = rag_prompt | llm
        
        with st.chat_message("assistant", avatar="✨"):
            message_placeholder = st.empty()
            full_response = ""
            
            for chunk in chain.stream({"user_input": user_input, "knowledge": knowledge}):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        response_text = full_response
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})

st.markdown("---")
st.markdown("<div class='footer-caption'>🔐 This app uses RAG to answer questions based on Yalda's provided information.</div>", unsafe_allow_html=True)