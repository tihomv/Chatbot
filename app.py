import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Set page config
st.set_page_config(page_title="Pizza Restaurant Chatbot", page_icon="🍕")

# App title and description
st.title("🍕 Pizza Restaurant Chatbot")
st.markdown("Ask questions about our pizza restaurant based on customer reviews.")

# Initialize model and chain
@st.cache_resource
def load_chain():
    model = OllamaLLM(model="llama3.2")
    template = """
    You are an expert in answering questions about a pizza restaurant

    Here are some relevant reviews: {reviews}

    Here is the question to answer: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

chain = load_chain()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask a question about our pizza restaurant"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get relevant reviews
            reviews = retriever.invoke(question)
            
            # Show reviews in an expander
            with st.expander("Relevant reviews used"):
                st.write(reviews)
            
            # Generate response
            response = chain.invoke({"reviews": reviews, "question": question})
            
            # Display response
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a way to clear chat history
# if st.button("Clear Chat History"):
#     st.session_state.messages = []
#     st.rerun() 