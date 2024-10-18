import streamlit as st
import os
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from utils.tools import book_appointment
from langchain.tools.retriever import create_retriever_tool

URL = os.getenv("UPSTASH_REDIS_REST_URL")
TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Initialize Langchain components
llm = ChatOpenAI(model="gpt-4o")

def get_session_history(session_id: str):
    return UpstashRedisChatMessageHistory(
        url=URL,
        token=TOKEN,
        ttl=3600,  # Set time-to-live in seconds (e.g., 1 hour)
        session_id=session_id
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Azodha, an intelligent virtual assistant for a healthcare appointment booking application. Your primary functions are to:

1. Guide patients towards booking appointments with appropriate healthcare professionals rather than attempting to answer medical questions directly.
2. Assist with the appointment booking process by providing available time slots and helping patients schedule with the right type of healthcare provider.
3. Offer general information about the healthcare facility, its services, and policies.
4. Provide reminders for upcoming appointments.
5. Help patients navigate the app and access basic, non-medical information.

Key points to remember:
- Always maintain a professional, empathetic, and patient-friendly tone.
- Prioritize patient privacy and confidentiality.
- Do not attempt to diagnose conditions, answer specific medical questions, or provide medical advice.
- For any medical concerns or questions, consistently recommend booking an appointment with an appropriate healthcare professional.
- For emergencies, always direct patients to call emergency services immediately.
- Each response should be at max 50 words, but try to to not hit it.

Your role is to facilitate connections between patients and healthcare providers, not to serve as a source of medical information."""),
   ("placeholder", "{chat_history}"),
    ("human", "{input}"),
     ("placeholder", "{agent_scratchpad}"),
])

chain = prompt | llm

embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))

tool = create_retriever_tool(
    retriever,
    "convert_response_to_vector",
    "this tool is used when you call another tools but the context token for them is a lot, so first you convert it to vector and then use it!.",
)

# Initialize tools
tools = [book_appointment, tool]

# Create tool calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

def get_agent_response(user_input, session_id):
    history = get_session_history(session_id)
    
    # Add user message to history
    history.add_user_message(user_input)
    
    # Prepare the full conversation history
    chat_history = history.messages
    
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
    
    # Add AI message to history
    history.add_ai_message(result['output'])
    
    return result['output']

def main():
    st.title("Azodha AI Assistant")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0

    # Fetch conversation history from Redis if available
    session_id = "112"  # Example session ID, you might want to generate or fetch this dynamically
    history = get_session_history(session_id)
    chat_history = history.messages

    # Update session state with fetched history
    if chat_history:
        st.session_state.conversation_history = [{"role": msg.type, "content": msg.content} for msg in chat_history]

    # Create a chat-like UI
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Create a text input for user messages
    user_input = st.text_input("Type your message here", key=f"user_input_{st.session_state.input_key}")

    # Create a button to send the message
    if st.button("Send") or user_input:
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get AI response
            with st.spinner("AI is thinking..."):
                ai_response = get_agent_response(user_input, session_id)
            
            # Display AI response
            with st.chat_message("ai"):
                st.write(ai_response)
            
            # Update conversation history
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({"role": "ai", "content": ai_response})
            
            # Clear the input box by changing the input key
            st.session_state.input_key += 1
            st.rerun()
        else:
            st.warning("Please enter a message before sending.")

    # Button to clear the conversation history
    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        history.clear()  # Clear the history in Redis as well
        st.session_state.input_key += 1
        st.rerun()

if __name__ == "__main__":
    main()