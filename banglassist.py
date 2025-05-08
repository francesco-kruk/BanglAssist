# Import necessary libraries
import uuid
import ast
import pandas as pd
import streamlit as st
import numpy as np
from FlagEmbedding import FlagReranker
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Define data directory
DATA_DIR = "data/faq.csv" # This directory refers to the company FAQ data

# Define OpenAI embedding model
EMBED_MODEL = "text-embedding-3-large"
# Define BAAI multilingual reranker model
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# Define OpenAI LLM
LLM_MODEL = "gpt-4o"

# Define number of documents retrieved in first retrieval step
TOP_K = 5
# Define number of documents kept after reranking retrieved documents
TOP_K_RERANK = 3

# Add OpenAI API key to run LLM and embedding model and set up OpenAI client
OPENAI_API_KEY = "<OpenAI API Key>"
client = OpenAI(api_key=OPENAI_API_KEY)


def reset_chat():
    """
    Resets the Streamlit session chat history as an empty list and the Streamlit session FAQ question as an empty string.
    """
    st.session_state.messages = []
    st.session_state.faq = ""


# Initialize the Streamlit session ID with a Universally Unique IDentifier (here version 4) and initialize the Streamlit session cache as an empty dictionary
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

# Initialize the Streamlit session chat history as an empty list
if "messages" not in st.session_state:
    reset_chat()

# Initialize the Streamlit session FAQ question as an empty string
if "faq" not in st.session_state:
    reset_chat()


@st.cache_resource(show_spinner=False) # Function saved in cache after first call to prevent the app from loading the reranker model at every run
def load_rerank_model():
    """
    Loads the reranker model.

    Returns:
        rerank_model: loaded reranker model.
    """
    # Load the reranker model through FlagReranker library
    rerank_model = FlagReranker(RERANK_MODEL, use_fp16=True) # Using half precision to speed up computation with a slight performance degradation
    
    return rerank_model


@st.cache_resource(show_spinner=False) # Function saved in cache after first call to prevent the app from loading the data at every run
def load_data():
    """
    Loads the FAQ data from a CSV as a Pandas dataframe.

    Returns:
        data (pandas.dataframe): Pandas dataframe of the FAQ data.
    """
    data = pd.read_csv(DATA_DIR)

    return data


def translate_question(question):
    """
    Translates the user question from whichevere language it was formulated in into English.

    Args:
        question (str): original user question.

    Returns:
        translated_question (str): user question translated into English.
    """
    # Define the prompt used by the LLM to translate the question into English
    prompt = f"""<s>[INST] Your task is to translate the user question to English.
        Your answer should only include the translated question.\n\n
        User Question: {question}[/INST] </s>"""
    
    # Define system and user messages
    messages = [
        {"role": "system", "content": "You are an expert translator."},
        {"role": "user", "content": prompt}
    ]

    # Generate the response from the LLM
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0 # Temperature set to 0 to improve accuracy and prompt adherence
    )

    # Select first LLM output as response
    translated_question = response.choices[0].message.content

    return translated_question

def retrieve_faq(question_embed, top_k=TOP_K):
    """
    Computes the cosine similarity between a query and multiple passages, and returns the top_k most relevant passages and their scores.

    Args:
        question_embed (numpy.ndarray): embedding of the (translated) query (1D array).
        top_k (int): number of top passages to return.

    Returns:
        top_k_passages (list): top_k retrieved passages.
        top_k_scores (list): similarity scores corresponding to top_k passages.
    """
    # Load passages from FAQ data
    passages = data["faq_combined"]
    # Load passage embeddings from FAQ data
    passages_embed = data["faq_combined_embed"]
    # Turn str into list
    passages_embed = passages_embed.apply(ast.literal_eval)
    # Turn list into numpy.ndarray
    passages_embed = np.stack(passages_embed)

    # Compute cosine similarities using sklearn
    similarities = cosine_similarity(question_embed, passages_embed).flatten()
    
    # Get the indices of the top_k scores
    top_k_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]  # Sort top_k in descending order
    
    # Retrieve the top_k passages and their scores
    top_k_passages = [passages[i] for i in top_k_indices]
    top_k_scores = [similarities[i] for i in top_k_indices]
    
    return top_k_passages, top_k_scores

def rerank_faq(query, query_embed, top_n=TOP_K_RERANK):
    """
    Takes as input a query, finds the top k passages, reranks them, and returns the top n passages.

    Args:
        query (str): (translated) user query.
        question_embed (numpy.ndarray): embedding of the (translated) query (1D array).
        top_n (int): number of top reranked passages to return.

    Returns
        passages (list): top_k retrieved passages.
        retrieval_scores (list): similarity scores corresponding to top_k passages.
        reranked_context (list): top_n reranked passages.
        reranked_scores (list): scores corresponding to top_n reranked passages.
    """

    # Retrieve the k most relevant passages and their scores
    passages, retrieval_scores = retrieve_faq(query_embed)

    # Compute the relevance scores through the reranker model
    scores = rerank_model.compute_score([[query, passage] for passage in passages])

    # Initialize the list of reranked passages
    reranked_passages = []
    # Add the passages and corresponding relevance scores to a dictionary withing the reranked passages list
    for reranked_passage, score in zip(passages, scores):
        reranked_passages.append({"passage": reranked_passage, "score": score})

    # Sort the list of reranked passages based on the relevance scores in descending order
    reranked_passages = sorted(reranked_passages, key=lambda x: x["score"], reverse=True)
    # Retrieve only the top n entries of the reranked passages list
    reranked_passages = reranked_passages[0:top_n]

    # Save the reranked passages and corresponding scores in separate lists
    reranked_context = [context["passage"] for context in reranked_passages]
    reranked_scores = [score["score"] for score in reranked_passages]
    
    return passages, retrieval_scores, reranked_context, reranked_scores

def generate_answer(question):
    """
    Takes as input a user question and generates the customer service answer.

    Args:
        question (str): user query

    Returns
        passages (list): top_k retrieved passages.
        retrieval_scores (list): similarity scores corresponding to top_k passages.
        reranked_context (list): top_n reranked passages.
        reranked_scores (list): scores corresponding to top_n reranked passages.
        answer (str): generated customer service answer to the user question
    """
    # Translate the user question into English to deal with complex linguistic scenarios
    eng_question = translate_question(question)

    # Embed the translated question and turn into numpy.ndarray
    eng_question_embed = client.embeddings.create(input = [eng_question], model=EMBED_MODEL).data[0].embedding
    eng_question_embed = np.array(eng_question_embed)
    eng_question_embed = eng_question_embed.reshape(1, -1)

    # Retreive reranked relevant passages
    passages, retrieval_scores, reranked_context, reranked_scores = rerank_faq(eng_question, eng_question_embed)
    
    # Define the prompt used by the LLM to generate the customer service answer
    prompt = f"""
        <s>[INST] You are a professional and empathetic customer service assistant for Hoichoi, a leading Bengali streaming service provider. 
        Your primary goal is to deliver accurate, clear, and helpful responses to customer questions while maintaining a conversational and respectful tone. 
        Use the provided context to answer the question.\n\n

        # Guidelines:\n
        1. **Match the script of the user's question in your response**:\n
            - Always respond in the same script and language as the user question, regardless of the language or script used in the provided context.\n
            - Analyze the script of the user question (e.g., Bengali, Roman, or others) and ensure your answer matches it exactly.\n
            - Examples:\n
                - User Question: "Eta dekhache. Ki korbo ekhon?" (Roman script)\n
                - Context: "ভৌগলিক বিধিনিষেধের কারণে কয়েকটি নির্দিষ্ট স্থান থেকে অল্প কিছু সামগ্রী দেখা সীমাবদ্ধ করা হয়েছে।" (Bengali script)\n
                - Answer: "Apni je shamogriṭi khunzhchen, ta apnar deshe upolobdho noy." (Roman script)\n

        2. Act as if the provided context is your own inherent knowledge. Do not reveal or imply that the information comes from external inputs.

        3. If the provided context does not allow you to answer confidently, be transparent about it. Instead of suggesting email communication, offer to directly connect the customer with a human operator for further assistance.

        4. **Conversation Flow**:\n
            - If this is the first interaction (no chat history), warmly greet the customer and address them by name if available.\n
            - If there is an ongoing conversation, skip the greeting and respond directly to the query.\n\n

        # Inputs:\n
        - User Question: {question}\n
        - Context: {reranked_context}
        [/INST] </s>
    """

    # Define system and user messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # Generate the response from the LLM
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0
    )

    # Select first LLM output as response
    answer = response.choices[0].message.content

    return passages, retrieval_scores, reranked_context, reranked_scores, answer

def display_faq(question):
    """
    Sets the Streamlit session FAQ question as a question selected by the user.

    Args:
        question (str): FAQ question selected by the user through the interface.
    """
    st.session_state.faq = question

def check_cache(query):
    """
    Takes as input a user query and checks whether it is already included in the FAQ list.
    If so, it retrieves the corresponding answer. If not, it generates a customer service answer.

    Args:
        query (str): user query

    Returns
        passages (list): top_k retrieved passages.
        retrieval_scores (list): similarity scores corresponding to top_k passages.
        reranked_context (list): top_n reranked passages.
        reranked_scores (list): scores corresponding to top_n reranked passages.
        answer (str): retrieved or generated customer service answer to the user question
    """
    # Embed the user query and turn into numpy.ndarray
    query_embed = client.embeddings.create(input = [query], model=EMBED_MODEL).data[0].embedding
    query_embed = np.array(query_embed)
    query_embed = query_embed.reshape(1, -1)

    # Load questions from FAQ data
    questions = data["faq_question"]
    # Load question embeddings from FAQ data
    questions_embed = data["faq_question_embed"]
    # Turn str into list
    questions_embed = questions_embed.apply(ast.literal_eval)
    # Turn list into numpy.ndarray
    questions_embed = np.stack(questions_embed)

    # Compute cosine similarities using sklearn
    questions_similarities = cosine_similarity(query_embed, questions_embed).flatten()

    # Get the indices of the top score
    top_score = np.argmax(questions_similarities)
    top_question = questions[top_score]

    # If similarity score greater than 0.8, retrieve answer corresponding to question in FAQ
    if questions_similarities[top_score] > 0.8:
        answer = data[data["faq_question"]==top_question].iloc[0]['faq_answer']
        passages = [top_question, "", "", "", ""]
        retrieval_scores = [questions_similarities[top_score], 0, 0, 0, 0]
        reranked_context = ["", "", ""]
        reranked_scores = [0, 0, 0]
    # Else, generate the response
    else:
        passages, retrieval_scores, reranked_context, reranked_scores, answer = generate_answer(query)

    return passages, retrieval_scores, reranked_context, reranked_scores, answer


# Display the app title
st.title("Welcome to Our Customer Service!")

# Display content in the sidebar
with st.sidebar:
    # Load reranker model
    rerank_model = load_rerank_model()
    # Load data
    data = load_data()
    # Show button to reset chat
    if st.button("New chat",use_container_width = True):
        # Reset the chat if button is clicked
        reset_chat()
        st.rerun()

# Define text to show in input bar
question_text = "How can I help you today?"
# If user asks a question, display the question and the retrieved/generated answer
if question := st.chat_input(question_text):
    # Display the current chat history
    chat_history = st.session_state.messages
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Add the user question to the chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display the user question
    with st.chat_message("user"):
        st.markdown(question)
    # Generate and display the assistant's response to the user question
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            passages, retrieval_scores, reranked_context, reranked_scores, answer = check_cache(question)
            print(f"Retrieved Passages:\n{passages}")
            print(f"Retrieval Scores:\n{retrieval_scores}")
            print(f"Reranked Passages:\n{reranked_context}")
            print(f"Reranking Scores:\n{reranked_scores}")
            # Add the assistant's answer to the chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(answer)
# If user has not asked a question, display three buttons showing a randomply sampled FAQ question
elif st.session_state.faq == "":
    st.markdown("Here are some questions you might be interested in:")
    col1, col2, col3 = st.columns(3)
    random_faq = data.sample(n=3)
    with col1:
        faq_text_1 = random_faq['faq_question'].values[0]
        st.button(faq_text_1, on_click=display_faq, args=(faq_text_1,), use_container_width=True)
    with col2:
        faq_text_2 = random_faq['faq_question'].values[1]
        st.button(faq_text_2, on_click=display_faq, args=(faq_text_2,), use_container_width=True)
    with col3:
        faq_text_3 = random_faq['faq_question'].values[2]
        st.button(faq_text_3, on_click=display_faq, args=(faq_text_3,), use_container_width=True)
# If user selects an FAQ question by clicking the corresponding button, display the question and the retrieved answer
else:
    question = st.session_state.faq
    # Add the user question to the chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display the user question
    with st.chat_message("user"):
        st.markdown(question)
    # Display the assistant's response to the user question
    with st.chat_message("assistant"):
        answer = data[data["faq_question"]==question].iloc[0]['faq_answer']
        # Add the assistant's answer to the chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(answer)
    