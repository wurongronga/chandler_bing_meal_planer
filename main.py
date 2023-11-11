from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.chains import LLMChain,SequentialChain

load_dotenv()  # take environment variables from .env.
API_KEY = os.environ['OPENAI_API_KEY']

llm = OpenAI(openai_api_key=API_KEY,temperature=0.9)

prompt_template = PromptTemplate(
    template="""Give me an example of a meal could be made using the following ingredients: {ingredients}. 
                The output should give a little introduction for the meal, then list all the ingredients, and the cooking instructions.""",
    input_variables=['ingredients']
)

chandler_bing_template = PromptTemplate(
    template="""Re-write the meals given below in the style of Chandler Bing:
    
    Meals:
    {meals}
    """,
    input_variables=['meals']
)

meal_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="meals",
    verbose=True,
)

chandler_bing_chain = LLMChain(
    llm=llm,
    prompt=chandler_bing_template,
    output_key="chandler_meals",
    verbose=True,
)

overall_chain = SequentialChain(
    chains=[meal_chain,chandler_bing_chain],
    input_variables=["ingredients"],
    output_variables=['meals','chandler_meals'],
    verbose=True
)

st.title("Chandler Bing's Meal Planer")
user_prompt = st.text_input("Enter a comma-separated list of ingredients")

if st.button("Generate") and user_prompt:
    with st.spinner("Generating..."):
        output = overall_chain({"ingredients":user_prompt})
        col1,col2 = st.columns(2)
        col1.subheader("Chandler Bing Says...")
        col1.write(output['chandler_meals'])
        col2.subheader("Recipe:")
        col2.write(output['meals'])

