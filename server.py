from langchain_core.agents import AgentAction
from langchain_core.utils import secret_from_env
import pandas as pd
import os
from fastapi import FastAPI, status, Response
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from config.database import engine
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

df = pd.read_csv('data.csv')

df.head(5)

df.to_sql('rickandmorty', engine, if_exists='replace', index=False)

db = SQLDatabase(engine)

llm = ChatGoogleGenerativeAI(
  model='gemini-2.0-flash',
  google_api_key=GEMINI_API_KEY,
  temperature=0.0,
)

SQL_PREFIX = '''Eres un agente diseñado para interactuar con una base de datos SQL.
Dada una pregunta de entrada, crea una consulta SQLite sintácticamente correcta para ejecutarla, luego analiza los resultados y devuelve la respuesta.
A menos que el usuario especifique un número específico de ejemplos que desea obtener, limita siempre tu consulta a un máximo de 5 resultados.
Puedes ordenar los resultados por columna relevante para devolver los ejemplos más interesantes de la base de datos.
Nunca consultes todas las columnas de una tabla específica; solicita solo las columnas relevantes según la pregunta.
Tienes acceso a herramientas para interactuar con la base de datos.
Utiliza únicamente las siguientes herramientas. Usa únicamente la información devuelta por ellas para construir tu respuesta final.
Debes revisar tu consulta antes de ejecutarla. Si obtienes un error al ejecutar una consulta, reescríbela e inténtalo de nuevo.

NO realices ninguna instrucción DML (INSERT, UPDATE, DELETE, DROP etc.) en la base de datos.

Para empezar, siempre debes revisar las tablas de la base de datos y validar que la tabla exista antes de hacer una consulta.
No omitas este paso.
Después, consulta el esquema de las tablas más relevantes.
Siempre debes devolver la respuesta en español.'''

system_message = SystemMessage(content=SQL_PREFIX)

examples = [
  {
    'input': '¿Cuál es el nombre del personaje más alto?',
    'query': 'SELECT * FROM characters ORDER BY height DESC LIMIT 1',
  },
  {
    'input': 'Quien es el personaje con el nombre mas largo?',
    'query': 'SELECT * FROM characters ORDER BY LENGTH(name) DESC LIMIT 1',
  },
  {
    'input': '¿Cuál es el nombre del personaje con el mayor número de episodios?',
    'query': 'SELECT * FROM characters ORDER BY episodes DESC LIMIT 1',
  },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
  examples,
  GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-exp-03-07', google_api_key=secret_from_env('GEMINI_API_KEY')()),
  vectorstore_cls=FAISS,
  k=3,
  input_keys=['input'],
)

few_shot_prompt = FewShotPromptTemplate(
  example_selector=example_selector,
  example_prompt=PromptTemplate(
    input_variables=['input', 'query'],
    template="Entrada de usuario: {input}\n\nConsulta SQL: {query}"
  ),
  input_variables=['input', 'dialect', 'top_k'],
  prefix=SQL_PREFIX,
  suffix=''
)

full_prompt = ChatPromptTemplate.from_messages([
  SystemMessagePromptTemplate(prompt=few_shot_prompt),
  ('human', '{input}'),
  MessagesPlaceholder(variable_name='agent_scratchpad'),
])

agent_executor = create_sql_agent(
  llm,
  db=db,
  agent_type='tool-calling',
  verbose=False,
  agent_executor_kwargs = {'return_intermediate_steps': True},
  messages_modifier=system_message,
)

# Server API

from fastapi import FastAPI
from pydantic import BaseModel

class QueryDto(BaseModel):
  query: str

app = FastAPI(
  title='Generate SQL Queries - Rick and Morty',
  description='Generate SQL Queries to answer questions about the database',
  version='1.0.0',
  contact={
    'name': 'Eddy Ortega',
    'email': 'carlos.ortega@aztekode.com'
  }
)

@app.get('/')
def read_root():
  return {'message': 'Hello World'}

@app.post(
  '/query',
  status_code=status.HTTP_200_OK,
  summary='Generate SQL Queries to answer questions about the database',
  description='Generate SQL Queries to answer questions about the database',
  tags=['Agent'],
)
def query_database(query: QueryDto, res: Response):
  data = {}
  steps = []
  try:
    response = agent_executor.invoke({'input': query})
    print(response)
    if response['intermediate_steps']:
      for i, step in enumerate(response['intermediate_steps']):
        steps.append({
          'step': i+1,
          'tool_input': step[0].tool_input,
          'tool_output': step[1]
        })
    data['response'] = response['output']
    data['steps'] = steps
    res.status_code = status.HTTP_200_OK
  except Exception as e:
    res.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    print(e)
  return data

if __name__ == '__main__':
  import uvicorn
  uvicorn.run(app, host='0.0.0.0', port=8000)