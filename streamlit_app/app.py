#!/usr/bin/env python
# coding: utf-8

"""
# 📘 Bedrock Multi-Model RAG App (Streamlit Version)
Esta aplicación permite:
- Cargar múltiples archivos de texto
- Crear un vectorstore FAISS
- Hacer preguntas
- Obtener respuestas de múltiples modelos de Amazon Bedrock simultáneamente
- Medir tiempo de respuesta para cada modelo
"""

import os
import time
import asyncio
import streamlit as st
import boto3
from botocore.exceptions import (
    ClientError, 
    NoCredentialsError, 
    CredentialRetrievalError, 
    EndpointConnectionError
)
from loguru import logger
import pandas as pd
from pathlib import Path

# Configuración de loguru
logger.remove()  # Remover el handler por defecto
logger.add("bedrock_rag_app.log", rotation="10 MB", level="INFO")
logger.add(lambda msg: st.sidebar.error(msg) if "ERROR" in msg else None, level="ERROR")

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Bedrock Multi-Model RAG App",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título principal
st.title("📘 Bedrock Multi-Model RAG App")
st.markdown("""
Esta aplicación permite comparar respuestas de múltiples modelos de Amazon Bedrock 
cuando se les hace la misma pregunta sobre documentos que hayas subido.
""")

# Lista de modelos disponibles
DEFAULT_MODELS = [
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    # "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    # "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
]

# Sidebar para configuración
with st.sidebar:
    st.header("Configuración")
    
    # Región para AWS
    aws_region = st.selectbox(
        "Región de AWS",
        options=["us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1"],
        index=0
    )
    
    # Selección de modelos
    st.subheader("Modelos a utilizar")
    selected_models = []
    for model in DEFAULT_MODELS:
        if st.checkbox(model, value=True, key=f"model_{model}"):
            selected_models.append(model)
    
    # Parámetros avanzados
    st.subheader("Parámetros avanzados")
    max_tokens = st.slider("Máximo de tokens", min_value=100, max_value=1000, value=300, step=50)
    temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    word_limit = st.slider("Límite de palabras", min_value=20, max_value=200, value=50, step=10)
    chunk_size = st.slider("Tamaño de chunk", min_value=100, max_value=1000, value=500, step=100)
    chunk_overlap = st.slider("Solapamiento de chunks", min_value=0, max_value=200, value=100, step=20)
    k_docs = st.slider("Número de documentos relevantes (k)", min_value=1, max_value=10, value=7, step=1)

# Estado de la aplicación
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'has_uploaded_files' not in st.session_state:
    st.session_state.has_uploaded_files = False
if 'client' not in st.session_state:
    st.session_state.client = None


# Función para verificar acceso a Bedrock
def verify_bedrock_access(model_ids=None):
    """Verifica el acceso a Amazon Bedrock y los modelos configurados."""
    if model_ids is None:
        model_ids = selected_models
        
    try:
        client = boto3.client('bedrock-runtime', region_name=aws_region)
        st.session_state.client = client
        
        # Verificar credenciales y conexión básica
        try:
            client.list_foundation_models()
        except AttributeError:
            # Esta operación puede no estar disponible en todas las versiones
            # Vamos a intentar otra operación básica
            pass
        
        # Verificar acceso a cada modelo configurado
        valid_models = []
        invalid_models = []
        
        # Mostrar progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_id in enumerate(model_ids):
            status_text.text(f"Verificando acceso al modelo: {model_id}")
            progress_value = (i + 1) / len(model_ids)
            progress_bar.progress(progress_value)
            
            try:
                # Simplemente verificamos que podemos invocar una operación
                # En lugar de get_foundation_model, que puede no estar disponible,
                # usamos otra operación más garantizada
                inference_config = {'maxTokens': 10, 'temperature': 0.5, 'topP': 0.9}
                messages = [{'role': 'user', 'content': [{'text': 'test'}]}]
                
                # Hacemos una solicitud muy pequeña para verificar permisos
                # Con un timeout bajo para no esperar mucho
                client.converse(
                    modelId=model_id,
                    messages=messages,
                    inferenceConfig=inference_config
                )
                valid_models.append(model_id)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_msg = e.response.get('Error', {}).get('Message', str(e))
                
                if error_code == 'AccessDeniedException':
                    invalid_models.append((model_id, f"Sin acceso: {error_msg}"))
                elif error_code == 'ValidationException':
                    invalid_models.append((model_id, f"Modelo inválido: {error_msg}"))
                else:
                    invalid_models.append((model_id, f"Error: {error_msg}"))
            except Exception as e:
                invalid_models.append((model_id, f"Error inesperado: {str(e)}"))
                
        # Limpiar elementos temporales
        progress_bar.empty()
        status_text.empty()
        
        return {
            "success": len(valid_models) > 0,
            "valid_models": valid_models,
            "invalid_models": invalid_models
        }
    
    except NoCredentialsError:
        logger.error("No se encontraron credenciales de AWS")
        return {
            "success": False,
            "error": "No se encontraron credenciales de AWS. Verifique su configuración de AWS CLI o variables de entorno."
        }
    except CredentialRetrievalError as e:
        logger.error(f"Error al recuperar credenciales: {str(e)}")
        return {
            "success": False,
            "error": f"Error al recuperar credenciales: {str(e)}"
        }
    except EndpointConnectionError as e:
        logger.error(f"Error de conexión al endpoint de Bedrock: {str(e)}")
        return {
            "success": False,
            "error": f"Error de conexión al endpoint de Bedrock: {str(e)}"
        }
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        
        logger.error(f"Error de cliente AWS ({error_code}): {error_msg}")
        if error_code == 'AccessDeniedException':
            return {
                "success": False,
                "error": f"Acceso denegado a Amazon Bedrock: {error_msg}"
            }
        else:
            return {
                "success": False,
                "error": f"Error de cliente AWS ({error_code}): {error_msg}"
            }
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        return {
            "success": False,
            "error": f"Error inesperado: {str(e)}"
        }

# Función para inicializar el cliente de Bedrock
def init_bedrock_client():
    try:
        client = boto3.client('bedrock-runtime', region_name=aws_region)
        st.session_state.client = client
        return True
    except Exception as e:
        logger.error(f"Error al crear cliente Bedrock: {str(e)}")
        st.error(f"Error al inicializar el cliente de Amazon Bedrock: {str(e)}")
        return False

# Función para cargar y dividir documentos de texto
def load_and_split_txts(txt_paths, chunk_size=500, chunk_overlap=100):
    """Carga y divide documentos de texto en chunks."""
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    with st.spinner("Procesando documentos..."):
        progress_bar = st.progress(0)
        for i, path in enumerate(txt_paths):
            try:
                loader = TextLoader(str(path), encoding='utf-8')
                docs = loader.load()
                chunks = splitter.split_documents(docs)
                all_docs.extend(chunks)
                # Actualizar progreso
                progress = (i + 1) / len(txt_paths)
                progress_bar.progress(progress)
            except Exception as e:
                logger.error(f"Error al procesar {path}: {str(e)}")
                st.error(f"Error al procesar {path}: {str(e)}")
    
    return all_docs

# Función para inicializar embeddings y vectorstore
def setup_embeddings_and_vectorstore(documents):
    """Inicializa el modelo de embeddings y crea el vectorstore."""
    from langchain.vectorstores import FAISS
    from langchain.embeddings import BedrockEmbeddings 
    
    try:
        with st.spinner("Configurando modelo de embeddings..."):
            # Intentamos crear el cliente de embeddings
            embeddings = BedrockEmbeddings(client=st.session_state.client)
            
            # Verificamos que funcione correctamente haciendo una prueba simple
            test_result = embeddings.embed_query("Texto de prueba para verificar embeddings")
            if not test_result or not isinstance(test_result, list) or len(test_result) < 10:
                raise ValueError("La respuesta de embeddings no tiene el formato esperado")
                
            st.success("✅ Modelo de embeddings configurado correctamente")
            
        with st.spinner("Creando índice vectorial..."):
            vectorstore = FAISS.from_documents(documents, embeddings)
            st.success(f"✅ Índice FAISS creado con {len(documents)} fragmentos.")
            return vectorstore, True
            
    except Exception as e:
        error_msg = str(e)
        
        if "AccessDeniedException" in error_msg:
            display_message = "No tiene permisos para acceder al servicio de embeddings. Verifique sus políticas IAM."
        elif "ResourceNotFoundException" in error_msg:
            display_message = "El modelo de embeddings solicitado no está disponible en su región."
        elif "ThrottlingException" in error_msg:
            display_message = "Se ha superado el límite de solicitudes para el servicio de embeddings."
        elif "ValidationException" in error_msg:
            display_message = "Error de validación en la solicitud de embeddings."
        elif "ServiceUnavailableException" in error_msg:
            display_message = "El servicio de embeddings no está disponible en este momento."
        elif "UnrecognizedClientException" in error_msg or "No credential provider" in error_msg:
            display_message = "Credenciales de AWS no válidas o no encontradas."
        else:
            display_message = f"Error al configurar embeddings: {error_msg}"
            
        logger.error(f"Error de embeddings: {display_message}")
        st.error(f"❌ Error en la configuración de embeddings: {display_message}")
        
        return None, False

# Función para invocar el modelo de Bedrock
async def invoke_converse(model_id, messages, inference_config):
    """Invoca al modelo de Bedrock con manejo detallado de errores.
    
    Args:
        model_id: El ID del modelo de Bedrock
        messages: Lista de mensajes para la conversación
        inference_config: Configuración de inferencia
        
    Returns:
        Un diccionario con la respuesta del modelo, con estructura
        esperada {output: {message: {content: [{text: "..."}]}}}
    
    Raises:
        RuntimeError: Si hay problemas con el cliente, permisos o modelo.
    """
    if st.session_state.client is None:
        logger.error("Client is None, Bedrock client no inicializado")
        raise RuntimeError("Cliente de Bedrock no inicializado correctamente.")
        
    try:
        # Llamar al API de Bedrock
        logger.info(f"Invocando modelo {model_id}")
        response = st.session_state.client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig=inference_config
        )
        
        # Verificar que la respuesta tenga la estructura esperada
        if not isinstance(response, dict):
            logger.error(f"Respuesta inesperada de tipo {type(response)} para modelo {model_id}")
            raise RuntimeError(f"Respuesta inesperada del modelo {model_id}")
            
        # Verificar la estructura de la respuesta
        if 'output' not in response:
            logger.error(f"Respuesta sin campo 'output' del modelo {model_id}: {response}")
            raise RuntimeError(f"Respuesta del modelo {model_id} no tiene el formato esperado")
            
        if 'message' not in response['output']:
            logger.error(f"Respuesta sin campo 'message' del modelo {model_id}: {response}")
            raise RuntimeError(f"Respuesta del modelo {model_id} no tiene el formato esperado")
            
        # Todo en orden, devolver la respuesta
        return response
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        
        if error_code == 'AccessDeniedException':
            logger.error(f"Acceso denegado al modelo {model_id}: {error_msg}")
            raise RuntimeError(f"Acceso denegado al modelo {model_id}. Verifique los permisos IAM.")
        elif error_code == 'ValidationException' and 'model not found' in error_msg.lower():
            logger.error(f"Modelo {model_id} no encontrado: {error_msg}")
            raise RuntimeError(f"Modelo {model_id} no encontrado o no disponible en su región.")
        elif error_code == 'ThrottlingException':
            logger.error(f"Límite de tasa excedido para modelo {model_id}: {error_msg}")
            raise RuntimeError(f"Límite de tasa excedido para el modelo {model_id}. Intente nuevamente más tarde.")
        elif error_code == 'ServiceQuotaExceededException':
            logger.error(f"Cuota de servicio excedida para modelo {model_id}: {error_msg}")
            raise RuntimeError(f"Cuota de servicio excedida para el modelo {model_id}.")
        else:
            # Reenvía el error original con información adicional
            logger.error(f"Error al invocar modelo {model_id}: {error_code} - {error_msg}")
            raise RuntimeError(f"Error al invocar modelo {model_id}: {error_msg}")
    except Exception as e:
        # Capturar cualquier otro error inesperado
        logger.error(f"Error inesperado al invocar modelo {model_id}: {str(e)}")
        raise RuntimeError(f"Error inesperado al invocar modelo {model_id}: {str(e)}")

# Función para ejecutar una consulta en un modelo específico
async def run_chain(model_id, query, context, word_limit):
    """Ejecuta una consulta en un modelo específico con manejo de errores mejorado."""
    try:
        # Crear el prompt
        prompt = f"""
        <instrucciones_agente>
          <proposito>
            Eres un asistente de información bancaria.
            Tu única función es responder exclusivamente con base en los documentos internos proporcionados.
          </proposito>

          <reglas>
            <regla>No inventes ni supongas información, aunque parezca razonable.</regla>
            <regla>No uses frases introductorias como "para abrir una cuenta necesitas...". Sé directo.</regla>
            <regla>Si la información no está en los documentos, responde "No tengo esa información."</regla>
            <regla>Tu respuesta no debe superar {word_limit} palabras bajo ninguna circunstancia.</regla>
          </reglas>
        </instrucciones_agente>

        Contesta solo con base en el siguiente resumen de políticas oficiales internas:

        <documentos_contexto>
        {context}
        </documentos_contexto>

        <pregunta_usuario>
        {query}
        </pregunta_usuario>

        <instruccion_final>
        Tu respuesta debe tener como máximo {word_limit} palabras. 
        Si no hay información suficiente en el contexto, responde claramente "No tengo esa información".
        </instruccion_final>
        """
        
        messages = [
            {
                'role': 'user',
                'content': [{'text': prompt}]
            }
        ]

        inference_config = {
            'maxTokens': max_tokens,
            'temperature': temperature,
            'topP': top_p
        }

        start = time.time()

        # Asegurarnos de esperar la coroutine y manejar posibles errores en la estructura de respuesta
        response = await invoke_converse(model_id, messages, inference_config)

        # Añadir comprobaciones para evitar errores si la estructura es inesperada
        if not response or 'output' not in response:
            logger.error(f"Respuesta inválida o nula del modelo {model_id}")
            raise ValueError(f"Respuesta inválida del modelo {model_id}")

        output_message = response['output'].get('message', {})
        if not output_message or 'content' not in output_message or not output_message['content']:
            logger.error(f"Formato de mensaje inesperado del modelo {model_id}: {response}")
            raise ValueError(f"Formato de respuesta inválido del modelo {model_id}")
            
        result = output_message['content'][0]['text']

        # Obtener el número de tokens de salida
        output_tokens = response.get('usage', {}).get('outputTokens', 'N/A')

        # Calcular el número aproximado de palabras
        if isinstance(output_tokens, int):
            words_estimate = output_tokens / 2  # Relación tokens-palabras para español
        else:
            words_estimate = 'N/A'

        elapsed = time.time() - start
        return {
            "model": model_id,
            "response": result.strip(),
            "time": round(elapsed, 2),
            "tokens": output_tokens,
            "words_estimate": words_estimate,
            "status": "success"
        }

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        error_msg = e.response.get('Error', {}).get('Message', '')
        
        error_messages = {
            'AccessDeniedException': f"❌ Error de permisos: No tiene acceso al modelo {model_id}.",
            'ThrottlingException': f"❌ Error de limitación: Se ha excedido el límite de peticiones para {model_id}.",
            'ValidationException': f"❌ Error de validación: {error_msg}",
            'ServiceUnavailableException': f"❌ Servicio no disponible: Amazon Bedrock está temporalmente no disponible.",
            'InternalServerException': f"❌ Error interno del servidor: Problema en Amazon Bedrock."
        }
        
        custom_message = error_messages.get(error_code, f"❌ Error de AWS ({error_code}): {error_msg}")
        logger.error(f"Error al invocar {model_id}: {error_code} - {error_msg}")
        
        return {
            "model": model_id,
            "response": custom_message,
            "time": None,
            "tokens": None,
            "words_estimate": None,
            "status": "error",
            "error_code": error_code
        }
    except NoCredentialsError:
        logger.error(f"No se encontraron credenciales para acceder a {model_id}")
        return {
            "model": model_id,
            "response": "❌ Error: No se encontraron credenciales de AWS. Verifique la configuración.",
            "time": None,
            "tokens": None,
            "words_estimate": None,
            "status": "error",
            "error_code": "NoCredentials"
        }
    except RuntimeError as e:
        logger.error(f"Error de tiempo de ejecución con {model_id}: {str(e)}")
        return {
            "model": model_id,
            "response": f"❌ Error: {str(e)}",
            "time": None,
            "tokens": None,
            "words_estimate": None,
            "status": "error",
            "error_code": "Runtime"
        }
    except Exception as e:
        logger.error(f"Error inesperado con {model_id}: {str(e)}")
        return {
            "model": model_id,
            "response": f"❌ Error inesperado: {str(e)}",
            "time": None,
            "tokens": None,
            "words_estimate": None,
            "status": "error",
            "error_code": "Unknown"
        }

# Función para ejecutar la consulta en todos los modelos
async def run_all_models(query, docs, word_limit=50):
    """Ejecuta la consulta en todos los modelos configurados."""
    context = "\n\n".join([d.page_content for d in docs])
    
    # Primero verificamos el acceso a Bedrock
    if st.session_state.client is None:
        logger.error("No se puede ejecutar consulta: cliente Bedrock no inicializado")
        return [{
            "model": "SYSTEM",
            "response": "❌ Error: Cliente de Amazon Bedrock no inicializado. Verifique sus credenciales y permisos.",
            "time": None,
            "tokens": None,
            "words_estimate": None,
            "status": "error",
            "error_code": "NoClient"
        }]
    
    # Filtramos cualquier modelo que esté vacío o mal formateado
    valid_models = [mid for mid in selected_models if mid and isinstance(mid, str)]
    if not valid_models:
        logger.error("No hay modelos válidos configurados")
        return [{
            "model": "SYSTEM",
            "response": "❌ Error: No hay modelos válidos configurados.",
            "time": None,
            "tokens": None,
            "words_estimate": None,
            "status": "error",
            "error_code": "NoModels"
        }]
    
    # Ejecutamos las consultas en todos los modelos válidos
    try:
        logger.info(f"Ejecutando consulta en {len(valid_models)} modelos")
        tasks = []
        for mid in valid_models:
            task = run_chain(mid, query, context, word_limit)
            tasks.append(task)
        
        # Ejecutar todas las tareas y esperar a que terminen
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar los resultados para manejar excepciones
        processed_results = []
        for i, result in enumerate(results):
            mid = valid_models[i]
            if isinstance(result, Exception):
                # Si el resultado es una excepción, crear una respuesta de error
                logger.error(f"Error al procesar modelo {mid}: {str(result)}")
                processed_results.append({
                    "model": mid,
                    "response": f"❌ Error: {str(result)}",
                    "time": None,
                    "tokens": None,
                    "words_estimate": None,
                    "status": "error",
                    "error_code": "ExecutionError"
                })
            else:
                # Si no es una excepción, añadir el resultado normal
                processed_results.append(result)
        
        return processed_results
    except Exception as e:
        logger.error(f"Error al ejecutar consultas en paralelo: {str(e)}")
        return [{
            "model": "SYSTEM",
            "response": f"❌ Error general al ejecutar modelos: {str(e)}",
            "time": None,
            "tokens": None,
            "words_estimate": None,
            "status": "error",
            "error_code": "ParallelExecutionError"
        }]

# Función para mostrar los resultados de la comparación
def display_model_results(results):
    """Muestra los resultados de los modelos en formato de tabla y detalle."""
    # Crear un DataFrame para comparación
    comparison_data = []
    for r in results:
        status = r.get("status", "unknown")
        model_name = r["model"]
        
        if status == "success":
            comparison_data.append({
                "Modelo": model_name,
                "Tiempo (s)": r["time"],
                "Tokens": r.get("tokens", "N/A"),
                "Palabras Est.": r.get("words_estimate", "N/A"),
                "Estado": "✅ OK"
            })
        else:
            error_code = r.get("error_code", "Unknown")
            comparison_data.append({
                "Modelo": model_name,
                "Tiempo (s)": "N/A",
                "Tokens": "N/A",
                "Palabras Est.": "N/A",
                "Estado": f"❌ Error: {error_code}"
            })
    
    # Mostrar tabla de comparación
    st.subheader("📊 Comparación de modelos")
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)
    
    # Mostrar las respuestas detalladas de cada modelo
    st.subheader("🤖 Respuestas por modelo")
    for r in results:
        with st.expander(f"🧠 {r['model']}", expanded=True):
            if r.get("status", "") == "success":
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric("Tiempo de respuesta", f"{r['time']} segundos")
                with col2:
                    st.metric("Tokens generados", r.get('tokens', 'N/A'))
                
                st.markdown("### Respuesta:")
                st.markdown(r["response"])
            else:
                st.error(r["response"])

# Cargar archivos
st.subheader("1️⃣ Cargar archivos de texto")
uploaded_files = st.file_uploader(
    "Arrastra o haz clic para seleccionar archivos de texto",
    type=["txt"],
    accept_multiple_files=True
)

# Crear directorio para archivos si no existe
upload_dir = Path("uploaded_txts")
upload_dir.mkdir(exist_ok=True)

# Procesar archivos subidos
if uploaded_files and not st.session_state.has_uploaded_files:
    # Guardar archivos subidos
    saved_files = []
    for file in uploaded_files:
        file_path = upload_dir / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_files.append(file_path)
    
    st.success(f"✅ {len(saved_files)} archivos guardados exitosamente")
    
    # Cargar y dividir los documentos
    try:
        documents = load_and_split_txts(saved_files, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.session_state.documents = documents
        st.session_state.has_uploaded_files = True
        st.success(f"✅ Se procesaron {len(documents)} fragmentos de texto")
    except Exception as e:
        logger.error(f"Error al procesar documentos: {str(e)}")
        st.error(f"Error al procesar documentos: {str(e)}")

# Inicialización de Bedrock y vectorstore
if st.session_state.has_uploaded_files and st.session_state.documents and st.session_state.vectorstore is None:
    # Inicializar cliente Bedrock
    if init_bedrock_client():
        # Verificar acceso a Bedrock
        access_status = verify_bedrock_access()
        
        if access_status["success"]:
            # Configurar embeddings y vectorstore
            vectorstore, success = setup_embeddings_and_vectorstore(st.session_state.documents)
            if success:
                st.session_state.vectorstore = vectorstore
        else:
            st.error(f"❌ Error de acceso a Bedrock: {access_status.get('error', 'Error desconocido')}")

# Sección de consultas
st.subheader("2️⃣ Hacer preguntas a los documentos")

# Verificar si tenemos todo listo para las consultas
is_ready = (st.session_state.has_uploaded_files and 
            st.session_state.documents and 
            st.session_state.vectorstore is not None and 
            st.session_state.client is not None)

if not is_ready:
    st.warning("⚠️ Primero debe cargar archivos y esperar a que se configuren los modelos de embeddings y el índice vectorial.")
else:
    # Campo para la consulta
    query = st.text_input("Escribe tu pregunta aquí:", placeholder="cuanto cuesta personalizar el plástico de guardadito kids?")
    
    # Botón para ejecutar la consulta
    if st.button("📝 Realizar consulta", type="primary"):
        if not query:
            st.warning("⚠️ Por favor ingresa una pregunta")
        elif not selected_models:
            st.warning("⚠️ Selecciona al menos un modelo en la barra lateral")
        else:
            with st.spinner("Buscando documentos relevantes..."):
                try:
                    # Buscar documentos relevantes
                    relevant_docs = st.session_state.vectorstore.similarity_search(query, k=k_docs)
                    if not relevant_docs:
                        st.warning("⚠️ No se encontraron documentos relevantes para esta consulta")
                    
                    # Mostrar fragmentos relevantes en un expander
                    with st.expander("📄 Ver fragmentos relevantes", expanded=False):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**Fragmento {i+1}:**")
                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            st.markdown("---")
                    
                    # Ejecutar la consulta en todos los modelos
                    with st.spinner("Consultando modelos de Bedrock..."):
                        results = asyncio.run(run_all_models(query, relevant_docs, word_limit))
                        # Mostrar resultados
                        display_model_results(results)
                        
                except Exception as e:
                    logger.error(f"Error al ejecutar la consulta: {str(e)}")
                    st.error(f"❌ Error al ejecutar la consulta: {str(e)}")

# Footer
st.markdown("---")
st.markdown("📘 **Bedrock Multi-Model RAG App** - Powered by Amazon Bedrock, LangChain, and Streamlit")
