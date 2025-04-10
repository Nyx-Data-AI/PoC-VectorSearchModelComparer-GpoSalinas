"""
# 📘 Bedrock Multi-Model RAG App (Streamlit Version)
Esta aplicación permite:
- Cargar múltiples archivos de texto
- Crear un vectorstore FAISS
- Hacer preguntas
- Obtener respuestas de múltiples modelos de Amazon Bedrock simultáneamente
- Medir tiempo de respuesta para cada modelo
- Usar un agente de Amazon Bedrock directamente (sin necesidad de cargar documentos)
"""

import time
import asyncio
import streamlit as st
import boto3
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    CredentialRetrievalError,
    EndpointConnectionError,
)
from loguru import logger
import pandas as pd
from pathlib import Path

logger.remove()
logger.add("bedrock_rag_app.log", rotation="10 MB", level="INFO")
logger.add(lambda msg: st.sidebar.error(msg) if "ERROR" in msg else None, level="ERROR")

st.set_page_config(
    page_title="Bedrock Multi-Model RAG App",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📘 Bedrock Multi-Model RAG App")
st.markdown(
    """
Esta aplicación permite:
1. **Modo RAG tradicional**: Comparar respuestas de múltiples modelos de Amazon Bedrock 
   cuando se les hace la misma pregunta sobre documentos que hayas subido.
2. **Modo Agente**: Enviar consultas directamente a un agente de Amazon Bedrock 
   sin necesidad de cargar documentos locales.
"""
)

DEFAULT_MODELS = [
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
]

with st.sidebar:
    st.header("Configuración")

    aws_region = st.selectbox(
        "Región de AWS",
        options=["us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1"],
        index=0,
    )

    st.subheader("Modelos a utilizar")
    selected_models = []
    for model in DEFAULT_MODELS:
        if st.checkbox(model, value=True, key=f"model_{model}"):
            selected_models.append(model)

    st.subheader("Parámetros avanzados")
    max_tokens = st.slider(
        "Máximo de tokens", min_value=100, max_value=1000, value=300, step=50
    )
    temperature = st.slider(
        "Temperatura", min_value=0.0, max_value=1.0, value=0.2, step=0.1
    )
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    word_limit = st.slider(
        "Límite de palabras", min_value=20, max_value=200, value=50, step=10
    )
    chunk_size = st.slider(
        "Tamaño de chunk", min_value=100, max_value=1000, value=500, step=100
    )
    chunk_overlap = st.slider(
        "Solapamiento de chunks", min_value=0, max_value=200, value=100, step=20
    )
    k_docs = st.slider(
        "Número de documentos relevantes (k)",
        min_value=1,
        max_value=10,
        value=7,
        step=1,
    )

    st.subheader("Modo de consulta")
    use_bedrock_agent = st.checkbox(
        "Usar agente de Amazon Bedrock",
        value=False,
        help="Si se marca, las consultas se enviarán a un agente de Amazon Bedrock en lugar de usar FAISS",
    )
    bedrock_agent_id = st.text_input(
        "ID del agente de Bedrock",
        placeholder="Ingrese el ID del agente",
        disabled=not use_bedrock_agent,
    )
    bedrock_agent_alias_id = st.text_input(
        "ID del alias del agente",
        placeholder="Ingrese el ID del alias del agente",
        disabled=not use_bedrock_agent,
    )

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "has_uploaded_files" not in st.session_state:
    st.session_state.has_uploaded_files = False
if "client" not in st.session_state:
    st.session_state.client = None


async def invoke_bedrock_agent(agent_id, agent_alias_id, query):
    try:
        start = time.time()

        if st.session_state.client is None:
            logger.error("Cliente Bedrock no inicializado para invocar agente")
            raise RuntimeError("Cliente de Bedrock no inicializado correctamente.")

        agents_client = boto3.client("bedrock-agent-runtime", region_name=aws_region)

        logger.info(f"Invocando agente {agent_id} con alias {agent_alias_id}")

        session_id = f"session-{int(time.time())}"

        response = agents_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=query,
        )

        full_response = ""
        output_tokens = 0  

        for event in response["completion"]:
            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    full_response += chunk["bytes"].decode("utf-8")

                if "attribution" in chunk:
                    citations = chunk["attribution"].get("citations", [])
                    for citation in citations:
                        references = citation.get("retrievedReferences", [])
                        for ref in references:
                            metadata = ref.get("metadata", {})
                            tokens = metadata.get("outputTokens")
                            if isinstance(tokens, int):
                                output_tokens += tokens

        elapsed = time.time() - start
        full_response = full_response.strip()

        negative_responses = [
            "sorry, i am unable to assist",
            "i cannot assist with",
            "i'm not able to provide",
            "i cannot provide",
            "i'm unable to help",
            "no puedo asistir",
            "no puedo ayudar",
        ]

        is_negative_response = any(
            phrase in full_response.lower() for phrase in negative_responses
        )

        words_count = len(full_response.split()) if full_response else 0

        if is_negative_response:
            logger.info(
                f"El agente {agent_id} rechazó responder a la consulta por políticas internas"
            )
            return {
                "model": f"Bedrock Agent: {agent_alias_id}",
                "response": full_response,
                "time": round(elapsed, 2),
                "words_estimate": words_count,
                "status": "policy_rejection",
                "error_code": "PolicyRejection",
            }
        else:
            return {
                "model": f"Bedrock Agent: {agent_alias_id}",
                "response": full_response,
                "time": round(elapsed, 2),
                "words_estimate": words_count,
                "status": "success",
            }

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_msg = e.response.get("Error", {}).get("Message", str(e))

        logger.error(f"Error al invocar agente: {error_code} - {error_msg}")

        error_messages = {
            "AccessDeniedException": f"❌ Error de permisos: No tiene acceso al agente de Bedrock.",
            "ResourceNotFoundException": f"❌ Error: Agente o alias no encontrado. Verifique los IDs proporcionados.",
            "ThrottlingException": f"❌ Error de limitación: Se ha excedido el límite de peticiones.",
            "ValidationException": f"❌ Error de validación: {error_msg}",
            "ServiceUnavailableException": f"❌ Servicio no disponible: Agents for Bedrock está temporalmente no disponible.",
        }

        custom_message = error_messages.get(
            error_code, f"❌ Error de AWS ({error_code}): {error_msg}"
        )

        return {
            "model": "Bedrock Agent",
            "response": custom_message,
            "time": None,
            "tokens": None,
            "words_estimate": None,
            "status": "error",
            "error_code": error_code,
        }
    except Exception as e:
        logger.error(f"Error inesperado al invocar agente: {str(e)}")
        return {
            "model": "Bedrock Agent",
            "response": f"❌ Error inesperado: {str(e)}",
            "time": None,
            "tokens": None,
            "words_estimate": None,
            "status": "error",
            "error_code": "Unknown",
        }


def verify_bedrock_access(model_ids=None):
    if model_ids is None:
        model_ids = selected_models

    try:
        client = boto3.client("bedrock-runtime", region_name=aws_region)
        st.session_state.client = client

        try:
            client.list_foundation_models()
        except AttributeError:
            pass

        valid_models = []
        invalid_models = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, model_id in enumerate(model_ids):
            status_text.text(f"Verificando acceso al modelo: {model_id}")
            progress_value = (i + 1) / len(model_ids)
            progress_bar.progress(progress_value)

            try:
                inference_config = {"maxTokens": 10, "temperature": 0.5, "topP": 0.9}
                messages = [{"role": "user", "content": [{"text": "test"}]}]

                client.converse(
                    modelId=model_id,
                    messages=messages,
                    inferenceConfig=inference_config,
                )
                valid_models.append(model_id)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                error_msg = e.response.get("Error", {}).get("Message", str(e))

                if error_code == "AccessDeniedException":
                    invalid_models.append((model_id, f"Sin acceso: {error_msg}"))
                elif error_code == "ValidationException":
                    invalid_models.append((model_id, f"Modelo inválido: {error_msg}"))
                else:
                    invalid_models.append((model_id, f"Error: {error_msg}"))
            except Exception as e:
                invalid_models.append((model_id, f"Error inesperado: {str(e)}"))

        progress_bar.empty()
        status_text.empty()

        return {
            "success": len(valid_models) > 0,
            "valid_models": valid_models,
            "invalid_models": invalid_models,
        }

    except NoCredentialsError:
        logger.error("No se encontraron credenciales de AWS")
        return {
            "success": False,
            "error": "No se encontraron credenciales de AWS. Verifique su configuración de AWS CLI o variables de entorno.",
        }
    except CredentialRetrievalError as e:
        logger.error(f"Error al recuperar credenciales: {str(e)}")
        return {"success": False, "error": f"Error al recuperar credenciales: {str(e)}"}
    except EndpointConnectionError as e:
        logger.error(f"Error de conexión al endpoint de Bedrock: {str(e)}")
        return {
            "success": False,
            "error": f"Error de conexión al endpoint de Bedrock: {str(e)}",
        }
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_msg = e.response.get("Error", {}).get("Message", str(e))

        logger.error(f"Error de cliente AWS ({error_code}): {error_msg}")
        if error_code == "AccessDeniedException":
            return {
                "success": False,
                "error": f"Acceso denegado a Amazon Bedrock: {error_msg}",
            }
        else:
            return {
                "success": False,
                "error": f"Error de cliente AWS ({error_code}): {error_msg}",
            }
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        return {"success": False, "error": f"Error inesperado: {str(e)}"}


def init_bedrock_client():
    try:
        client = boto3.client("bedrock-runtime", region_name=aws_region)
        st.session_state.client = client
        return True
    except Exception as e:
        logger.error(f"Error al crear cliente Bedrock: {str(e)}")
        st.error(f"Error al inicializar el cliente de Amazon Bedrock: {str(e)}")
        return False


def load_and_split_txts(txt_paths, chunk_size=500, chunk_overlap=100):
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    with st.spinner("Procesando documentos..."):
        progress_bar = st.progress(0)
        for i, path in enumerate(txt_paths):
            try:
                loader = TextLoader(str(path), encoding="utf-8")
                docs = loader.load()
                chunks = splitter.split_documents(docs)
                all_docs.extend(chunks)
                progress = (i + 1) / len(txt_paths)
                progress_bar.progress(progress)
            except Exception as e:
                logger.error(f"Error al procesar {path}: {str(e)}")
                st.error(f"Error al procesar {path}: {str(e)}")

    return all_docs


def setup_embeddings_and_vectorstore(documents):
    from langchain.vectorstores import FAISS
    from langchain.embeddings import BedrockEmbeddings

    try:
        with st.spinner("Configurando modelo de embeddings..."):
            embeddings = BedrockEmbeddings(client=st.session_state.client)

            test_result = embeddings.embed_query(
                "Texto de prueba para verificar embeddings"
            )
            if (
                not test_result
                or not isinstance(test_result, list)
                or len(test_result) < 10
            ):
                raise ValueError(
                    "La respuesta de embeddings no tiene el formato esperado"
                )

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
            display_message = (
                "El modelo de embeddings solicitado no está disponible en su región."
            )
        elif "ThrottlingException" in error_msg:
            display_message = "Se ha superado el límite de solicitudes para el servicio de embeddings."
        elif "ValidationException" in error_msg:
            display_message = "Error de validación en la solicitud de embeddings."
        elif "ServiceUnavailableException" in error_msg:
            display_message = (
                "El servicio de embeddings no está disponible en este momento."
            )
        elif (
            "UnrecognizedClientException" in error_msg
            or "No credential provider" in error_msg
        ):
            display_message = "Credenciales de AWS no válidas o no encontradas."
        else:
            display_message = f"Error al configurar embeddings: {error_msg}"

        logger.error(f"Error de embeddings: {display_message}")
        st.error(f"❌ Error en la configuración de embeddings: {display_message}")

        return None, False


async def invoke_converse(model_id, messages, inference_config):
    if st.session_state.client is None:
        logger.error("Client is None, Bedrock client no inicializado")
        raise RuntimeError("Cliente de Bedrock no inicializado correctamente.")

    try:
        logger.info(f"Invocando modelo {model_id}")
        response = st.session_state.client.converse(
            modelId=model_id, messages=messages, inferenceConfig=inference_config
        )

        if not isinstance(response, dict):
            logger.error(
                f"Respuesta inesperada de tipo {type(response)} para modelo {model_id}"
            )
            raise RuntimeError(f"Respuesta inesperada del modelo {model_id}")

        if "output" not in response:
            logger.error(
                f"Respuesta sin campo 'output' del modelo {model_id}: {response}"
            )
            raise RuntimeError(
                f"Respuesta del modelo {model_id} no tiene el formato esperado"
            )

        if "message" not in response["output"]:
            logger.error(
                f"Respuesta sin campo 'message' del modelo {model_id}: {response}"
            )
            raise RuntimeError(
                f"Respuesta del modelo {model_id} no tiene el formato esperado"
            )

        return response

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_msg = e.response.get("Error", {}).get("Message", str(e))

        if error_code == "AccessDeniedException":
            logger.error(f"Acceso denegado al modelo {model_id}: {error_msg}")
            raise RuntimeError(
                f"Acceso denegado al modelo {model_id}. Verifique los permisos IAM."
            )
        elif (
            error_code == "ValidationException"
            and "model not found" in error_msg.lower()
        ):
            logger.error(f"Modelo {model_id} no encontrado: {error_msg}")
            raise RuntimeError(
                f"Modelo {model_id} no encontrado o no disponible en su región."
            )
        elif error_code == "ThrottlingException":
            logger.error(f"Límite de tasa excedido para modelo {model_id}: {error_msg}")
            raise RuntimeError(
                f"Límite de tasa excedido para el modelo {model_id}. Intente nuevamente más tarde."
            )
        elif error_code == "ServiceQuotaExceededException":
            logger.error(
                f"Cuota de servicio excedida para modelo {model_id}: {error_msg}"
            )
            raise RuntimeError(f"Cuota de servicio excedida para el modelo {model_id}.")
        else:
            logger.error(
                f"Error al invocar modelo {model_id}: {error_code} - {error_msg}"
            )
            raise RuntimeError(f"Error al invocar modelo {model_id}: {error_msg}")
    except Exception as e:
        logger.error(f"Error inesperado al invocar modelo {model_id}: {str(e)}")
        raise RuntimeError(f"Error inesperado al invocar modelo {model_id}: {str(e)}")


async def run_chain(model_id, query, context, word_limit):
    try:
        prompt = f"""
        <instrucciones_agente>
            <proposito>
                Eres un asistente de información bancaria.
                Tu única función es responder exclusivamente con base en los documentos internos proporcionados.
            </proposito>

            En la base de conocimeintos tienes información sobre las siguientes categorías:
            <categorias>
                <categoria>Cuentas de Débito y Tarjetas</categoria>
                <categoria>Ahorro (Guardadito)</categoria>
                <categoria>Inversiones</categoria>
                <categoria>Créditos y Financiamiento</categoria>
                <categoria>Nómina y Portabilidad</categoria>
                <categoria>Pagos y Transferencias</categoria>
                <categoria>Retiros y Efectivo</categoria>
                <categoria>Seguridad y Acceso</categoria>
                <categoria>Servicios Adicionales</categoria>
            </categorias>

            <protocolo_respuesta>
                <regla>Brindar respuestas breves y precisas</regla>
                <regla>Usar lenguaje natural y amigable</regla>
                <regla>Seguir las políticas internas de SOLTIVA</regla>
                <regla>Mantener las respuestas claras, útiles y concisas</regla>
                <regla>Utilizar menos de 30 palabras para cada respuesta</regla>
            </protocolo_respuesta>

            <consultas_fuera_alcance>
                <accion>Rechazar amablemente preguntas fuera de las categorías permitidas</accion>
                <accion>Redirigir al usuario mencionando los temas disponibles</accion>
                <accion>No proporcionar información sobre temas no autorizados</accion>
            </consultas_fuera_alcance>

            <manejo_datos>
                <regla>Utilizar únicamente documentación interna autorizada</regla>
                <regla>No acceder ni compartir datos sensibles de clientes</regla>
                <regla>Cumplir con todos los protocolos de seguridad de la información</regla>
            </manejo_datos>

            <estilo_comunicacion>
                <tono>Profesional y cortés</tono>
                <tono>Consistente con la voz de la marca</tono>
                <tono>Enfocado en la claridad y eficiencia</tono>
                <tono>Respuestas relevantes y concisas</tono>
            </estilo_comunicacion>

            <reglas>
                <regla>No inventes ni supongas información, aunque parezca razonable.</regla>
                <regla>No uses frases introductorias como "para abrir una cuenta necesitas...". Sé directo.</regla>
                <regla>Si la información no está en los documentos, responde "No tengo esa información."</regla>
                <regla>Tu respuesta no debe superar {word_limit} palabras bajo ninguna circunstancia.</regla>
            </reglas>

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
        <instrucciones_agente>
        """

        messages = [{"role": "user", "content": [{"text": prompt}]}]

        inference_config = {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        }

        start = time.time()

        response = await invoke_converse(model_id, messages, inference_config)

        if not response or "output" not in response:
            logger.error(f"Respuesta inválida o nula del modelo {model_id}")
            raise ValueError(f"Respuesta inválida del modelo {model_id}")

        output_message = response["output"].get("message", {})
        if (
            not output_message
            or "content" not in output_message
            or not output_message["content"]
        ):
            logger.error(
                f"Formato de mensaje inesperado del modelo {model_id}: {response}"
            )
            raise ValueError(f"Formato de respuesta inválido del modelo {model_id}")

        result = output_message["content"][0]["text"]

        output_tokens = response.get("usage", {}).get("outputTokens", "N/A")

        if isinstance(output_tokens, int):
            words_estimate = output_tokens / 2
        else:
            words_estimate = "N/A"

        elapsed = time.time() - start
        return {
            "model": model_id,
            "response": result.strip(),
            "time": round(elapsed, 2),
            "tokens": output_tokens,
            "words_estimate": words_estimate,
            "status": "success",
        }

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_msg = e.response.get("Error", {}).get("Message", "")

        error_messages = {
            "AccessDeniedException": f"❌ Error de permisos: No tiene acceso al modelo {model_id}.",
            "ThrottlingException": f"❌ Error de limitación: Se ha excedido el límite de peticiones para {model_id}.",
            "ValidationException": f"❌ Error de validación: {error_msg}",
            "ServiceUnavailableException": f"❌ Servicio no disponible: Amazon Bedrock está temporalmente no disponible.",
            "InternalServerException": f"❌ Error interno del servidor: Problema en Amazon Bedrock.",
        }

        custom_message = error_messages.get(
            error_code, f"❌ Error de AWS ({error_code}): {error_msg}"
        )
        logger.error(f"Error al invocar {model_id}: {error_code} - {error_msg}")

        return {
            "model": model_id,
            "response": custom_message,
            "time": None,
            "tokens": None,
            "words_estimate": None,
            "status": "error",
            "error_code": error_code,
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
            "error_code": "NoCredentials",
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
            "error_code": "Runtime",
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
            "error_code": "Unknown",
        }


async def run_all_models(query, docs, word_limit=50):
    context = "\n\n".join([d.page_content for d in docs])

    if st.session_state.client is None:
        logger.error("No se puede ejecutar consulta: cliente Bedrock no inicializado")
        return [
            {
                "model": "SYSTEM",
                "response": "❌ Error: Cliente de Amazon Bedrock no inicializado. Verifique sus credenciales y permisos.",
                "time": None,
                "tokens": None,
                "words_estimate": None,
                "status": "error",
                "error_code": "NoClient",
            }
        ]

    valid_models = [mid for mid in selected_models if mid and isinstance(mid, str)]
    if not valid_models:
        logger.error("No hay modelos válidos configurados")
        return [
            {
                "model": "SYSTEM",
                "response": "❌ Error: No hay modelos válidos configurados.",
                "time": None,
                "tokens": None,
                "words_estimate": None,
                "status": "error",
                "error_code": "NoModels",
            }
        ]

    try:
        logger.info(f"Ejecutando consulta en {len(valid_models)} modelos")
        tasks = []
        for mid in valid_models:
            task = run_chain(mid, query, context, word_limit)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            mid = valid_models[i]
            if isinstance(result, Exception):
                logger.error(f"Error al procesar modelo {mid}: {str(result)}")
                processed_results.append(
                    {
                        "model": mid,
                        "response": f"❌ Error: {str(result)}",
                        "time": None,
                        "tokens": None,
                        "words_estimate": None,
                        "status": "error",
                        "error_code": "ExecutionError",
                    }
                )
            else:
                processed_results.append(result)

        return processed_results
    except Exception as e:
        logger.error(f"Error al ejecutar consultas en paralelo: {str(e)}")
        return [
            {
                "model": "SYSTEM",
                "response": f"❌ Error general al ejecutar modelos: {str(e)}",
                "time": None,
                "tokens": None,
                "words_estimate": None,
                "status": "error",
                "error_code": "ParallelExecutionError",
            }
        ]


def display_model_results(results, query=""):
    comparison_data = []
    for r in results:
        status = r.get("status", "unknown")
        model_name = r["model"]

        if status == "success":
            comparison_data.append(
                {
                    "Pregunta": query,
                    "Modelo": model_name,
                    "Tiempo (s)": r["time"],
                    "Tokens": r.get("tokens", "N/A"),
                    "Palabras": r.get("words_estimate", "N/A"),
                    "Estado": "✅ OK",
                    "Respuesta": r["response"],
                }
            )
        elif status == "policy_rejection":
            comparison_data.append(
                {
                    "Pregunta": query,
                    "Modelo": model_name,
                    "Tiempo (s)": r["time"],
                    "Tokens": r.get("tokens", "N/A"),
                    "Palabras": r.get("words_estimate", "N/A"),
                    "Estado": "ℹ️ Rechazada por políticas",
                    "Respuesta": r["response"],
                }
            )
        else:
            error_code = r.get("error_code", "Unknown")
            comparison_data.append(
                {
                    "Pregunta": query,
                    "Modelo": model_name,
                    "Tiempo (s)": "N/A",
                    "Tokens": "N/A",
                    "Palabras": "N/A",
                    "Estado": f"❌ Error: {error_code}",
                    "Respuesta": r["response"],
                }
            )

    st.subheader("📊 Comparación de modelos")
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)

    st.session_state.last_comparison_data = comparison_data

    if st.download_button(
        label="📥 Exportar resultados a CSV",
        data=pd.DataFrame(comparison_data).to_csv(index=False).encode("utf-8"),
        file_name="comparacion_modelos.csv",
        mime="text/csv",
        help="Descarga los resultados de la comparación en formato CSV",
    ):
        st.success("✅ Archivo CSV descargado exitosamente")

    st.subheader("🤖 Respuestas por modelo")
    for r in results:
        with st.expander(f"🧠 {r['model']}", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Tiempo de respuesta", f"{r['time']} segundos")
            with col2:
                tokens_val = r.get("tokens")
                if tokens_val is None or tokens_val == "N/A":
                    words_val = r.get("words_estimate", "N/A")
                    st.metric("Palabras generadas", words_val)
                else:
                    st.metric("Tokens generados", tokens_val)

            st.markdown("### Respuesta:")
            st.markdown(r["response"])


if st.session_state.client is None:
    init_bedrock_client()
    if st.session_state.client is not None:
        access_status = verify_bedrock_access()
        if not access_status["success"]:
            st.error(
                f"❌ Error de acceso a Bedrock: {access_status.get('error', 'Error desconocido')}"
            )

if not use_bedrock_agent:
    st.subheader("1️⃣ Cargar archivos de texto")
    uploaded_files = st.file_uploader(
        "Arrastra o haz clic para seleccionar archivos de texto",
        type=["txt"],
        accept_multiple_files=True,
    )

    upload_dir = Path("uploaded_txts")
    upload_dir.mkdir(exist_ok=True)

    if uploaded_files and not st.session_state.has_uploaded_files:
        saved_files = []
        for file in uploaded_files:
            file_path = upload_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_files.append(file_path)

        st.success(f"✅ {len(saved_files)} archivos guardados exitosamente")

        try:
            documents = load_and_split_txts(
                saved_files, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            st.session_state.documents = documents
            st.session_state.has_uploaded_files = True
            st.success(f"✅ Se procesaron {len(documents)} fragmentos de texto")
        except Exception as e:
            logger.error(f"Error al procesar documentos: {str(e)}")
            st.error(f"Error al procesar documentos: {str(e)}")

    if (
        st.session_state.has_uploaded_files
        and st.session_state.documents
        and st.session_state.vectorstore is None
    ):
        if st.session_state.client is not None:
            vectorstore, success = setup_embeddings_and_vectorstore(
                st.session_state.documents
            )
            if success:
                st.session_state.vectorstore = vectorstore
else:
    st.info(
        "ℹ️ Modo Agente de Bedrock activado: Las consultas se enviarán directamente al agente sin procesar documentos locales."
    )

if use_bedrock_agent:
    st.subheader("2️⃣ Hacer preguntas al agente de Bedrock")
else:
    st.subheader("2️⃣ Hacer preguntas a los documentos")

is_ready = st.session_state.client is not None and (
    (
        not use_bedrock_agent
        and st.session_state.has_uploaded_files
        and st.session_state.documents
        and st.session_state.vectorstore is not None
    )
    or (use_bedrock_agent and bedrock_agent_id and bedrock_agent_alias_id)
)

if not is_ready:
    if use_bedrock_agent and (not bedrock_agent_id or not bedrock_agent_alias_id):
        st.warning(
            "⚠️ Debe proporcionar el ID del agente y el ID del alias para usar un agente de Bedrock."
        )
    elif use_bedrock_agent and st.session_state.client is None:
        st.warning(
            "⚠️ Error al inicializar el cliente de Bedrock. Verifique sus credenciales de AWS."
        )
    elif not use_bedrock_agent and not st.session_state.has_uploaded_files:
        st.warning("⚠️ Primero debe cargar archivos de texto para procesarlos.")
    elif not use_bedrock_agent and st.session_state.vectorstore is None:
        st.warning(
            "⚠️ Espere a que se configuren los modelos de embeddings y el índice vectorial."
        )
    else:
        st.warning(
            "⚠️ La aplicación no está lista para procesar consultas. Verifique la configuración."
        )
else:
    query = st.text_input(
        "Escribe tu pregunta aquí:",
        placeholder="cuanto cuesta personalizar el plástico de guardadito kids?",
    )

    if st.button("📝 Realizar consulta", type="primary"):
        if not query:
            st.warning("⚠️ Por favor ingresa una pregunta")
        elif not use_bedrock_agent and not selected_models:
            st.warning("⚠️ Selecciona al menos un modelo en la barra lateral")
        else:
            try:
                if use_bedrock_agent:
                    with st.spinner("Consultando agente de Bedrock..."):
                        try:
                            result = asyncio.run(
                                invoke_bedrock_agent(
                                    bedrock_agent_id, bedrock_agent_alias_id, query
                                )
                            )
                            display_model_results([result], query=query)
                        except Exception as e:
                            st.error(
                                f"❌ Error al invocar el agente de Bedrock: {str(e)}"
                            )
                            logger.error(
                                f"Error al invocar el agente de Bedrock: {str(e)}"
                            )
                else:
                    with st.spinner("Buscando documentos relevantes..."):
                        relevant_docs = st.session_state.vectorstore.similarity_search(
                            query, k=k_docs
                        )
                        if not relevant_docs:
                            st.warning(
                                "⚠️ No se encontraron documentos relevantes para esta consulta"
                            )

                        with st.expander(
                            "📄 Ver fragmentos relevantes", expanded=False
                        ):
                            for i, doc in enumerate(relevant_docs):
                                st.markdown(f"**Fragmento {i+1}:**")
                                st.text(
                                    doc.page_content[:300] + "..."
                                    if len(doc.page_content) > 300
                                    else doc.page_content
                                )
                                st.markdown("---")

                    with st.spinner("Consultando modelos de Bedrock..."):
                        results = asyncio.run(
                            run_all_models(query, relevant_docs, word_limit)
                        )
                        display_model_results(results, query=query)
            except Exception as e:
                logger.error(f"Error al ejecutar la consulta: {str(e)}")
                st.error(f"❌ Error al ejecutar la consulta: {str(e)}")

st.markdown("---")
st.markdown(
    "📘 **Bedrock Multi-Model RAG App** - Powered by Amazon Bedrock, LangChain, and Streamlit"
)
