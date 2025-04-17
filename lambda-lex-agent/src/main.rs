use lambda_runtime::{run, service_fn, Error, LambdaEvent};
use serde_json::{json, Value};
use indexmap::IndexMap;
use std::env;

use aws_sdk_bedrockagentruntime::config::Region;
use aws_sdk_bedrockagentruntime as bedrock;
use aws_sdk_bedrockagentruntime::types::ResponseStream;

#[tokio::main]
async fn main() -> Result<(), Error> {
    eprintln!("🚀 Lambda is starting up");
    run(service_fn(handler)).await
}

async fn handler(event: LambdaEvent<Value>) -> Result<Value, Error> {
    eprintln!("📥 Received event: {:?}", event);
    let (event, _context) = event.into_parts();

    let session_state = event["sessionState"].as_object().ok_or("Missing sessionState")?;
    let intent = session_state["intent"].as_object().ok_or("Missing intent")?;
    let intent_name = intent["name"].as_str().unwrap_or_default();

    eprintln!("🎯 Detected intent: {}", intent_name);

    if intent_name == "FallbackIntent" {
        return handle_fallback(&event).await;
    }

    eprintln!("⚠️ Unsupported intent: {}", intent_name);
    Ok(json!({
        "statusCode": 400,
        "body": "Unsupported intent"
    }))
}

async fn handle_fallback(event: &Value) -> Result<Value, Error> {
    eprintln!("🤖 Executing handle_fallback...");

    let input_transcript = event["inputTranscript"].as_str().unwrap_or_default();
    eprintln!("💬 inputTranscript: {}", input_transcript);

    let session_state   = &event["sessionState"];
    let default_attrs   = serde_json::Map::new();
    let session_attrs   = session_state["sessionAttributes"]
        .as_object()
        .unwrap_or(&default_attrs);

    let original_customer_id = session_attrs
        .get("CustomerId")
        .and_then(|v| v.as_str())
        .unwrap_or("SESSION_ID") 
        .to_string();

    let customer_id = original_customer_id.replace('+', "");

    if input_transcript.trim().is_empty() {
        eprintln!("⚠️ Empty input transcript detected, returning null content response");

        let mut ordered_session_attrs = IndexMap::new();
        ordered_session_attrs.insert(
            "available_intents".to_string(),
            Value::String("FallbackIntent".to_string()),
        );
        ordered_session_attrs.insert(
            "CustomerId".to_string(),
            Value::String(original_customer_id.clone()),
        );

        let session_state_map = json!({
            "dialogAction": { "type": "Close" },
            "intent": {
                "name":  "FallbackIntent",
                "slots": {},
                "state": "Fulfilled"
            },
            "sessionAttributes": ordered_session_attrs
        });

        let messages_array = json!([
            { "contentType": "PlainText", "content": null }
        ]);

        let mut response = IndexMap::new();
        response.insert("sessionState".to_string(), session_state_map);
        response.insert("messages".to_string(),     messages_array);

        return Ok(serde_json::to_value(response)?);
    }

    let result = query_agent(input_transcript, &customer_id).await?;

    let mut final_session_attrs = IndexMap::new();
    final_session_attrs.insert(
        "CustomerId".to_string(),
        Value::String(original_customer_id), 
    );

    let final_session_state_map = json!({
        "dialogAction": { "type": "Close" },
        "intent": {
            "name":  "FallbackIntent",
            "slots": session_state["intent"]["slots"].clone(),
            "state": "Fulfilled"
        },
        "sessionAttributes": final_session_attrs
    });

    let final_messages_array = json!([
        { "contentType": "PlainText", "content": result }
    ]);

    let mut response = IndexMap::new();
    response.insert("sessionState".to_string(), final_session_state_map);
    response.insert("messages".to_string(),     final_messages_array);

    Ok(serde_json::to_value(response)?)
}

async fn query_agent(question: &str, session_id: &str) -> Result<String, Error> {
    eprintln!("📡 Starting query_agent with question: '{}'", question);
    let region_str = env::var("BEDROCK_REGION").map_err(|_| "BEDROCK_REGION must be set")?;
    eprintln!("🌎 Region: {}", region_str);

    let region = Region::new(region_str);
    let config = aws_config::from_env().region(region).load().await;
    let client = bedrock::Client::new(&config);

    let agent_id = env::var("AGENT_ID").map_err(|e| {
        eprintln!("❌ AGENT_ID not defined: {}", e);
        e
    })?;

    let agent_alias_id = env::var("AGENT_ALIAS_ID").map_err(|e| {
        eprintln!("❌ AGENT_ALIAS_ID not defined: {}", e);
        e
    })?;

    eprintln!("🪪 Agent ID: {}, Alias ID: {}", agent_id, agent_alias_id);

    let resp_result = client.invoke_agent()
        .agent_id(agent_id)
        .agent_alias_id(agent_alias_id)
        .session_id(session_id)
        .input_text(question)
        .enable_trace(false)
        .end_session(false)
        .send()
        .await;

    let resp = match resp_result {
        Ok(r) => {
            eprintln!("✅ Bedrock Agent successfully invoked.");
            r
        },
        Err(e) => {
            eprintln!("❌ Error invoking Bedrock Agent: {:?}", e);
            return Ok(format!("Sorry, there was an error contacting the agent: {:?}", e).to_string());
        }
    };

    let mut full_response = String::new();
    let mut receiver = resp.completion;

    loop {
        match receiver.recv().await {
            Ok(Some(event)) => {
                match event {
                    ResponseStream::Chunk(part) => {
                        if let Some(bytes) = part.bytes {
                            if let Ok(text) = std::str::from_utf8(bytes.as_ref()) {
                                eprintln!("📦 Received chunk: {}", text);
                                full_response.push_str(text);
                            }
                        }
                    }
                    _ => eprintln!("ℹ️ Non-chunk event received"),
                }
            }
            Ok(None) => {
                eprintln!("✅ End of response");
                break;
            }
            Err(e) => {
                eprintln!("❌ Error in receiver.recv(): {:?}", e);
                break;
            }
        }
    }

    Ok(if full_response.is_empty() {
        eprintln!("⚠️ Empty response");
        "No response".into()
    } else {
        eprintln!("✅ Full response: {}", full_response);
        full_response
    })
}
