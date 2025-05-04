use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct PromptRequest {
    prompt: String,
    // Add other parameters as needed by the API
    // max_tokens: u32,
    // temperature: f32,
}

#[derive(Deserialize, Debug)]
struct PromptResponse {
    // Define the structure based on the actual response from the API
    // For example:
    completion: String,
}

async fn llama_chat_completion(prompt: &str, api_url: &str, api_key: &str) -> Result<PromptResponse, reqwest::Error> {
    let client = reqwest::Client::new();
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert("Authorization", HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());

    let request_body = PromptRequest {
        prompt: prompt.to_string(),
        // Initialize other parameters as needed
    };

    let res = client
        .post(api_url)
        .headers(headers)
        .json(&request_body)
        .send()
        .await?
        .json::<PromptResponse>()
        .await?;

    Ok(res)
}

#[tokio::main]
//async fn main() {
//    let api_url = "https://your-llama-api-endpoint.com/completion"; // Replace with your actual API endpoint
//    let api_key = "your_api_key_here"; // Replace with your actual API key
//    let prompt = "Hello, how are you?";
//
//    match llama_chat_completion(prompt, api_url, api_key).await {
//        Ok(response) => println!("{:?}", response),
//        Err(e) => eprintln!("Error: {}", e),
//    }
//}
