import json
import sys
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI


def ask_prompts_batch(
    client: OpenAI,
    prompts: List[str],
    model: str = "gpt-4o-mini",
    system_message: Optional[str] = None,
    response_format_type: Optional[str] = "json_object",
    json_schema: Optional[Dict[str, Any]] = None,
    max_tokens: int = 6000,
    temperature: float = 1.0,
    timeout: int = 3600*24, # 24 hours
    check_interval: int = 60
) -> List[str]:
    assert response_format_type is not None or json_schema is not None, "Either response_format_type or json_schema must be provided"

    print(f"Preparing batch request for {len(prompts)} prompts...")
    print(f"Model: {model}, Max Tokens: {max_tokens}, Temperature: {temperature}, Timeout: {timeout}s")
    
    # Create JSONL file with batch requests
    batch_requests = []
    for i, prompt in enumerate(prompts):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add response format (prioritize json_schema if provided)
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema
            }
        else:
            body["response_format"] = {"type": response_format_type}
        
        request = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        batch_requests.append(request)
    
    # Write to temporary JSONL file
    batch_file_path = Path("batch_input.jsonl")
    with open(batch_file_path, "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")
    
    try:
        # Upload the batch file
        print(f"Uploading batch file with {len(prompts)} prompts...")
        with open(batch_file_path, "rb") as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
        
        # Create the batch job
        print(f"Creating batch job...")
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        # Wait for batch to complete
        print(f"Batch created with ID: {batch.id}")
        print(f"Waiting for batch to complete...")
        
        start_time = time.time()
        while True:
            batch = client.batches.retrieve(batch.id)
            print(f"Status: {batch.status} | Completed: {batch.request_counts.completed}/{batch.request_counts.total}")
            
            if batch.status == "completed":
                break
            elif batch.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch failed with status: {batch.status}")
            
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch did not complete within {timeout} seconds")
            
            time.sleep(check_interval)
        
        # Download and process results
        print("Downloading results...")
        result_file_id = batch.output_file_id
        
        # Check if there's an error file instead of output file
        if result_file_id is None:
            if batch.error_file_id:
                print(f"Batch completed with errors. Downloading error file")
                error_content = client.files.content(batch.error_file_id)
                error_text = error_content.text
                print(f"Error file content:\n{error_text[:1000]}")  # Show first 1000 chars
                raise Exception(f"Batch failed. Error file ID: {batch.error_file_id}")
            else:
                raise Exception(f"Batch completed but no output_file_id or error_file_id available. Batch status: {batch.status}")
        
        result_content = client.files.content(result_file_id)
        result_text = result_content.text
        
        # Parse results
        results = {}
        for line in result_text.strip().split("\n"):
            result_obj = json.loads(line)
            custom_id = result_obj["custom_id"]
            request_idx = int(custom_id.split("-")[1])
            
            if result_obj.get("response"):
                response_body = result_obj["response"]["body"]
                # Check if response body contains an error instead of choices
                if "error" in response_body:
                    error = response_body["error"]
                    print(f"Request {request_idx} failed with error: {error}")
                    results[request_idx] = f"Error: {error}"
                else:
                    response = response_body["choices"][0]["message"]["content"]
                    results[request_idx] = response
            else:
                # Handle errors at request level
                error = result_obj.get("error", {})
                print(f"Request {request_idx} failed at request level: {error}")
                results[request_idx] = f"Error: {error}"
        
        # Return results in original order
        ordered_results = [results.get(i, None) for i in range(len(prompts))]
        
        print(f"Batch completed successfully! Processed {len(ordered_results)} prompts.")
        return ordered_results
        
    finally:
        # Cleanup temporary file
        if batch_file_path.exists():
            batch_file_path.unlink()


def ask_prompts_sequential(
    client: OpenAI,
    prompts: List[str],
    model: str = "gpt-4o-mini",
    system_message: Optional[str] = None,
    response_format_type: Optional[str] = "json_object",
    json_schema: Optional[Dict[str, Any]] = None,
    max_tokens: int = 6000,
    temperature: float = 1.0,
    max_concurrent: int = 1
) -> List[str]:
    assert response_format_type is not None or json_schema is not None, "Either response_format_type or json_schema must be provided"
    
    async def process_prompts_async():
        """Internal async function to process prompts concurrently."""
        async_client = AsyncOpenAI(api_key=client.api_key, base_url=client.base_url)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_prompt(i: int, prompt: str) -> tuple[int, str]:
            """Process a single prompt with rate limiting."""
            async with semaphore:
                print(f"Processing prompt {i+1}/{len(prompts)}...")
                
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                
                # Build response_format parameter
                if json_schema is not None:
                    response_format = {
                        "type": "json_schema",
                        "json_schema": json_schema
                    }
                else:
                    response_format = {"type": response_format_type}
                
                try:
                    response = await async_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format=response_format
                    )
                    content = response.choices[0].message.content
                    return (i, content)
                except Exception as e:
                    print(f"Error processing prompt {i+1}: {str(e)}")
                    print(f"Error processing prompt {i+1}: {str(e)}", file=sys.stderr)
                    print(f"Problematic messages:\n {messages} \n", file=sys.stderr)
                    return (i, f"Error: {str(e)}")
        
        # Create tasks for all prompts
        tasks = [process_single_prompt(i, prompt) for i, prompt in enumerate(prompts)]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Close the async client
        await async_client.close()
        
        return results
    
    # Run async function and get results
    results = asyncio.run(process_prompts_async())
    
    # Sort results by index to maintain original order
    sorted_results = sorted(results, key=lambda x: x[0])
    responses = [content for _, content in sorted_results]
    
    print(f"Completed processing {len(responses)} prompts.")
    return responses


def ask_prompts(
    client: OpenAI,
    prompts: List[str],
    model: str = "gpt-4o-mini",
    batch_process: bool = True,
    response_format_type: Optional[str] = "json_object",
    json_schema: Optional[Dict[str, Any]] = None,
    temperature: float = 1.0,
    **kwargs
) -> List[str]:
    if batch_process:
        return ask_prompts_batch(client, prompts, model=model, response_format_type=response_format_type, json_schema=json_schema, max_tokens=6000, temperature=temperature, **kwargs)
    else:
        return ask_prompts_sequential(client, prompts, model=model, response_format_type=response_format_type, json_schema=json_schema, max_tokens=6000, temperature=temperature, **kwargs)

