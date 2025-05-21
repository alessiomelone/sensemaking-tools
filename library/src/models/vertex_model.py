# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Module to interact with models available on Google Cloud's Model Garden

import asyncio
import json
from typing import Any, Callable, Dict, List, TypedDict, Optional

# Attempt to import Google Cloud Vertex AI SDK components
# These are based on common patterns; actual names might vary slightly.
try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1beta1.types import ( # Or aiplatform.gapic.VertexAI or similar
        HarmCategory as SDKHarmCategory,
        HarmBlockThreshold as SDKHarmBlockThreshold,
        Schema as SDKSchema, # This might be dict or a specific class
        GenerationConfig as SDKGenerationConfig,
        SafetySetting as SDKSafetySetting,
        Content as SDKContent, # For response parsing
        Part as SDKPart, # For response parsing
    )
    # GenerativeModel and VertexAI might be part of aiplatform or aiplatform.gapic
    from google.cloud.aiplatform_v1beta1 import PredictionServiceClient # Placeholder
    from google.cloud.aiplatform.models import GenerativeModel as SDKGenerativeModel # More likely
    from google.cloud.aiplatform importgapic # Placeholder for VertexAI client

    # For ModelParams, RequestOptions - these might be dicts or specific classes
    # For now, we'll assume they can be represented as Dicts or use SDK types if found
    SDKModelParams = Dict[str, Any] 
    SDKRequestOptions = Dict[str, Any]
except ImportError:
    print("Warning: Google Cloud Vertex AI SDK not fully available. Some types are placeholders.")
    # Define placeholders if the SDK is not available (e.g., in a test environment)
    SDKHarmCategory = Any 
    SDKHarmBlockThreshold = Any
    SDKSchema = Dict[str, Any] 
    SDKGenerativeModel = Any 
    SDKVertexAI = Any
    SDKModelParams = Dict[str, Any]
    SDKRequestOptions = Dict[str, Any]
    SDKGenerationConfig = Dict[str, Any]
    SDKSafetySetting = Dict[str, Any]
    SDKContent = Any
    SDKPart = Any


from .model import Model, PySchema
from ..types import check_data_schema # Assuming types.py is in the parent directory of models
from .model_util import RETRY_DELAY_MS, DEFAULT_VERTEX_PARALLELISM, MAX_LLM_RETRIES

# Placeholder for retryCall from sensemaker_utils
# This would need to be an async function in Python.
async def retry_call(
    async_fn: Callable[..., Any],
    validator_fn: Callable[[Any], bool],
    max_retries: int,
    error_message: str,
    delay_ms: int,
    fn_args: List[Any] = [],
    validator_args: List[Any] = [],
) -> Any:
    """
    Placeholder for the retryCall utility.
    In a real implementation, this would handle retries with delays for async functions.
    """
    print(f"Warning: Using placeholder for retry_call for function {async_fn.__name__}")
    attempt = 0
    last_exception = None
    while attempt < max_retries:
        try:
            # For placeholder, we assume async_fn doesn't need arguments from fn_args directly here
            # and validator_fn doesn't need validator_args.
            # A real implementation would pass these.
            result = await async_fn(*fn_args) # Pass arguments if needed
            if validator_fn(result, *validator_args): # Pass arguments if needed
                return result
            else:
                print(f"Validation failed for {async_fn.__name__}: {error_message}")
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1} for {async_fn.__name__} failed: {e}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(delay_ms / 1000)
        attempt += 1
    
    if last_exception:
        raise Exception(f"{error_message} after {max_retries} retries. Last error: {last_exception}")
    else:
        raise Exception(f"{error_message} after {max_retries} retries. Validation failed.")


class VertexModel(Model):
    """
    Class to interact with models available through Google Cloud's Model Garden.
    """
    # vertex_ai: SDKVertexAI # Actual type from SDK
    vertex_ai: Any # Using Any for placeholder if SDKVertexAI is not precisely defined
    model_name: str
    limit: asyncio.Semaphore

    def __init__(
        self,
        project: str,
        location: str,
        model_name: str = "gemini-1.5-pro-preview-0409", # Updated to a common model
    ):
        super().__init__()
        try:
            # self.vertex_ai = SDKVertexAI(project=project, location=location) # Example SDK usage
            # For Vertex AI, initialization is often done via aiplatform.init()
            aiplatform.init(project=project, location=location)
            self.vertex_ai = aiplatform # Store the module or a client instance
        except NameError: # SDKVertexAI might not be defined if import failed
             print("Warning: SDKVertexAI not found, VertexModel may not function correctly.")
             self.vertex_ai = None # Or some mock object

        self.model_name = model_name
        
        print(f"Creating VertexModel with {DEFAULT_VERTEX_PARALLELISM} parallel workers...")
        self.limit = asyncio.Semaphore(DEFAULT_VERTEX_PARALLELISM)

    def get_generative_model(self, schema: Optional[PySchema] = None) -> SDKGenerativeModel:
        """
        Get generative model corresponding to structured data output specification.
        """
        # request_options: SDKRequestOptions = {"timeout": 150000} # 2.5 min
        # The Python SDK for GenerativeModel might not use RequestOptions in the same way.
        # It's often part of the model generation call itself or model parameters.

        model_params = get_model_params(self.model_name, schema)
        
        # In Python, GenerativeModel is often instantiated directly
        # return self.vertex_ai.get_generative_model(model_params, request_options) # TS-like
        try:
            # The Python SDK uses a slightly different way to get the model
            model_instance = SDKGenerativeModel(self.model_name)
            # Configuration (like schema, safety settings) might be applied when calling methods
            # on model_instance, or by setting properties if the SDK allows.
            # For now, we assume model_params are handled by `get_model_params` and used in `call_llm`.
            # If `model_params` includes safety settings and generation config, they might be passed
            # to `generate_content_stream` or similar methods.
            return model_instance
        except NameError:
            print("Warning: SDKGenerativeModel not found.")
            return None


    async def generate_text(self, prompt: str) -> str:
        """
        Generate text based on the given prompt.
        """
        model = self.get_generative_model()
        if model is None:
            raise ConnectionError("Failed to initialize the generative model.")
        return await self.call_llm(prompt, model)

    async def generate_data(self, prompt: str, schema: PySchema) -> Any:
        """
        Generate structured data based on the given prompt.
        """
        # `check_data_schema` is assumed to be translated in types.py
        # from ..types import check_data_schema 
        
        def validate_response(response_text: str) -> bool:
            try:
                parsed_response = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"Model returned a non-JSON response:\n{response_text}\n{e}")
                return False
            
            # Assuming check_data_schema is available and works like the TS version
            if not check_data_schema(schema, parsed_response):
                print(f"Model response does not match schema: {response_text}")
                return False
            return True

        model = self.get_generative_model(schema=schema)
        if model is None:
            raise ConnectionError("Failed to initialize the generative model for data generation.")

        response_str = await self.call_llm(prompt, model, validator=validate_response, schema_for_model=schema)
        return json.loads(response_str)

    async def call_llm(
        self,
        prompt: str,
        model: SDKGenerativeModel, # Type from Python SDK
        validator: Callable[[str], bool] = lambda response: True,
        schema_for_model: Optional[PySchema] = None # Pass schema if needed for model call
    ) -> str:
        """
        Calls an LLM to generate text based on a given prompt and handles rate limiting, 
        response validation and retries.
        """
        # req = get_request(prompt) # This is for the older API structure
        # For new SDKs, the prompt is often passed directly, or a simple list of contents.

        async def llm_call_attempt():
            # The Python SDK's generate_content_stream or generate_content might take the prompt directly
            # and other configurations like safety_settings, generation_config (with schema).
            
            # Prepare generation_config and safety_settings if they are passed to generate_content
            model_params_for_call = get_model_params(self.model_name, schema_for_model) # Get full params
            generation_config = model_params_for_call.get('generation_config')
            safety_settings = model_params_for_call.get('safety_settings')

            # The new SDK expects a list of contents, or just the string prompt
            # response_stream = await model.generate_content_stream(req) # Old way
            # response = await response_stream.response # Old way
            
            # Example using generate_content (async) from the SDK
            # The exact method might vary (e.g. generate_content_async)
            # Prompt can be passed directly or as part of a list of contents
            response = await model.generate_content_async(
                contents=[prompt], # Or construct content objects if needed by SDK
                generation_config=generation_config,
                safety_settings=safety_settings,
                # stream=True # if you want to stream, then iterate over response
            )
            return response # This 'response' object structure needs to be known for the validator

        def response_validator(response: Any) -> bool: # `response` type depends on SDK
            if not response:
                print("Failed to get a model response.")
                return False
            
            # Accessing text depends on the SDK's response structure
            # It might be response.text, or response.candidates[0].content.parts[0].text
            # Assuming a structure similar to the TS example for now:
            try:
                # This path needs to be confirmed with Python SDK
                # response.candidates[0].content.parts[0].text
                # If `response` is already the full text, this is simpler.
                # If `response` is a complex object:
                if not response.candidates or not response.candidates[0].content or \
                   not response.candidates[0].content.parts or not response.candidates[0].content.parts[0].text:
                    # Try a simpler path if the above is too complex or response is already simpler
                    if hasattr(response, 'text') and response.text:
                         response_text = response.text
                    else: # Fallback or error if no text found
                        print(f"Model returned an empty or malformed response: {response}")
                        return False
                else: # If the complex path is valid
                    response_text = response.candidates[0].content.parts[0].text

                if not response_text:
                     print(f"Model returned an empty text response: {response}")
                     return False
            except AttributeError: # If response structure is different
                 # Check for a simple .text attribute as a common case
                if hasattr(response, 'text') and response.text:
                    response_text = response.text
                else:
                    print(f"Could not extract text from model response due to unexpected structure: {response}")
                    return False
            except IndexError:
                 print(f"Could not extract text from model response candidates: {response}")
                 return False


            if not validator(response_text): # User-provided validator
                return False

            # Usage metadata might also be structured differently
            # prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 'N/A'
            # candidates_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 'N/A'
            # print(f"✓ Completed LLM call (input: {prompt_tokens} tokens, output: {candidates_tokens} tokens)")
            return True

        # Wrap the entire retryCall sequence with the asyncio.Semaphore
        async with self.limit:
            final_response_obj = await retry_call(
                async_fn=llm_call_attempt,
                validator_fn=response_validator,
                max_retries=MAX_LLM_RETRIES,
                error_message="Failed to get a valid model response.",
                delay_ms=RETRY_DELAY_MS,
            )
        
        # Extract text from the final validated response object
        # This needs to be robust based on Python SDK's actual response structure
        try:
            if hasattr(final_response_obj, 'text') and final_response_obj.text:
                return final_response_obj.text
            # Fallback to the more complex structure if .text is not available/primary
            return final_response_obj.candidates[0].content.parts[0].text
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Failed to extract text from final LLM response: {final_response_obj}. Error: {e}")


# Safety settings mapping from TS HarmCategory/HarmBlockThreshold to Python SDK
# This assumes SDKHarmCategory and SDKHarmBlockThreshold are enum-like objects
# or string constants as defined by the Python SDK.
try:
    safety_settings_map = [
        (SDKHarmCategory.HARM_CATEGORY_HATE_SPEECH, SDKHarmBlockThreshold.BLOCK_NONE),
        (SDKHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, SDKHarmBlockThreshold.BLOCK_NONE),
        (SDKHarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, SDKHarmBlockThreshold.BLOCK_NONE),
        (SDKHarmCategory.HARM_CATEGORY_HARASSMENT, SDKHarmBlockThreshold.BLOCK_NONE),
        # (SDKHarmCategory.HARM_CATEGORY_UNSPECIFIED, SDKHarmBlockThreshold.BLOCK_NONE), # UNSPECIFIED might not be directly used or available
    ]
    # Convert to list of SafetySetting objects if the SDK requires that
    # For now, assume they are passed as tuples or dicts to the model config
    
    # The Python SDK usually takes a list of SafetySetting objects
    # safety_settings_for_sdk = [
    #     SDKSafetySetting(category=cat, threshold=thr) for cat, thr in safety_settings_map
    # ]
    # However, if SDKSafetySetting is not a type but a dict structure:
    safety_settings_for_sdk = [
        {"category": cat, "threshold": thr} for cat, thr in safety_settings_map
    ]


except NameError: # If SDK types are not available
    print("Warning: SDK HarmCategory/HarmBlockThreshold not found. Using placeholder safety settings.")
    safety_settings_for_sdk = []


def get_model_params(model_name: str, schema: Optional[PySchema] = None) -> SDKModelParams:
    """
    Creates a model specification object for Vertex AI generative models.
    Python SDK might use slightly different parameter names or structures.
    """
    # generation_config: SDKGenerationConfig = { # Type from SDK
    generation_config: Dict[str, Any] = { # Using Dict for flexibility
        "temperature": 0.0, # Python SDK typically uses float for temperature
        "top_p": 0.0, # Ensure this is float if SDK expects it
        # "candidate_count": 1, # Often a default or set elsewhere
    }

    if schema:
        generation_config["response_mime_type"] = "application/json"
        # The schema format for Python SDK might need to be an OpenAPISchema object
        # or a dictionary representing it.
        # from google.cloud.aiplatform_v1beta1.types import OpenApiSchema
        # generation_config["response_schema"] = OpenApiSchema(type=..., properties=...)
        generation_config["response_schema"] = schema # Assuming schema is already in correct dict format

    model_params: SDKModelParams = {
        # 'model': model_name, # Model name is often part of SDKGenerativeModel constructor
        'generation_config': generation_config,
        'safety_settings': safety_settings_for_sdk,
    }
    # Note: 'model' (model_name) is usually specified when creating SDKGenerativeModel instance,
    # not necessarily in this params dictionary that's passed to generate_content.
    # However, some SDK versions or helper functions might accept it.
    return model_params

# The 'Request' type in TS was for a specific JSON structure.
# The Python SDK's generate_content method usually takes the prompt string(s)
# directly or a list of Content objects.
# class ContentPart(TypedDict):
# text: str

# class ContentItem(TypedDict):
# role: str
# parts: List[ContentPart]

# class RequestPayload(TypedDict):
# contents: List[ContentItem]

# def get_request(prompt: str) -> RequestPayload:
# """Converts a simple prompt string to the SDK's expected content format."""
# return {
# "contents": [{"role": "user", "parts": [{"text": prompt}]}]
# }
# This get_request function is likely not needed if prompts are passed directly.
# If a complex structure is needed, it should match google.cloud.aiplatform_v1beta1.types.Content
# e.g. from google.cloud.aiplatform_v1beta1.types import Content, Part
# Content(role='user', parts=[Part(text=prompt)])
