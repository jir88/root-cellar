import httpx
import openai
import asyncio
from abc import ABC
from typing import Union,Dict,Optional,Any,Literal,List,Type
from pydantic import BaseModel,Field,ConfigDict

class LLM(BaseModel, ABC):
    """
    Generic large language model interface. Extend this class to implement specific LLM providers.
    """

    llm_class:Literal['base'] = "base"

    model: str = Field(default=None, description="Name of the LLM to use.")
    sampling_options: Optional[Dict[str, Any]] = Field(
        default={
            "num_predict": 1024,
            "temperature": 1.0,
            "min_p": 0.1,
            "keep_alive": "15m"
        },
        description="Dictionary of OpenAI-compatible sampling parameters to use."
    )
    context_window: int = Field(
        default=8192,
        description="Context window size for the underlying LLM."
    )

    model_config = ConfigDict(arbitrary_types_allowed = True)

    def generate(self, prompt, stream=True):
        """
        Generate a response to a given text prompt. If stream is true, function returns a generator
        that yields the response chunks as they become available. Otherwise, the full response is
        returned as a string.

        Args:
        prompt (str): The prompt that the LLM should respond to
        stream (bool): Whether the response should be streamed as it is generated

        Returns:
        A generator function if stream is true, otherwise a string containing the response.
        """
        raise NotImplementedError("Method must be implemented in a subclass!")

    def generate_instruct(self, messages, stream=True):
        """
        Generate a response to a given text prompt. If stream is true, function returns a generator
        that yields the response chunks as they become available. Otherwise, the full response is
        returned as a string.

        Args:
            messages (list[dict]): The chat messages that the LLM should respond to
            stream (bool): Whether the response should be streamed as it is generated

        Returns:
            A generator function if stream is true, otherwise a string containing the response.
        """
        raise NotImplementedError("Method must be implemented in a subclass!")
    
    def generate_structured(self, messages:List[Dict[str,str]], response_model:Type[BaseModel]):
        """
        Respond to a prompt using structured JSON mode.

        Args:
            messages (List[Dict[str,str]]): The list of prior messages.
            response_model (BaseModel): A pydantic type defining the JSON schema the LLM must use to respond.

        Returns:
            The response as a pydantic object.
        """
        raise NotImplementedError("Method must be implemented in a subclass!")
    
    def count_tokens(self, text:str):
        """
        Count (or estimate) the number of tokens in a given string.

        Args:
        text (str): The input string to tokenize.

        Returns:
        int: The (approximate) number of tokens in the input string.
        """
        raise NotImplementedError("Method must be implemented in a subclass!")

class OpenAILLM(LLM):
    """
    Interact with any OpenAI compatible backend.
    """
    # type name for deserialization
    llm_class:Literal['OpenAILLM'] = "OpenAILLM"

    api_key: str = Field(
        default="sk_fake",
        description="The API key to use; can use an arbitrary string for local endpoints that do not require a key."
    )
    base_url: str = Field(
        default="http://127.0.0.1:8080/v1",
        description="The URL of the API endpoint."
    )
    server_type: Literal['unknown', 'llama-server', 'llama-swap', 'ollama', 'openai'] = Field(
        default="unknown",
        description="Type of server this instance is connected to. This determines what extra endpoints are available."
    )
    upstream_type: Literal['unknown', 'none', 'llama-server', 'ollama', 'openai'] = Field(
        default="unknown",
        description="If this instance is connected to a proxy like llama-swap, defines the type of upstream server."
    )

    # client field is only populated at runtime
    client: Optional[Any] = Field(default=None, exclude=True)

    def model_post_init(self, context:Any) -> None:
        """
        Called to set up the OpenAI client object once the object is initialized.
        """
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=1200,
            max_retries=10
        )
    
    async def check_server_type(self, timeout:int=30) -> str:
        """
        Determine what type of server this instance is connected to.

        Args:
            timeout (int): Timeout length in seconds for GET requests.

        Returns:
            A string describing the server type. For llama-swap servers, \
            query self.upstream_type to determine what upstream server is \
            behind the llama-swap proxy.
        """
        # get the server location
        llm_url = self.client.base_url
        url_host = llm_url.scheme + "://" + llm_url.netloc.decode()

        # check for ollama
        if self._check_endpoint(url=url_host + "/api/tags"):
            self.server_type = "ollama"
            self.upstream_type = "none"
            return self.server_type

        # check for llama-server
        if self._check_endpoint(url=url_host + "/lora-adapters"):
            self.server_type = "llama-server"
            self.upstream_type = "none"
            return self.server_type

        # check for llama-swap
        if self._check_endpoint(url=url_host + "/running"):
            self.server_type = "llama-swap"
            # check backend for llama-server
            llm_url = self.client.base_url
            upstream_url = llm_url.scheme + "://" + llm_url.netloc.decode() + "/upstream/" + self.model
            
            # check for ollama
            if self._check_endpoint(url=upstream_url + "/api/tags"):
                self.upstream_type = "ollama"
                return self.server_type

            # check for llama-server
            if self._check_endpoint(url=upstream_url + "/lora-adapters"):
                self.upstream_type = "llama-server"
                return self.server_type
            # use /models to check for OpenAI-compliant
            if self._check_endpoint(url=upstream_url + "/models"):
                self.upstream_type = "openai"
                return self.server_type
            # don't know backend
            print("Unknown backend")
            self.upstream_type = "unknown"
            return self.server_type
        
        # use /models to check for OpenAI-compliant
        if self._check_endpoint(url=url_host + "/models"):
            self.server_type = "openai"
            self.upstream_type = "none"
            return self.server_type
        
        # don't know server
        print("Unknown server")
        self.server_type = "unknown"
        self.upstream_type = "unknown"
        return self.server_type

    def _check_endpoint(self, url, timeout:int=30):
        """
        Check to see if an endpoint exists.
        Args:
        url (str|URL): The URL to check with a GET request.
        timeout (int): Timeout length in seconds.

        Returns:
        True if GET request returns successfully, else false.
        """
        response = httpx.get(url=url, timeout=timeout)
        return response.status_code == 200

    def generate(self, prompt, stream=True):
        """
        Generate a response to a given text prompt. If stream is true, function returns a generator
        that yields the response chunks as they become available. Otherwise, the full response is
        returned as a string.

        Args:
        prompt (str): The prompt that the LLM should respond to
        stream (bool): Whether the response should be streamed as it is generated

        Returns:
        A generator function if stream is true, otherwise a string containing the response.
        """
        # ollama generates dicts with keys 'response' (the text), eval_count, eval_duration (tokens generated and time it took in ms)
        # prompt_eval_count (how much prompt was sent and processed)
        # OpenAI format puts the text in response['choices'][0]['message']['content']
        response = self.client.completions.create(
            model=self.model, 
            prompt=prompt, 
            stream=stream,
            # shove all sampling parameters through this mechanism to avoid manually
            # specifying the canonical OpenAI ones
            extra_body=self.sampling_options
        )
        
        if not stream:
            ol_dict = {
                'response': response.choices[0].text
            }
            # add generation speed if available
            if response.usage is not None:
                ol_dict['prompt_eval_count'] = response.timings['prompt_n']
                ol_dict['eval_count'] = response.timings['predicted_n']
                # ollama outputs times in nanoseconds for some reason...
                ol_dict['eval_duration'] = response.timings['predicted_ms']*1.0e6
            yield ol_dict
        else:
            for chunk in response:
                ol_dict = {
                    'response': chunk.choices[0].text
                }
                # add generation speed if available
                if chunk.usage is not None:
                    ol_dict['prompt_eval_count'] = chunk.timings['prompt_n']
                    ol_dict['eval_count'] = chunk.timings['predicted_n']
                    # ollama outputs times in nanoseconds for some reason...
                    ol_dict['eval_duration'] = chunk.timings['predicted_ms']*1.0e6
                yield ol_dict

    async def generate_instruct(self, messages, stream=True):
        """
        Generate a response to a given text prompt. If stream is true, function returns a generator
        that yields the response chunks as they become available. Otherwise, the full response is
        returned as a string.

        Args:
            messages (list[dict]): The chat messages that the LLM should respond to
            stream (bool): Whether the response should be streamed as it is generated

        Returns:
            A generator function if stream is true, otherwise a string containing the response.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            # shove all sampling parameters through this mechanism to avoid manually
            # specifying the canonical OpenAI ones
            extra_body=self.sampling_options
        )
        
        if not stream:
            ol_dict = {
                'response': response.choices[0].message.content
            }
            # add generation speed if available
            if response.choices[0].finish_reason == 'stop':
                ol_dict['prompt_n'] = response.timings['prompt_n']
                ol_dict['prompt_per_second'] = response.timings['prompt_per_second']
                ol_dict['cache_n'] = response.timings['cache_n']
                ol_dict['predicted_n'] = response.timings['predicted_n']
                ol_dict['predicted_per_second'] = response.timings['predicted_per_second']
            yield ol_dict
        else:
            try:
                async for chunk in response:
                    ol_dict = {
                        'response': chunk.choices[0].delta.content
                    }
                    # add generation speed if available
                    if chunk.choices[0].finish_reason == 'stop':
                        ol_dict['prompt_n'] = chunk.timings['prompt_n']
                        ol_dict['prompt_per_second'] = chunk.timings['prompt_per_second']
                        ol_dict['cache_n'] = chunk.timings['cache_n']
                        ol_dict['predicted_n'] = chunk.timings['predicted_n']
                        ol_dict['predicted_per_second'] = chunk.timings['predicted_per_second']
                    yield ol_dict
            finally:
                # stop the LLM generating
                await response.close()

    def generate_structured(self, messages:List[Dict[str,str]], response_model:Type[BaseModel]):
        """
        Respond to a prompt using structured JSON mode.

        Args:
            messages (List[Dict[str,str]]): The list of prior messages.
            response_model (BaseModel): A pydantic type defining the JSON schema the LLM must use to respond.

        Returns:
            The response as a pydantic object.
        """
        response = self.client.chat.completions.create(
            model=self.model, 
            messages=messages, 
            stream=False,
            response_format={"type": "json_schema", "json_schema": response_model.model_json_schema()},
            # try shoving all sampling parameters through this mechanism to avoid manually
            # specifying the canonical OpenAI ones
            extra_body=self.sampling_options
        )
        # validate the response
        parsed_response = response_model.model_validate_json(response.choices[0].message.content)
        return parsed_response

    async def count_tokens(self, text:str) -> int:
        """
        Count the number of tokens in a given string using the /tokenize upstream endpoint, if available on this server.
        This only really works with llama-swap and llama.cpp. Otherwise, tokens are estimated based on 3.5 chars per token.

        Args:
        text (str): The input string to tokenize.

        Returns:
        int: The number of tokens in the input string.
        """
        # check server type if needed
        if self.server_type == "unknown" or self.upstream_type == "unknown":
            await self.check_server_type(timeout=30)
        
        # llama.cpp llama-server
        if self.server_type == "llama-server":
            headers = {'Content-Type': 'text/plain'}
            # Create the request payload
            payload = {'content': text}
            llm_url = self.client.base_url
            tk_url = llm_url.scheme + "://" + llm_url.netloc.decode() + "/tokenize"
            # Send the POST request
            async with httpx.AsyncClient() as client:
                # The 'await' keyword allows the event loop to run other tasks 
                # while waiting for the network response.
                response = await client.post(tk_url, headers=headers, json=payload, timeout=60)
                # error out if request failed
                response.raise_for_status()
            # Parse the JSON response
            data = response.json()
            # Extract the number of tokens from the response
            tokens = data['tokens']
            # return token count
            return len(tokens)

        if self.server_type == "llama-swap" and self.upstream_type == "llama-server":
            headers = {'Content-Type': 'text/plain'}
            # Create the request payload
            payload = {'content': text}
            # get the upstream URL, llama-swap doesn't support directly
            llm_url = self.client.base_url
            tk_url = llm_url.scheme + "://" + llm_url.netloc.decode() + "/upstream/" + self.model + "/tokenize"
            # Send the POST request
            async with httpx.AsyncClient() as client:
                # The 'await' keyword allows the event loop to run other tasks 
                # while waiting for the network response.
                response = await client.post(tk_url, headers=headers, json=payload, timeout=30)
                # raise errors for calling code to deal with as needed
                response.raise_for_status()

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                # Extract the number of tokens from the response
                tokens = data['tokens']
                return len(tokens)
            else:
                print(f"Error counting tokens: {response.status_code} - {response.text}")
                print("Estimating token count instead...")
                return round(max(1, len(text)/3.5))
        
        # otherwise, we'll just cop out and guess
        print(f"Server type {self.server_type} with backend {self.upstream_type} does not support tokenization. Estimating token count.")
        return round(max(1, len(text)/3.5))

# a union type covering the possible LLM types
# you can discriminate it by using Field(discriminator='llm_class')
LLMType = Union[LLM, OpenAILLM]

if __name__ == "__main__":
    # test OpenAILLM
    samp_params = {
        "temperature": 1.6,
        "min_p": 0.01,
        "max_tokens": 12
    }
    llm = OpenAILLM(
        model="gemma-4-26B-A4B-it-UD-Q3_K_M-cpu",
        sampling_options=samp_params,
    )
    # test converting to JSON
    llm_txt = llm.model_dump_json(indent=2)
    print(llm_txt)

    # test converting back
    rehydrated_llm = OpenAILLM.model_validate_json(llm_txt)

    # test tokenizing
    token_count = llm.count_tokens("Hello my name is bob.")
    print("Token count: " + str(token_count))

    print("Checking model capabilities...")
    llm_url = llm.client.base_url
    url_host = llm_url.scheme + "://" + llm_url.netloc.decode()
    # check for llama-swap
    if llm._check_endpoint(url=url_host + "/health"):
        llm.server_type = "llama-swap"
        print("Model running on llama-swap!")
    
    async def test_check_server_type():
        await llm.check_server_type()
        print("Server type: " + llm.server_type)
        print("Upstream type: " + llm.upstream_type)
        count = await llm.count_tokens("Hi I am bill!")
        print("Tokens: " + str(count))
    
    asyncio.run(test_check_server_type())
