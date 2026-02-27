import openai
import requests
from abc import ABC
from typing import Union,Dict,Optional,Any,Literal,List,Type
from pydantic import BaseModel,Field

class LLM(BaseModel, ABC):
    """
    Generic large language model interface. Extend this class to implement specific LLM providers.
    """

    llm_class:Literal['base'] = "base"

    model: str = Field(default=None, description="Name of the LLM to use.")
    sampling_options: Optional[Dict[str, Any]] = Field(
        default={
            "num_predict": 1024,
            "num_ctx": 8192,
            "temperature": 1.0,
            "min_p": 0.1,
            "keep_alive": "15m"
        },
        description="Dictionary of OpenAI-compatible sampling parameters to use."
    )

    class Config:
        arbitrary_types_allowed = True

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

    def generate_instruct(self, messages, respond=True, response_role=None, stream=True):
        """
        Generate a response to a given text prompt. If stream is true, function returns a generator
        that yields the response chunks as they become available. Otherwise, the full response is
        returned as a string.

        Args:
        messages (list[dict]): The chat messages that the LLM should respond to
        respond (bool): If true, LLM will respond to last message. If false, LLM will
            continue generating from the end of the last message.
        response_role (str): The role LLM should use when responding.
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
    # client field is only populated at runtime
    client: Optional[Any] = Field(default=None, exclude=True)

    def model_post_init(self, context:Any) -> None:
        """
        Called to set up the OpenAI client object once the object is initialized.
        """
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=1200,
            max_retries=10
        )

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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
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

    def count_tokens(self, text:str):
        """
        Count the number of tokens in a given string using the /tokenize upstream endpoint, if available on this server.
        This only really works with llama-swap.

        Args:
        text (str): The input string to tokenize.

        Returns:
        int: The number of tokens in the input string.
        """
        headers = {'Content-Type': 'text/plain'}

        # Create the request payload
        payload = {'content': text}
        # get the upstream URL, llama-swap doesn't support directly
        llm_url = self.client.base_url
        tk_url = llm_url.scheme + "://" + llm_url.netloc.decode() + "/upstream/" + self.model + "/tokenize"
        # Send the POST request
        response = requests.post(tk_url, headers=headers, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Extract the number of tokens from the response
            tokens = data['tokens']

            return len(tokens)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

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
        model="gemma-3-4B-it-UD-Q4_K_XL-cpu",
        sampling_options=samp_params,
    )
    # test converting to JSON
    llm_txt = llm.model_dump_json(indent=2)
    print(llm_txt)

    # test converting back
    rehydrated_llm = OpenAILLM.model_validate_json(llm_txt)
    
    # print("Generating in instruct mode...")
    # test_messages = [
    #     {
    #         "role": "user",
    #         "content": "I'm a cat! What are you?"
    #     }
    # ]
    # response = llm.generate_instruct(
    #     messages=test_messages,
    #     respond=True,
    #     response_role="assistant",
    #     stream=False
    # )
    # print(response)
    # for chunk in response:
    #     print(chunk)
       
    # print("Generating in raw mode...") 
    # response = llm.generate(
    #     prompt="I'm a cat, what are you?",
    #     stream=False
    # )
    # print(response)
    # for chunk in response:
    #     print(chunk)
    
    # # Streaming output

    # print("Streaming in instruct mode...")
    # test_messages = [
    #     {
    #         "role": "user",
    #         "content": "I'm a cat! What are you?"
    #     }
    # ]
    # response = llm.generate_instruct(
    #     messages=test_messages,
    #     respond=True,
    #     response_role="assistant",
    #     stream=True
    # )
    # print(response)
    # for chunk in response:
    #     print(chunk)
       
    # print("Streaming in raw mode...") 
    # response = llm.generate(
    #     prompt="I'm a cat, what are you?",
    #     stream=True
    # )
    # print(response)
    # for chunk in response:
    #     print(chunk)
    
    