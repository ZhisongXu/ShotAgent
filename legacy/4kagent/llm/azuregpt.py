from pathlib import Path
import requests
import logging
from typing import Callable, Optional
from time import sleep
import random
import re
import os
import json

from .base_llm import BaseLLM
from utils.misc import encode_img
from openai import AzureOpenAI


class AzureGPT(BaseLLM):
    """Parameters when called: img_path_lst, prompt, format_check."""

    def __init__(self,
                 config_path: Path = Path("config.yml"),
                 log_path: Optional[Path | str] = None,
                 logger: Optional[logging.Logger] = None,
                 silent: bool = False,
                 system_message: Optional[str] = None,
                 model: Optional[str] = None
                 ):
        super().__init__(
            config_path=config_path,
            log_path=log_path,
            logger=logger,
            silent=silent
        )  # set attributes: cfg, logger, silent

        self.api_key = self.cfg["AZUREGPT"]["API_KEY"]
        self.model = self.cfg["AZUREGPT"]["MODEL"]
        self.max_tokens = self.cfg["AZUREGPT"]["MAX_TOKENS"]
        self.temperature = self.cfg["AZUREGPT"]["TEMPERATURE"]
        self.endpoint = self.cfg["AZUREGPT"]["ENDPOINT"]
        self.api_version = self.cfg["AZUREGPT"]["API_VERSION"]

        self.prompt_tokens = 0
        self.completion_tokens = 0

        self.system_message = system_message
        if self.system_message is not None:
            self._log("_Note: These user-assistant interactions are independent "
                      "and the system message is always attached in each turn for GPT._")
            self._log("**System message for GPT**")
            self._log(self.system_message)
        
        self.endpoint = os.getenv("ENDPOINT_URL", self.endpoint)
        self.deployment = os.getenv("DEPLOYMENT_NAME", self.model)  
        self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY", self.api_key)
        self.client = AzureOpenAI(  
            azure_endpoint=self.endpoint,  
            api_key=self.subscription_key,  
            api_version=self.api_version,
        )

    def query(self,
              img_path_lst: Optional[list[Path]] = None,
              prompt: str = "",
              format_check: Optional[Callable[[object], None]] = None,
              ) -> tuple[str, str]:
        messages = self._prepare_for_request(
            prompt, img_path_lst)
        # print('messages: ', messages)
        while True:
            completion = self.client.chat.completions.create(  
                model=self.deployment,
                messages=messages,
                max_tokens=800,  
                temperature=0.7,  
                top_p=0.95,  
                frequency_penalty=0,  
                presence_penalty=0,
                stop=None,  
                stream=False
            )

            rsp_json = completion.to_json()
            rsp_data = json.loads(rsp_json)
            usage = rsp_data["usage"]
            self.prompt_tokens += usage["prompt_tokens"]
            self.completion_tokens += usage["completion_tokens"]

            rsp_text: str = rsp_data['choices'][0]['message']['content']
            json_pattern = r'({.*?})'

            # Try to find JSON-like structure with or without backticks
            match = re.search(json_pattern, rsp_text, re.DOTALL)

            if match:
                json_str = match.group(1).strip()
                # Parse the extracted JSON string into a Python dictionary
                extracted_dict = json.loads(json_str)
            else:
                json_str = None

            inner_rsp_text = json_str
            valid, inner_rsp_text = self._check_syntax(inner_rsp_text, format_check)
            if format_check is not None:
                valid, inner_rsp_text = self._check_syntax(inner_rsp_text, format_check)
                if not valid:
                    continue
            
            return prompt, inner_rsp_text

    def _prepare_for_request(self, prompt: str,
                             img_path_lst: Optional[list[Path]] = None
                             ) -> tuple[dict, dict]:
        content = [{
            "type": "text",
            "text": prompt
        }]
        if img_path_lst is not None:
            for img_path in img_path_lst:
                img_base64 = encode_img(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_base64,
                        "detail": "auto"
                    }
                })

        messages = []
        if self.system_message is not None:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        messages.append({
            "role": "user",
            "content": content
        })

        return messages

    def _send_request(self, headers: dict, payload: dict,
                      max_retries: int = 5,
                      initial_delay: int = 3,
                      exp_base: int = 2,
                      jitter: bool = True) -> requests.Response:
        """Sends a request to the OpenAI API and handles errors with exponential backoff."""

        n_retries = 0
        backoff_delay = initial_delay
        while True:
            try:
                response = requests.post(self.endpoint,
                                         headers=headers, json=payload)
                is_valid, recommended_delay = self._check_response(response)
                if is_valid:
                    return response
            except Exception as e:
                self._log("An error occurred when sending a request: "
                          f"{type(e).__name__}: {e}",
                          level='warning')
                recommended_delay = None

            n_retries += 1
            if n_retries > max_retries:
                raise RuntimeError(
                    "Too many errors occurred when querying LLM.")
            if recommended_delay is not None:
                delay = recommended_delay
            else:
                backoff_delay *= exp_base * (1 + jitter*random.random())
                delay = backoff_delay
            self._log(
                f"Retrying in {delay:.3f} seconds...", level='warning')
            sleep(delay)

    def _check_response(self, response: requests.Response) -> tuple[bool, Optional[float]]:
        """Checks if the response is valid. If error occurs, gets the recommended delay if any.

        Args:
            response (requests.Response): Response from the OpenAI API to check.

        Returns:
            is_valid (bool): Whether the response is valid.
            recommended_delay (float | None): The delay recommended by the API if any.
        """

        if "error" in response.json():
            err_msg: str = response.json()['error']['message']
            self._log(f"An error occurred when querying LLM: {err_msg}",
                      level='warning')

            recommended_delay = None
            if response.json()['error']['code'] == 'rate_limit_exceeded':
                # there may exist "Please try again in xxs/xxmxxs/xxms." in the error message
                match = re.search(
                    R"(?<=Please try again in )(\d+m)?\d+\.?\d*(?=s)", err_msg)
                if match is not None:
                    t = match.group().split('m')
                    m = t[0] if len(t) > 1 else 0
                    s = t[-1]
                    recommended_delay = 60*int(m) + float(s)

            return False, recommended_delay

        if (finish_reason := response.json()['choices'][0]['finish_reason']) != 'stop':
            self._log(f"finish_reason if {finish_reason}", level='warning')

        return True, None

    def _check_syntax(self, rsp_text: str, format_check: Callable[[object], None]
                      ) -> tuple[bool, str]:
        """Checks whether the response is a valid Python object and follows the specified format. 
        If valid, returns the processed response (the valid response may be wrapped in something)."""
        # Check if the response is a valid Python object
        try:
            obj = eval(rsp_text)
        except:
            # GPT may wrap the response in a code block
            inner_rsp_text = rsp_text.strip("```").lstrip("json").strip()
            print('inner_rsp_text: ', inner_rsp_text)
            try:
                obj = eval(inner_rsp_text)
                rsp_text = inner_rsp_text
            except:
                self._log("Failed to parse the response:", level='warning')
                self._log(rsp_text, level='warning')
                return False, ""
        # Check if the response follows the specified format
        try:
            format_check(obj)
        except AssertionError as e:
            self._log(f"Failed to pass the format check: {e}", level='warning')
            self._log(f"Response: {obj}", level='warning')
            return False, ""
        return True, rsp_text
    
    def _post_process(self):
        """Logs the token usage and cost."""        
        self._log("Token usage so far: "
                  f"{self.prompt_tokens} prompt tokens, "
                  f"{self.completion_tokens} completion tokens")
        total_cost = self.prompt_tokens/1000*0.01 + self.completion_tokens/1000*0.03
        self._log(f"Cost so far: ${total_cost:.5f}")
