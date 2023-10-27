# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, ConcatenateIterator
from transformers import AutoModel, AutoTokenizer

import patch_chatglm

MODEL_PATH = "./chatglm3-6b-32k"

from transformers.utils import logging

logging.set_verbosity_error()


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )

        model = AutoModel.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        ).cuda()
        patch_chatglm.patch(self.tokenizer, model)
        self.model = model.eval()

    def predict(
        self,
        prompt: str = Input(
            description="Prompt for completion",
            default='<|system|>\nYou are a helpful assistant\n<|user|>\n请使用英文重复这段话："为了使模型生成最优输出，当使用 ChatGLM3-6B 时需要使用特定的输入格式，请按照示例格式组织输入。"\n<|assistant|>\n',
        ),
        max_tokens: int = Input(
            description="Max new tokens to generate", default=2048, ge=1, le=32768
        ),
        temperature: float = Input(description="Temperature", default=0.75, ge=0, le=5),
        top_p: float = Input(description="Top_p", default=0.8, ge=0, le=1),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""

        do_sample = True
        if temperature <= 0:
            do_simple = False
            temperature = 0.01

        offset = 0

        for response, _ in self.model.stream_chat(
            self.tokenizer,
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            max_length=32768,
        ):
            chunk = response[offset:]
            offset = len(response)
            if chunk:
                yield chunk
