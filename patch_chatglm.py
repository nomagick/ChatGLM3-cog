import types
import re

chat_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]

pattern = re.compile(
    r"({})".format("|".join(re.escape(token) for token in chat_special_tokens))
)


def encode_chat(self, query: str, **_kwargs):
    matches = pattern.finditer(query)
    input_ids = []

    pos = 0
    for m in matches:
        input_ids.extend(self.encode(query[pos : m.start()]))
        tok = m.group(1)
        input_ids.append(self.special_tokens[tok])
        pos = m.end()

    input_ids.extend(self.encode(query[pos:]))

    return input_ids


def build_chat_input(self, query, **_kwargs):
    return self.batch_encode_plus(
        [self.tokenizer.encode_chat(query)],
        return_tensors="pt",
        is_split_into_words=True,
    )

import torch
from typing import Optional, List, Callable
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
import copy

@torch.inference_mode()
def stream_generate(
    self,
    input_ids,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    return_past_key_values=False,
    **kwargs,
):
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = self.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    model_kwargs["use_cache"] = generation_config.use_cache
    bos_token_id, eos_token_id = (
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = (
        torch.tensor(eos_token_id).to(input_ids.device)
        if eos_token_id is not None
        else None
    )
    has_default_max_length = (
        kwargs.get("max_length") is None and generation_config.max_length is not None
    )

    # 2. Set generation parameters if not already defined
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = self._get_logits_warper(generation_config)
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        next_token_logits = outputs.logits[:, -1, :]
        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)
        # sample
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)
        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1)
            .ne(eos_token_id_tensor.unsqueeze(1))
            .prod(dim=0)
        )
        if return_past_key_values:
            yield input_ids, outputs.past_key_values
        else:
            yield input_ids
        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break


def process_response(self, output, history):
    return output, history

def patch(tokenizer, model):
    tokenizer.tokenizer.encode_chat = types.MethodType(encode_chat, tokenizer.tokenizer)
    tokenizer.build_chat_input = types.MethodType(build_chat_input, tokenizer)
    model.stream_generate = types.MethodType(stream_generate, model)
    model.process_response = types.MethodType(process_response, model)
