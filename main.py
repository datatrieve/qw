from vllm import LLM, SamplingParams

MODEL_NAME = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"

llm = LLM(
    model=MODEL_NAME,
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
)


def run(messages: list = None, prompt: str = None, tools: list = None,
        temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 2048,
        **kwargs):
    if messages:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            tools=tools,
        )
        output = outputs[0].outputs[0]
        result = {"role": "assistant", "content": output.text}
        if output.tool_calls:
            result["tool_calls"] = [
                {"id": tc.function.name, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in output.tool_calls
            ]
        return result
    elif prompt:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = llm.generate([prompt], sampling_params)
        return {"response": outputs[0].outputs[0].text}
    return {"error": "Provide 'messages' (chat) or 'prompt' (completion)"}
