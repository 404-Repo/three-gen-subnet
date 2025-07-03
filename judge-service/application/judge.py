from loguru import logger
from PIL import Image
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


JUDGE_INSTRUCTION = """You are an expert 3d model evaluator. You are given two images each containing a number of views of a 3D model. Both 3D models were AI-generated from the same text prompt: {prompt}. 
Decide which object matches the prompt more closely and is of higher quality or if they match the prompt equally well and have the same quality.

Criteria for prompt matching:
* Only objects that exist in the prompt are generated. No additional objects.
* The generated objects makes sense from a common sense point of view.

Criteria for quality estimation:
* The model has no artifacts.
* The model has good detail.
* The model is orientented properly in the images and NOT rotated upside down or tilted.
* The textures are sharp.
* The lighting is pleasing and not too harsh.

Instructions:
First, provide a comparative analysis of both 3d models in terms of prompt matching.
Second, provide a comparative analysis of both 3d models quality.
Then score both images on a scale of 1-10, with 1 meaning the 3d model is of very low quality and 10 meaning the 3d model is of the highest quality, considering the quality analysis.
Lastly, considering your prior analyses and scoring, decide if one 3d model is superior or both are equally good.
If both images are close or you're unsure which is better, don't be afraid to make it a draw.

Return 1 if the first image wins, 2 for the second and 0 if it's a draw.
Output your result as a JSON object of the following form: {{prompt_matching: <Your prompt matching analysis>, quality: <Your prompt matching analysis>, quality_score1:<score of the first image [1;10]>, quality_score2:<score of the second image [1;10]>, winner: <Your decision [1,2,0]>}}.

Answer just with the pure JSON object and nothing else as your output will be json-parsed."""  # noqa


class JudgeResponse(BaseModel):
    prompt_matching: str
    quality: str
    quality_score1: int
    quality_score2: int
    winner: int


class JudgeModel:
    def __init__(
        self,
        model_name: str = "AIDC-AI/Ovis2-16B",
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8096,
        seed: int = 56,
    ) -> None:
        logger.info(f"Initializing JudgeModel with {model_name}")

        self.model_name = model_name
        self.instruction_prompt = JUDGE_INSTRUCTION
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        try:
            self._judge_model = LLM(
                model=model_name,
                trust_remote_code=False,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                disable_mm_preprocessor_cache=False,
                max_num_seqs=2,
                seed=seed,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._sampling_parameters = self._create_sampling_params()
            logger.info("JudgeModel initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize JudgeModel: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize JudgeModel: {str(e)}") from e

    def _create_sampling_params(self) -> SamplingParams:
        json_schema = JudgeResponse.model_json_schema()
        guided_decoding_params_json = GuidedDecodingParams(json=json_schema)
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop_token_ids=None,
            top_p=self.top_p,
            guided_decoding=guided_decoding_params_json,
        )
        return sampling_params

    def create_chat_template(self, images: list[Image.Image], instruction_prompt: str) -> dict:
        placeholders = "\n".join(f"Image-{i}: <image>\n" for i, _ in enumerate(images, start=1))
        messages = [{"role": "user", "content": f"{placeholders}\n{instruction_prompt}"}]
        prompt_template = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        chat_template = {"prompt": prompt_template, "multi_modal_data": {"image": images}}

        return chat_template

    def judge(self, image1: Image.Image, image2: Image.Image, prompt: str) -> JudgeResponse:
        instruction_prompt = self.instruction_prompt.format(prompt=prompt)
        chat_template = self.create_chat_template([image1, image2], instruction_prompt)

        output = self._judge_model.generate(chat_template, sampling_params=self._sampling_parameters)
        result: JudgeResponse = JudgeResponse.model_validate_json(output[0].outputs[0].text)

        return result
