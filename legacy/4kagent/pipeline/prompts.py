system_message = """You are an expert in image restoration. Given an image of low quality, your task is guiding the user to utilize various tools to enhance its quality. The input image may suffer from various kinds of degradations, including low resolution, noise, motion blur, defocus blur, haze, rain, dark, and jpeg compression artifact. The available tools each specialize in addressing one of the above eight degradations, i.e., super-resolution, denoising, motion deblurring, defocus deblurring, dehazing, deraining, brightening, and jpeg compression artifact removal. The following will be a continuation of an interaction between you and a user to restore an image. Note that if the user specifies the output format, you must strictly follow it without any other words."""


gpt_evaluate_degradation_prompt = """Here's an image to restore. Please assess it with respect to the following seven degradations: noise, motion blur, defocus blur, haze, rain, dark, and jpeg compression artifact. For each degradation, please explicitly give your thought and the severity. Be as precise and concise as possible. Your output must be in the format of a list of JSON objects, each having three fields: "degradation", "thought", and "severity". "degradation" must be one of ["noise", "motion blur", "defocus blur", "haze", "rain", "dark", "jpeg compression artifact"]; "thought" is your thought on this degradation of the image; "severity" must be one of "very low", "low", "medium", "high", "very high". Here's a simple example of the format:
[
    {
        "degradation": "noise",
        "thought": "The image does not appear to be noisy.",
        "severity": "low"
    },
    {
        "degradation": "motion blur",
        "thought": "The image is blurry in the vertical direction, which is likely caused by motion of the camera.",
        "severity": "high"
    },
    {
        "degradation": "defocus blur",
        "thought": "The image does not seem to be out of focus.",
        "severity": "low"
    },
    {
        "degradation": "haze",
        "thought": "There is somewhat haze in the image.",
        "severity": "medium"
    },
    {
        "degradation": "rain",
        "thought": "There is no visible rain in the image.",
        "severity": "very low"
    },
    {
        "degradation": "dark",
        "thought": "The lighting in the image is bright.",
        "severity": "very low"
    },
    {
        "degradation": "jpeg compression artifact",
        "thought": "Blocking artifacts, ringing artifacts, and color bleeding are visible in the image, indicating jpeg compression artifact.",
        "severity": "very high"
    },
]"""


enhanced_gpt_evaluate_degradation_prompt = """You are an expert in image quality assessment (IQA) and well-versed in popular IQA metrics, including Q-align, CLIPIQA+, TOPIQ_NR, MUSIQ, and NIQE. Note that for NIQE, a lower score indicates better image quality, whereas for the other metrics, higher scores generally reflect better quality. Here's an image to restore, along with its corresponding quality scores evaluated using the aforementioned IQA metrics (provided at the end). Please assess the image based on both the metric scores and your prior visual knowledge, with respect to the following seven degradations: noise, motion blur, defocus blur, haze, rain, dark, and jpeg compression artifact. For each degradation, please explicitly give your thought and the severity. Be as precise and concise as possible. Your output must be in the format of a list of JSON objects, each having three fields: "degradation", "thought", and "severity". "degradation" must be one of ["noise", "motion blur", "defocus blur", "haze", "rain", "dark", "jpeg compression artifact"]; "thought" is your thought on this degradation of the image; "severity" must be one of "very low", "low", "medium", "high", "very high". Here's a simple example of the format:
[
    {
        "degradation": "noise",
        "thought": "The image does not appear to be noisy.",
        "severity": "low"
    },
    {
        "degradation": "motion blur",
        "thought": "The image is blurry in the vertical direction, which is likely caused by motion of the camera.",
        "severity": "high"
    },
    {
        "degradation": "defocus blur",
        "thought": "The image does not seem to be out of focus.",
        "severity": "low"
    },
    {
        "degradation": "haze",
        "thought": "There is somewhat haze in the image.",
        "severity": "medium"
    },
    {
        "degradation": "rain",
        "thought": "There is no visible rain in the image.",
        "severity": "very low"
    },
    {
        "degradation": "dark",
        "thought": "The lighting in the image is bright.",
        "severity": "very low"
    },
    {
        "degradation": "jpeg compression artifact",
        "thought": "Blocking artifacts, ringing artifacts, and color bleeding are visible in the image, indicating jpeg compression artifact.",
        "severity": "very high"
    },
]"""


depictqa_evaluate_degradation_prompt = """What's the severity of {degradation} in this image? Answer the question using a single word or phrase in the followings: very low, low, medium, high, very high."""


distill_knowledge_prompt = """We are studying image restoration with multiple degradations. The degradation types we are focusing on are: low-resolution, noise, motion blur, defocus blur, rain, haze, dark, and jpeg compression artifact. We have tools to address these degradations, that is, we can conduct these tasks: super-resolution, denoising, motion deblurring, defocus deblurring, deraining, dehazing, brightening, and jpeg compression artifact removal. The problem is, given the tasks to conduct, we need to determine the order of them. This is very complicated because different tasks may have special requirements and side-effects, and the correct order of tasks can significantly affect the final result. We have conducted some trials and collected the following experience:
{experience}
Please distill knowledge from this experience that will be valuable for determining the order of tasks. Note that the degradations can be more complex than what we have encountered above."""


reason_to_schedule_prompt = """There's an image suffering from degradations {degradations}. We will invoke dedicated tools to address these degradations, i.e., we will conduct these tasks: {agenda}. Please provide some insights into the correct order of these unordered tasks. You should pay special attention to the essence and side-effects of these tasks."""


schedule_w_retrieval_prompt = """There's an image suffering from degradations {degradations}. We will invoke dedicated tools to address these degradations, i.e., we will conduct these tasks: {agenda}. Now we need to determine the order of these unordered tasks. For your information, based on past trials, we have the following experience:
{experience}
Based on this experience, please give the correct order of the tasks. Your output must be a JSON object with two fields: "thought" and "order", where "order" must be a permutation of {agenda} in the order you determine."""


schedule_wo_retrieval_prompt = """There's an image suffering from {degradations}. We will invoke dedicated tools to address these degradations, i.e., we will conduct these tasks: {agenda}. To determine the order of them, we should consider that: 
{insights} 
Based on these insights, please give the correct order of the tasks. Your output must be a list of the tasks in the order you determine, which is a permutation of {agenda}."""


reschedule_ps_prompt = "\nBesides, in attempts just now, we found the result is unsatisfactory if {failed_tries} is conducted first. Remember not to arrange {failed_tries} in the first place."


gpt_evaluate_tool_result_prompt = """What's the severity of {degradation} in this image? Please provide your reasoning. Your output must be a JSON object with two fields: "thought" and "severity", where "severity" must be one of "very low", "low", "medium", "high", "very high"."""


gpt_compare_prompt = """Which of the two images do you consider to be of better quality? Please provide your reasoning. Your output must be a JSON object with two fields: "thought" and "choice", where "choice" is either "former" or "latter", indicating which image you think is of higher quality. An exception is that if you think the difference is negligible, you can choose "neither" as "choice"."""


depictqa_compare_prompt = "Which of the two images, Image A or Image B, do you consider to be of better quality? Answer the question using a single word or phrase."


updated_perception_system_message = """You are an expert tasked with image quality assessment (IQA) and well-versed in popular IQA metrics, 
including CLIPIQA+, TOPIQ_NR, MUSIQ, and NIQE. Note that for NIQE, a lower score indicates better image quality, 
whereas for the other metrics, higher scores generally reflect better quality. Here's an image to restore, 
along with its corresponding quality scores evaluated using the aforementioned IQA metrics.

Your first step is to assess the image based on both the metric scores and your prior visual knowledge, 
with respect to the following seven degradations: noise, motion blur, defocus blur, haze, rain, dark, jpeg compression artifact. 
Images may suffer from one or more of these degradations.

Here is the map between degradation and restoration task:
("noise": "denoising", "motion blur": "motion deblurring", "defocus blur": "defocus deblurring", "haze": "dehazing", "rain": "deraining", "dark": "brightening", "jpeg compression artifact": "jpeg compression artifact removal").
Your second step is to list the restoration tasks correspond to the degradations in the image.

Your third goal is to describe the content and style of the input image, the description must not contain its image quality.

The final output should be formatted as a JSON object containing image assessment result, restoration plan, image content/style description. 
The keys in the JSON object should be: `degradations`, `tasks`, `image_description`."""


updated_perception_system_prompt = """Information about the input image : IQA metrics: {iqa_result}."""


updated_plan_system_message = """You are an expert in image restoration. Given an image of low quality, your task is to guide the user to utilize various tools to enhance its quality. 
The input image requires a list of restoration tasks. Your goal is to make a plan (the order of the tasks) based on the task list.
The final output should be formatted as a JSON object containing the restoration plan (the correct order of the tasks). The key in the JSON object should be: `plan`."""


updated_plan_system_prompt = """Information about the input image: its description is: {image_description}. It suffer from degradations {degradations}, the list of restoration tasks: {tasks}. For your information, based on past trials, we have the following experience in making a restoration plan:
{experience}
Based on this experience, please give the correct order of the tasks in the restoration plan. The restoration plan must be a permutation of {tasks} in the order you determine."""


schedule_updated_w_retrieval_prompt = """"There's an image, its description is: {image_description}. This image is suffering from degradations {degradations}. We will invoke dedicated tools to address these degradations, i.e., we will conduct these tasks: {agenda}. Now we need to determine the order of these unordered tasks. For your information, based on past trials, we have the following experience:
{experience}
Based on this experience, please give the correct order of the tasks. Your output must be a JSON object with two fields: "thought" and "order", where "order" must be a permutation of {agenda} in the order you determine."""


llama_vision_agent_perception_system_message = """You are an expert tasked with image quality assessment (IQA) and well-versed in popular IQA metrics, 
including CLIPIQA+, TOPIQ_NR, MUSIQ, and NIQE. Note that for NIQE, a lower score indicates better image quality, 
whereas for the other metrics, higher scores generally reflect better quality. Here's an image to restore, along with its corresponding quality scores evaluated using the aforementioned IQA metrics.
 
First, please describe the content and style of the input image, the description must not contain its image quality.
         
Second, please assess the image based on both the metric scores and your prior visual knowledge, 
with respect to the following seven degradations: noise, motion blur, defocus blur, haze, rain, dark, jpeg compression artifact. 
Images may suffer from one or more of these degradations.

**Do not output any explanations or comments.** **Strictly return only a JSON object** containing degradation types and image content/style description. 
The keys in the JSON object should be: `degradations` and `image_description`. 
Information about the input image : IQA metrics: {iqa_result}."""


llama_vision_agent_plan_system_message = """You are an expert in image restoration. Given an image of low quality, your task is to guide the user to utilize various tools to enhance its quality. 
The input image requires a list of restoration tasks. Your goal is to make a plan (the order of the tasks) based on the task list.
The final output should be formatted as a JSON object containing the restoration plan (the correct order of the tasks). The key in the JSON object should be: `plan`.
Information about the input image: its description is: {image_description}. It suffer from degradations {degradations}, the list of restoration tasks: {tasks}. For your information, based on past trials, we have the following experience in making a restoration plan:
{experience}
Based on this experience, please give the correct order of the tasks in the restoration plan. The restoration plan must be a permutation of {tasks} in the order you determine.
**Do not output any explanations or comments.**  **Strictly return only a JSON object** containing plan. 
The keys in the JSON object should be: `plan`."""


llama_vision_agent_perception_no_brighten_system_message = """You are an expert tasked with image quality assessment (IQA) and well-versed in popular IQA metrics, 
including CLIPIQA+, TOPIQ_NR, MUSIQ, and NIQE. Note that for NIQE, a lower score indicates better image quality, 
whereas for the other metrics, higher scores generally reflect better quality. Here's an image to restore, along with its corresponding quality scores evaluated using the aforementioned IQA metrics.
 
First, please describe the content and style of the input image, the description must not contain its image quality.
         
Second, please assess the image based on both the metric scores and your prior visual knowledge, 
with respect to the following two degradations: noise, motion blur, defocus blur, haze, rain, jpeg compression artifact. 
Images may suffer from one or more of these degradations.

**Do not output any explanations or comments.** **Strictly return only a JSON object** containing degradation types and image content/style description. 
The keys in the JSON object should be: `degradations` and `image_description`. 
Information about the input image : IQA metrics: {iqa_result}."""
