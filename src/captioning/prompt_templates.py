import outlines

@outlines.prompt
def construct_caption_prompt(question: str):
    """
    You are an expert dense image captioning system.
    
    Analyze the image carefully and generate a rich, descriptive caption that supports answering the given question, without directly answering it.

    Your caption should be clear, concise, and informative. It may include (but is not limited to):
    - Identification of key objects, people, or actions in the image.
    - Description of the setting and overall visual scene.
    - Culturally or geographically relevant cues.
    - Visual elements that may help explain the connection to the question.

    Avoid including opinions or answering the question yourself. Your role is only to describe the visual content as fully and accurately as possible.

    ### QUESTION
    {{ question }}
    
    ### RESPONSE
    """

@outlines.prompt
def construct_golden_caption_prompt(question: str, country: str, golden_answer: str):
    """
    You are an expert dense image captioning system.

    Given a question, the country the image is associated with, and the correct answer, generate a comprehensive, informative caption that helps a model arrive at the correct answer.

    Your caption should be:
    - Descriptive, factual, and visually grounded.
    - Explicitly include cultural, geographic, or contextual cues that support the answer.
    - Designed to connect the image content meaningfully to the question and answer.

    Avoid simply restating the answer, instead, embed visual evidence and context that leads to the correct answer.

    ### QUESTION
    {{ question }}

    ### COUNTRY
    {{ country }}

    ### CORRECT ANSWER
    {{ golden_answer }}

    ### RESPONSE
    """