import os

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(CUR_DIR))))
DATA_DIR = os.path.join(ROOT_DIR, "data")

CVQA_PROMPT_PATH = os.path.join(DATA_DIR, "cvqa_annotations", "cvqa_prompts.json")

OPENAI_RETRIES = 3
DEBUG_COUNT = 10
RANDOM_SEED = 42
INFINITE_CHUNK_SIZE = 1000000

VISION_MODEL_LIST = [
    # Proprietary models
    "gpt-4o-mini",
    "gemini-2.0-flash",
    
    # Open source >14B
    "meta-llama/Llama-3.2-90B-Vision",
    "meta-llama/Llama-4-Scout-17B-16E",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    
    # Open source <14B
    "meta-llama/Llama-3.2-11B-Vision",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "microsoft/Phi-4-multimodal-instruct",
    "neulab/Pangea-7B-hf",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it"
]

VISION_DATASET_DICT_SIZE = {
    "worldcuisines": 60000,
    "cvqa": 10374,
    "worldcuisines_mcq": 60000,
    "cvqa_mcq": 10374,
}

RAG_MODE_LIST = ['default', 'golden']

LANG_EMBED_INSTRUCTION_QUESTION = {
    'en': "Read the following question and retrieve some relevant text to help answer the question\n\nQuestion: {question}",
    'te': "కింది ప్రశ్నను చదివి, ప్రశ్నకు సమాధానం ఇవ్వడానికి సహాయపడే కొంత సంబంధిత వచనాన్ని తిరిగి పొందండి.\n\nప్రశ్న: {question}",
    'th': "อ่านคำถามต่อไปนี้และค้นหาข้อความที่เกี่ยวข้องเพื่อช่วยตอบคำถาม\n\nคำถาม: {question}",
    'tl': "Basahin ang sumusunod na tanong at kunin ang ilang nauugnay na teksto para makatulong sa pagsagot sa tanong.\n\nTanong: {question}",
    'sw': "Soma swali lifuatalo na upate maandishi yanayofaa ili kukusaidia kujibu swali.\n\nSwali: {question}",
    'ja': "次の質問を読んで、質問に答えるのに役立つ関連テキストをいくつか取得してください。\n\n質問: {question}",
    'fi': "Lue seuraava kysymys ja hae asiaankuuluvaa tekstiä, joka auttaa vastaamaan kysymykseen.\n\nKysymys: {question}",
    'bn': "নিচের প্রশ্নটি পড়ুন এবং প্রশ্নের উত্তর দেওয়ার জন্য কিছু প্রাসঙ্গিক লেখা সংগ্রহ করুন।\n\nপ্রশ্ন: {question}",
    'ru': "Прочитайте следующий вопрос и найдите соответствующий текст, который поможет ответить на него.\n\nВопрос: {question}",
    'el': "Διαβάστε την παρακάτω ερώτηση και ανακτήστε κάποιο σχετικό κείμενο που θα σας βοηθήσει να απαντήσετε στην ερώτηση.\n\nΕρώτηση: {question}",
    'fa': "سؤال زیر را بخوانید و چند متن مرتبط را برای کمک به پاسخ به سؤال بازیابی کنید.\n\nسوال: {question}",
    'ar': "اقرأ السؤال التالي واحصل على بعض النصوص ذات الصلة للمساعدة في الإجابة على السؤال.\n\nالسؤال: {question}",
    'az': "Aşağıdakı sualı oxuyun və suala cavab vermək üçün müvafiq mətni əldə edin.\n\nSual: {question}",
    'zh': "阅读以下问题并检索一些相关文本以帮助回答问题。\n\n问题：{question}",
    'id': "Bacalah pertanyaan berikut dan ambil beberapa teks relevan untuk membantu menjawab pertanyaan tersebut.\n\nPertanyaan: {question}",
    'es': "Lea la siguiente pregunta y recupere algún texto relevante que ayude a responderla.\n\nPregunta: {question}",
    'ko': "다음 질문을 읽고 질문에 답하는 데 도움이 되는 관련 텍스트를 검색하세요.\n\n질문: {question}",
    'su': "Baca patarosan di handap ieu sareng kéngingkeun sababaraha téks anu relevan pikeun ngabantosan ngajawab patarosan.\n\nPananya: {question}",
    'as': "তলৰ প্ৰশ্নটো পঢ়ক আৰু প্ৰশ্নটোৰ উত্তৰত সহায় কৰিবলৈ কিছুমান প্ৰাসংগিক লিখনী উদ্ধাৰ কৰক।\n\nপ্ৰশ্ন: {question}",
    'ha': "Karanta tambaya mai zuwa kuma a samo wani rubutu mai dacewa don taimakawa amsa tambayar.\n\nTambaya: {question}",
    'am': "ለጥያቄው መልስ እንዲረዳው የሚከተለውን ጥያቄ አንብብ እና አንዳንድ ተዛማጅ ጽሑፎችን ያውጣ።\n\nጥያቄ: {question}"
}

LANG_EMBED_INSTRUCTION_TEXT = {
    'en': "Given the input below, retrieve relevant texts that are semantically related. These  may include explanations, factual descriptions, or supporting details that help elaborate on or provide context for the input text.\n\nInput Text: {text}",
}