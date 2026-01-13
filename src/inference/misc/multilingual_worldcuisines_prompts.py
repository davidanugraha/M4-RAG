from typing import List, Optional
from pydantic import BaseModel
import outlines


@outlines.prompt
def construct_worldcuisines_mcq_prompt_ar(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    بالنظر إلى سؤال الاختيار من متعدد أدناه، اختر أفضل إجابة واحدة بناءً على السؤال وأي سياق ذي صلة مقدم. استجب فقط برقم الخيار الصحيح (أي 1، 2، 3، 4، أو 5). استخدم السياق إذا كان مفيدًا، ولكن تجاهل المعلومات غير ذات الصلة.

    ### سؤال
    {{ question }}
    {% if context_list %}

    ### سياق
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### خيارات
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### تنسيق الإجابة
    يرجى تقديم إجابتك بالتنسيق JSON المحدد:

    {{ format | schema }}

    ### استجابة
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_id_casual(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Dikasih pertanyaan pilihan ganda di bawah, pilih satu jawaban terbaik berdasarkan pertanyaan dan konteks yang dikasih. Jawab cuma nomor pilihan yang bener aja (misal: 1, 2, 3, 4, atau 5). Pake konteks kalo perlu, tapi abaikan info yang nggak nyambung.

    ### Pertanyaan
    {{ question }}
    {% if context_list %}

    ### Konteks
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Pilihan
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Format Jawaban
    Tolong jawab pake format JSON yang udah ditentuin ya:

    {{ format | schema }}

    ### Jawaban
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_id_formal(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Diberikan pertanyaan pilihan ganda di bawah ini, pilih satu jawaban terbaik berdasarkan pertanyaan dan konteks relevan yang diberikan. Tanggapi hanya dengan nomor pilihan yang benar (misalnya, 1, 2, 3, 4, atau 5). Gunakan konteks jika membantu, tetapi abaikan informasi yang tidak relevan.

    ### Pertanyaan
    {{ question }}
    {% if context_list %}

    ### Konteks
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Pilihan
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Format Jawaban
    Mohon berikan jawaban Anda dalam format JSON yang ditentukan:

    {{ format | schema }}

    ### Tanggapan
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_jv_krama(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Am Colby pitakonan pilihan ganda ing ngisor iki, pilih siji wangsulan sing paling apik adhedhasar pitakonan lan konteks sing relevan. Wangsulana mung nomer pilihan sing bener (yaiku, 1, 2, 3, 4, utawa 5). Gunakake konteks yen mbiyantu, nanging ora nggatekake informasi sing ora ana gandhengane.

    ### Pitakonan
    {{ question }}
    {% if context_list %}

    ### Konteks
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Pilihan
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Format Wangsulan
    Mangga wenehana wangsulan panjenengan ing format JSON sing wis ditemtokake:

    {{ format | schema }}

    ### Wangsulan
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_jv_ngoko(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Dikei pitakonan pilihan ganda iki, pilih siji wangsulan sing paling apik adhedhasar pitakonan lan konteks sing relevan. Wangsulan mung nomer opsi sing bener (yaiku, 1, 2, 3, 4, utawa 5). Gunakake konteks yen mbiyantu, nanging ojo nggatekake informasi sing ora relevan.

    ### Pitakonan
    {{ question }}
    {% if context_list %}

    ### Konteks
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Pilihan
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Format Jawaban
    Mangga wenehana jawabanmu ing format JSON sing wis ditemtokake:

    {{ format | schema }}

    ### Wangsulan
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_su_loma(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Diberikeun patarosan pilihan ganda di handap, pilih hiji jawaban panghadéna dumasar kana patarosan sareng konteks anu aya. Tanggapan ngan ukur nomer pilihan anu leres (nyaéta, 1, 2, 3, 4, atanapi 5). Anggo konteks upami ngabantosan, tapi teu dipalire inpormasi anu teu aya hubunganana.

    ### Patarosan
    {{ question }}
    {% if context_list %}

    ### Konteks
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Pilihan
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Format Jawaban
    Punten masihan jawaban anjeun dina format JSON anu ditetepkeun:

    {{ format | schema }}

    ### Réspon
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_tl(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Batay sa tanong na multiple-choice sa ibaba, piliin ang pinakamahusay na sagot batay sa tanong at anumang nauugnay na konteksto na ibinigay. Tumugon lamang gamit ang numero ng tamang opsyon (hal., 1, 2, 3, 4, o 5). Gamitin ang konteksto kung ito ay nakakatulong, ngunit huwag pansinin ang mga hindi kaugnay na impormasyon.

    ### Tanong
    {{ question }}
    {% if context_list %}

    ### Konteksto
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Mga Opsyon
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Format ng Sagot
    Mangyaring ibigay ang iyong sagot sa tinukoy na format ng JSON:

    {{ format | schema }}

    ### Tugon
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_bn(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    নিচের বহু-বাছাই প্রশ্নের জন্য, প্রশ্ন এবং প্রদত্ত প্রাসঙ্গিক তথ্যের উপর ভিত্তি করে একটি সেরা উত্তর নির্বাচন করুন। শুধুমাত্র সঠিক বিকল্পের সংখ্যা দিয়ে উত্তর দিন (অর্থাৎ, 1, 2, 3, 4, বা 5)। প্রয়োজনে প্রসঙ্গ ব্যবহার করুন, কিন্তু অপ্রাসঙ্গিক তথ্য উপেক্ষা করুন।

    ### প্রশ্ন
    {{ question }}
    {% if context_list %}

    ### প্রসঙ্গ
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### বিকল্প
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### উত্তর বিন্যাস
    অনুগ্রহ করে নির্দিষ্ট JSON বিন্যাসে আপনার উত্তর প্রদান করুন।:

    {{ format | schema }}

    ### উত্তর
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_cs(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    U dané otázky s výběrem odpovědí vyberte jedinou nejlepší odpověď na základě otázky a jakéhokoli relevantního poskytnutého kontextu. Odpovídejte pouze číslem správné možnosti (tj. 1, 2, 3, 4 nebo 5). Pokud je to užitečné, použijte kontext, ale ignorujte nesouvisející informace.

    ### Otázka
    {{ question }}
    {% if context_list %}

    ### Kontext
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Možnosti
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Formát odpovědi
    Poskytněte prosím svou odpověď ve specifikovaném formátu JSON:

    {{ format | schema }}

    ### Odpověď
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_en(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Given the multiple-choice question below, choose the single best answer based on the question and any relevant context provided. Respond only with the number of the correct option (i.e., 1, 2, 3, 4, or 5). Use the context if helpful, but ignore unrelated information.

    ### Question
    {{ question }}
    {% if context_list %}

    ### Context
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Options
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Answer Format
    Please provide your answer in the specified JSON format:

    {{ format | schema }}

    ### Response
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_es(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Dada la siguiente pregunta de opción múltiple, elige la mejor respuesta única basada en la pregunta y cualquier contexto relevante proporcionado. Responde solo con el número de la opción correcta (es decir, 1, 2, 3, 4 o 5). Usa el contexto si es útil, pero ignora la información no relacionada.

    ### Pregunta
    {{ question }}
    {% if context_list %}

    ### Contexto
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Opciones
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Formato de respuesta
    Por favor, proporciona tu respuesta en el formato JSON especificado:

    {{ format | schema }}

    ### Respuesta
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_fr(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Étant donné la question à choix multiples ci-dessous, choisissez la meilleure réponse unique en fonction de la question et de tout contexte pertinent fourni. Répondez uniquement avec le numéro de l'option correcte (c'est-à-dire 1, 2, 3, 4 ou 5). Utilisez le contexte si cela vous est utile, mais ignorez les informations non pertinentes.

    ### Question
    {{ question }}
    {% if context_list %}

    ### Contexte
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Options
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Format de réponse
    Veuillez fournir votre réponse dans le format JSON spécifié:

    {{ format | schema }}

    ### Réponse
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_hi(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    नीचे दिए गए बहुविकल्पीय प्रश्न के आधार पर, प्रश्न और किसी भी प्रासंगिक संदर्भ के आधार पर सबसे अच्छा एकल उत्तर चुनें। केवल सही विकल्प की संख्या (अर्थात, 1, 2, 3, 4, या 5) के साथ उत्तर दें। यदि सहायक हो तो संदर्भ का उपयोग करें, लेकिन असंबंधित जानकारी को अनदेखा करें।

    ### प्रश्न
    {{ question }}
    {% if context_list %}

    ### संदर्भ
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### विकल्प
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### उत्तर प्रारूप
    कृपया अपना उत्तर निर्दिष्ट JSON प्रारूप में प्रदान करें:

    {{ format | schema }}

    ### प्रतिक्रिया
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_it(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Data la domanda a scelta multipla qui sotto, scegli la migliore risposta singola basata sulla domanda e su qualsiasi contesto pertinente fornito. Rispondi solo con il numero dell'opzione corretta (cioè 1, 2, 3, 4 o 5). Usa il contesto se utile, ma ignora le informazioni non correlate.

    ### Domanda
    {{ question }}
    {% if context_list %}

    ### Contesto
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Opzioni
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Formato risposta
    Si prega di fornire la risposta nel formato JSON specificato:

    {{ format | schema }}

    ### Risposta
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_mr(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    खालील बहुपर्यायी प्रश्न विचारात घ्या, प्रश्न आणि दिलेल्या कोणत्याही संबंधित संदर्भावर आधारित एकमेव सर्वोत्तम उत्तर निवडा. केवळ योग्य पर्यायाचा क्रमांक (म्हणजे 1, 2, 3, 4, किंवा 5) द्या. संदर्भ उपयुक्त असल्यास वापरा, परंतु असंबंधित माहितीकडे दुर्लक्ष करा.

    ### प्रश्न
    {{ question }}
    {% if context_list %}

    ### संदर्भ
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### पर्याय
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### उत्तराचे स्वरूप
    कृपया निर्दिष्ट JSON स्वरूपात आपले उत्तर द्या:

    {{ format | schema }}

    ### प्रतिसाद
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_ru_casual(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Для данного вопроса с несколькими вариантами ответа, выбери единственный лучший ответ, основываясь на вопросе и любом предоставленном контексте. Отвечай только номером правильного варианта (например, 1, 2, 3, 4 или 5). Если контекст полезен, используй его, но игнорируй не относящуюся к делу информацию.

    ### Вопрос
    {{ question }}
    {% if context_list %}

    ### Контекст
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Варианты
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Формат ответа
    Пожалуйста, предоставь свой ответ в указанном формате JSON:

    {{ format | schema }}

    ### Ответ
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_ru_formal(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Учитывая приведенный ниже вопрос с несколькими вариантами ответов, выберите единственный наилучший ответ на основе вопроса и любого предоставленного релевантного контекста. Отвечайте только номером правильного варианта (то есть 1, 2, 3, 4 или 5). Используйте контекст, если это необходимо, но игнорируйте нерелевантную информацию.

    ### Вопрос
    {{ question }}
    {% if context_list %}

    ### Контекст
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Варианты
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Формат ответа
    Пожалуйста, предоставьте свой ответ в указанном формате JSON:

    {{ format | schema }}

    ### Ответ
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_sc(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Cun sa chestione a tzogheddu de s'opzione de chircare chi abàlet, chirca sa única resone mègius chi s'agatat in base a sa chestione e a donzi àteru contextu chi est presentadu. Rispondin solu cun su nùmeru de s'opzione curreta (o de 1, 2, 3, 4, o 5). Usa su contextu si est de agiudu, ma ismentiga sas informatziones chi no sunt ligadas.

    ### Chestione
    {{ question }}
    {% if context_list %}

    ### Contextu
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Opziones
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Formatu de resposta
    Per favore, dae sa risposta tua in su JSON formatu indicadu:

    {{ format | schema }}

    ### Risposta
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_si_formal_spoken(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    පහත දැක්වෙන බහුවරණ ප්‍රශ්නය සඳහා, ප්‍රශ්නය සහ සපයා ඇති අදාළ සන්දර්භය මත පදනම්ව, හොඳම තනි පිළිතුර තෝරන්න. නිවැරදි විකල්පයේ අංකය පමණක් (එනම්, 1, 2, 3, 4, හෝ 5) ලෙස පිළිතුරු දෙන්න. සන්දර්භය ප්‍රයෝජනවත් නම් භාවිතා කරන්න, නමුත් අදාළ නොවන තොරතුරු නොසලකා හරින්න.

    ### ප්‍රශ්නය
    {{ question }}
    {% if context_list %}

    ### සන්දර්භය
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### විකල්ප
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### පිළිතුරු ආකෘතිය
    කරුණාකර ඔබේ පිළිතුර සඳහන් කරන ලද JSON ආකෘතියෙන් සපයන්න:

    {{ format | schema }}

    ### පිළිතුර
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_ja_casual(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    下の多肢選択式問題について、質問と関連する文脈に基づいて最も良い答えを一つ選んで。答えは正しい選択肢の番号（つまり、1、2、3、4、または5）のみで答えて。文脈が役立つなら使って、関係ない情報は無視して。

    ### 質問
    {{ question }}
    {% if context_list %}

    ### 文脈
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### 選択肢
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### 回答形式
    指定されたJSON形式で回答してね。:

    {{ format | schema }}

    ### 回答
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_ja_formal(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    提供された質問と関連する文脈に基づき、最も適切な単一の選択肢を選んでください。回答は正しい選択肢の番号（例：1、2、3、4、または5）のみとしてください。文脈が役立つ場合は使用し、無関係な情報は無視してください。

    ### 質問
    {{ question }}
    {% if context_list %}

    ### 文脈
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### 選択肢
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### 回答形式
    指定されたJSON形式で回答をご提供ください。:

    {{ format | schema }}

    ### 回答
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_ko_casual(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    아래 객관식 문제에서 질문과 제공된 관련 내용을 바탕으로 가장 좋은 답을 하나 골라줘. 정답 선택지의 번호(예: 1, 2, 3, 4, 또는 5)만 응답으로 해줘. 내용이 도움이 되면 사용하되, 관련 없는 정보는 무시해도 돼.

    ### 문제
    {{ question }}
    {% if context_list %}

    ### 내용
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### 선택지
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### 답변 형식
    지정된 JSON 형식으로 답변해 줘:

    {{ format | schema }}

    ### 응답
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_ko_formal(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    아래의 객관식 질문에 대해, 질문과 제공된 관련 맥락을 바탕으로 단 하나의 최적의 답을 선택하십시오. 정답의 번호(예: 1, 2, 3, 4, 또는 5)만 응답하십시오. 맥락이 도움이 된다면 사용하되, 관련 없는 정보는 무시하십시오.

    ### 질문
    {{ question }}
    {% if context_list %}

    ### 맥락
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### 선택지
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### 응답 형식
    지정된 JSON 형식으로 답변을 제공해주십시오:

    {{ format | schema }}

    ### 응답
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_th(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    เมื่อพิจารณาคำถามแบบเลือกตอบด้านล่างนี้ ให้เลือกคำตอบที่ดีที่สุดเพียงข้อเดียวตามคำถามและบริบทที่เกี่ยวข้องที่ให้ไว้ ตอบเพียงหมายเลขของตัวเลือกที่ถูกต้องเท่านั้น (เช่น 1, 2, 3, 4 หรือ 5) ใช้บริบทถ้าเป็นประโยชน์ แต่ให้ละเว้นข้อมูลที่ไม่เกี่ยวข้อง

    ### คำถาม
    {{ question }}
    {% if context_list %}

    ### บริบท
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### ตัวเลือก
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### รูปแบบคำตอบ
    โปรดให้คำตอบของคุณในรูปแบบ JSON ที่ระบุ:

    {{ format | schema }}

    ### คำตอบ
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_yo(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Fun ibeere ti o ni awọn yiyan pupọ ti o wa ni isalẹ, yan idahun kan ti o dara julọ da lori ibeere naa ati eyikeyi ọrọ ti o yẹ ti a pese. Dahun nikan pẹlu nọmba ti aṣayan ti o tọ (ie, 1, 2, 3, 4, tabi 5). Lo ọrọ ti o ba wulo, ṣugbọn foju eyikeyi alaye ti ko ṣe pataki.

    ### Ibeere
    {{ question }}
    {% if context_list %}

    ### Ọrọ
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Awọn yiyan
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Ọna Idahun
    Jọwọ pese idahun rẹ ni ọna JSON ti a sọ:

    {{ format | schema }}

    ### Idahun
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_nan(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Tiong ti chhoán-chè siok-tè tiⁿ-chòe thiaⁿ-kóng, chiūⁿ-chòe siok-tè thiaⁿ-kóng chòe tiⁿ-chòe chhoán-chè. Chhap chòe tiⁿ-chòe thiaⁿ-kóng tiⁿ-chòe 1, 2, 3, 4, iah-sī 5. Chhap-chòe tiⁿ-chòe tiⁿ-chòe tiⁿ-chòe tiⁿ-chòe tiⁿ-chòe tiⁿ-chòe.

    ### Thiaⁿ-kóng
    {{ question }}
    {% if context_list %}

    ### Tiⁿ-chòe
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Chhoán-chè
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Chhoán-chè chhoán-chè
    Chhoán-chè chhoán-chè tiⁿ-chòe chhoán-chè chhoán-chè JSON chhoán-chè:

    {{ format | schema }}

    ### Chhoán-chè
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_nan_spoken(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Thiau siu li ê toan-suán tē-sìⁿ ê būn-tê, chhiūⁿ-kòe būn-tê kap ū sêng-kuan ê kò͘-chāi, chha̍p chòe-hó ê tiāu-hōe. Chhú-thiat siú-tiāu ê tiāu-hōe (chhiūⁿ 1, 2, 3, 4, 5). Ū-kiōng chhiūⁿ kò͘-chāi nā ū hāng-chúi, tān-sī bô-sǹg ū liân-kòan ê chu-liāu.

    ### Būn-tê
    {{ question }}
    {% if context_list %}

    ### Kò͘-chāi
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Tiāu-hōe
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Sî-chāi-pāng
    Chhiáⁿ chòe-chòe siú-tiāu ê JSON sî-chāi-pāng ê tòe-ōe:

    {{ format | schema }}

    ### Tòe-ōe
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_yue(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    根據以下嘅選擇題，根據問題同埋提供嘅相關內容，揀出一個最佳答案。只係用正確選項嘅號碼（例如 1, 2, 3, 4, 或者 5）嚟回應。如果內容有幫助，請使用佢，但忽略唔關事嘅資訊。

    ### 問題
    {{ question }}
    {% if context_list %}

    ### 內容
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### 選項
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### 答案格式
    請用指定嘅JSON格式提供你嘅答案。:

    {{ format | schema }}

    ### 回應
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_zh_cn(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    根据以下多项选择题，根据问题和提供的任何相关背景信息，选择唯一的最佳答案。仅以正确选项的编号（即 1、2、3、4 或 5）作为响应。如果背景信息有帮助，请使用它，但忽略无关信息。

    ### 问题
    {{ question }}
    {% if context_list %}

    ### 背景
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### 选项
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### 回答格式
    请以指定的 JSON 格式提供您的答案。:

    {{ format | schema }}

    ### 响应
    """


@outlines.prompt
def construct_worldcuisines_mcq_prompt_az(question: str, format: BaseModel, options: List[str], context_list: Optional[List[str]]=None):
    """
    Aşağıda verilmiş çoxseçimli suala əsasən, suala və verilmiş hər hansı əlaqəli kontekstə uyğun olaraq yeganə ən yaxşı cavabı seçin. Yalnız düzgün variantın nömrəsi ilə cavab verin (yəni, 1, 2, 3, 4 və ya 5). Kontekstdən faydalı olsa istifadə edin, lakin əlaqəli olmayan məlumatları nəzərə almayın.

    ### Sual
    {{ question }}
    {% if context_list %}

    ### Kontekst
    {% for context in context_list %}
    - {{ context }}
    {% endfor %}

    {% endif %}
    ### Variantlar
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}

    ### Cavabın formatı
    Zəhmət olmasa cavabınızı müəyyən edilmiş JSON formatında təqdim edin:

    {{ format | schema }}

    ### Cavab
    """

WORLD_CUISINES_MCQ_PROMPTS = {
    "ar": construct_worldcuisines_mcq_prompt_ar,
    "id_casual": construct_worldcuisines_mcq_prompt_id_casual,
    "id_formal": construct_worldcuisines_mcq_prompt_id_formal,
    "jv_krama": construct_worldcuisines_mcq_prompt_jv_krama,
    "jv_ngoko": construct_worldcuisines_mcq_prompt_jv_ngoko,
    "su_loma": construct_worldcuisines_mcq_prompt_su_loma,
    "tl": construct_worldcuisines_mcq_prompt_tl,
    "bn": construct_worldcuisines_mcq_prompt_bn,
    "cs": construct_worldcuisines_mcq_prompt_cs,
    "en": construct_worldcuisines_mcq_prompt_en,
    "es": construct_worldcuisines_mcq_prompt_es,
    "fr": construct_worldcuisines_mcq_prompt_fr,
    "hi": construct_worldcuisines_mcq_prompt_hi,
    "it": construct_worldcuisines_mcq_prompt_it,
    "mr": construct_worldcuisines_mcq_prompt_mr,
    "ru_casual": construct_worldcuisines_mcq_prompt_ru_casual,
    "ru_formal": construct_worldcuisines_mcq_prompt_ru_formal,
    "sc": construct_worldcuisines_mcq_prompt_sc,
    "si_formal_spoken": construct_worldcuisines_mcq_prompt_si_formal_spoken,
    "ja_casual": construct_worldcuisines_mcq_prompt_ja_casual,
    "ja_formal": construct_worldcuisines_mcq_prompt_ja_formal,
    "ko_casual": construct_worldcuisines_mcq_prompt_ko_casual,
    "ko_formal": construct_worldcuisines_mcq_prompt_ko_formal,
    "th": construct_worldcuisines_mcq_prompt_th,
    "yo": construct_worldcuisines_mcq_prompt_yo,
    "nan": construct_worldcuisines_mcq_prompt_nan,
    "nan_spoken": construct_worldcuisines_mcq_prompt_nan_spoken,
    "yue": construct_worldcuisines_mcq_prompt_yue,
    "zh_cn": construct_worldcuisines_mcq_prompt_zh_cn,
    "az": construct_worldcuisines_mcq_prompt_az,
}