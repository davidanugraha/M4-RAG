import spacy
from spacy.util import get_installed_models

# Global dictionary to store loaded SpaCy models
SPACY_MODELS = {}

def preload_spacy_models():
    """Automatically detect and load all installed SpaCy models."""
    installed_models = get_installed_models()  # Get all installed SpaCy models
    lang_to_model = {}

    for model in installed_models:
        lang_code = model.split("_")[0]  # Extract language prefix (e.g., "en_core_web_sm" → "en")
        
        if lang_code == "xx":
            lang_to_model["xx"] = "xx_sent_ud_sm"
        elif lang_code not in lang_to_model:  # Store only one model per language
            lang_to_model[lang_code] = model

    # Load the detected models
    for lang, model in lang_to_model.items():
        SPACY_MODELS[lang] = spacy.load(model)

def get_spacy_model(lang):
    """Attempt to load the SpaCy model dynamically based on the language code."""
    if lang in SPACY_MODELS:
        return SPACY_MODELS[lang]
    else:
        return SPACY_MODELS["xx"]

def count_tokens(text, tokenizer=None, use_tokens=False):
    """Count tokens in the text using the provided tokenizer function."""
    if use_tokens:
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        return len(text)

def split_sentence(sentence, max_tokens, overlap_tokens, tokenizer=None, use_tokens=False):
    """
    If a single sentence exceeds max_tokens, split it into smaller pieces at the token level.
    Overlap is maintained between splits.
    """
    if use_tokens:
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        chunks = []
        start = 0

        while start < len(token_ids):
            end = start + max_tokens
            chunk_ids = token_ids[start:end]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            chunks.append(chunk_text)

            # Move start to enable overlap with the previous chunk.
            start = max(0, end - overlap_tokens)

        return chunks
    else:
        chunks = []
        start = 0

        while start < len(sentence):
            end = start + max_tokens
            chunks.append(sentence[start:end])

            # Move start to enable overlap with the previous chunk.
            start = max(0, end - overlap_tokens)

        return chunks

def split_by_sentence(text, max_tokens, overlap_tokens, nlp, tokenizer=None, use_tokens=False):
    """
    Splits text into chunks based on sentence boundaries.
    Uses regex to split on punctuation (.!?). If a sentence is too long, it is further split
    at the token level.
    """
    # Split text on sentence-ending punctuation.
    sentences = [sent.text.strip() for sent in nlp(text).sents]
    
    final_chunks = []
    current_chunk_list = []
    current_chunk_list_len = 0
    
    for sentence in sentences:
        sentence_token_count = count_tokens(sentence, tokenizer=tokenizer, use_tokens=use_tokens)
        
        # If adding the sentence stays within the limit, append it.
        if current_chunk_list_len + sentence_token_count <= max_tokens:
            current_chunk_list.append(sentence)
            current_chunk_list_len += sentence_token_count
        elif sentence_token_count > max_tokens:
            # Flush current chunk, then start a new one with overlap from previous chunk.
            if len(current_chunk_list) != 0:
                final_chunks.append(" ".join(current_chunk_list))
            sub_chunks = split_sentence(sentence, max_tokens, overlap_tokens, tokenizer=tokenizer, use_tokens=use_tokens)
            final_chunks.extend(sub_chunks[:-1])
            current_chunk_list = [sub_chunks[-1]]
            current_chunk_list_len = count_tokens(current_chunk_list[0], tokenizer=tokenizer, use_tokens=use_tokens)
        else:
            # Get some overlapping from previous sentences if possible
            if len(current_chunk_list) != 0:
                final_chunks.append(" ".join(current_chunk_list))
            i = len(current_chunk_list) - 1
            overlap_sub_chunks = [sentence]
            added_chunks_length = 0
            while i >= 0:
                new_mini_chunk_length = count_tokens(current_chunk_list[i], tokenizer=tokenizer, use_tokens=use_tokens)
                if added_chunks_length + new_mini_chunk_length <= overlap_tokens and \
                    new_mini_chunk_length + added_chunks_length + sentence_token_count <= max_tokens:
                    added_chunks_length += new_mini_chunk_length
                    overlap_sub_chunks.insert(0, current_chunk_list[i])
                    i -= 1
                else:
                    break
                
            current_chunk_list = overlap_sub_chunks
            current_chunk_list_len = added_chunks_length + sentence_token_count
            
    if len(current_chunk_list) != 0:
        final_chunks.append(" ".join(current_chunk_list))
    
    return final_chunks

def recursive_text_split(paragraph_list, max_tokens, overlap_tokens,
                         nlp, tokenizer=None, use_tokens=False):
    """
    `use_tokens` is either False (for using characters) or True (for using tokenizer)
    
    Recursively splits text into chunks no larger than max_tokens tokens.
    
    Process:
      1. Split the text into paragraphs using double newlines ("\n\n").
      2. Combine paragraphs into chunks until adding a new one would exceed max_tokens.
      3. If a single paragraph exceeds max_tokens, further split it by sentence boundaries.
      4. For each new chunk (except the first), prepend the last `overlap_tokens` tokens from the previous chunk.
    """
    # Base case: if the whole text is within the token limit, return it as one chunk.
    if count_tokens("\n\n".join(paragraph_list), tokenizer=tokenizer, use_tokens=use_tokens) <= max_tokens:
        return ["\n\n".join(paragraph_list)]
    
    # Split text into paragraphs.
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraph_list:
        para_tokens = count_tokens(paragraph, tokenizer=tokenizer, use_tokens=use_tokens)
        
        if count_tokens(current_chunk, tokenizer=tokenizer, use_tokens=use_tokens) + para_tokens <= max_tokens:
            # Append paragraph to current_chunk.
            current_chunk = paragraph if not current_chunk else current_chunk + "\n\n" + paragraph
        else:
            # current_chunk + paragraph would exceed max_tokens.
            paragraph = paragraph if not current_chunk else current_chunk + "\n\n" + paragraph
            
            # Paragraph is too large; split by sentences.
            sentence_chunks = split_by_sentence(paragraph, max_tokens, overlap_tokens, nlp,
                                                tokenizer=tokenizer, use_tokens=use_tokens)

            # Append the sentence_chunks (which are already respecting sentence boundaries)
            chunks.extend(sentence_chunks[:-1])
            current_chunk = sentence_chunks[-1]

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
