import os
import re
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def read_chat_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        chat_lines = file.readlines()
    return chat_lines

def parse_chat(chat_lines):
    user_msgs = []
    ai_msgs = []
    for line in chat_lines:
        line = line.strip()
        if line.startswith("User:"):
            user_msgs.append(line.replace("User:", "").strip())
        elif line.startswith("AI:"):
            ai_msgs.append(line.replace("AI:", "").strip())
    return user_msgs, ai_msgs

def count_messages(user_msgs, ai_msgs):
    return len(user_msgs) + len(ai_msgs), len(user_msgs), len(ai_msgs)

def clean_text(messages):
    stop_words = set(stopwords.words('english'))
    all_text = ' '.join(messages).lower()
    all_text = all_text.translate(str.maketrans('', '', string.punctuation))
    words = re.findall(r'\b\w+\b', all_text)
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

def extract_keywords(words, top_n=5):
    counter = Counter(words)
    return counter.most_common(top_n)

def extract_keywords_tfidf(user_msgs, ai_msgs, top_n=5):
    docs = [' '.join(user_msgs), ' '.join(ai_msgs)]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(docs)
    feature_names = tfidf.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    keyword_scores = list(zip(feature_names, scores))
    sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]

def generate_summary(user_msgs, ai_msgs, keywords):
    total, user_count, ai_count = count_messages(user_msgs, ai_msgs)
    print("Summary:")
    print(f"- The conversation had {total} exchanges.")
    if 'python' in [kw[0] for kw in keywords]:
        print("- The user asked mainly about Python and its uses.")
    elif 'machine' in [kw[0] for kw in keywords]:
        print("- The conversation focused on machine learning.")
    else:
        print("- The conversation was general.")
    print("- Most common keywords:", ', '.join([kw[0] for kw in keywords]))

def summarize_chat_log(file_path, use_tfidf=False):
    print(f"\nAnalyzing chat log: {file_path}")
    chat_lines = read_chat_log(file_path)
    user_msgs, ai_msgs = parse_chat(chat_lines)
    if use_tfidf:
        keywords = extract_keywords_tfidf(user_msgs, ai_msgs)
    else:
        all_msgs = user_msgs + ai_msgs
        words = clean_text(all_msgs)
        keywords = extract_keywords(words)
    generate_summary(user_msgs, ai_msgs, keywords)

if __name__ == "__main__":
    # Single file summary
    summarize_chat_log("chat.txt", use_tfidf=True)

    # Uncomment for folder-based processing
    # summarize_folder("chat_logs", use_tfidf=True)
