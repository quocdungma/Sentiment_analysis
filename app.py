
import random
import streamlit as st
import requests
import pandas as pd
import json
import pickle
import regex
from underthesea import word_tokenize, pos_tag, sent_tokenize
from pyvi import ViTokenizer
import re
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import joblib

# Load the trained model and vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('random_forest_model.pkl')

# Define the URL of the API
url = "https://tiki.vn/api/v2/products?q="
search_query = "may tinh bang"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36'
}

def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        # ...
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    #...
    return document

# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def find_words(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []

    for word in list_of_words:
        if word in document_lower:
            word_count += document_lower.count(word)
            word_list.append(word)

    return word_count, word_list

def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

neutral_words = ["chấp nhận được", "trung bình", "bình thường", "tạm ổn", "trung lập", "có thể"
                 "không nổi bật", "đủ ổn", "đủ tốt", "có thể chấp nhận", "bình thường",
                 "thường xuyên", "tương đối", "hợp lý", "tương tự",
                 "có thể sử dụng", "bình yên", "bình tĩnh", "không quá tệ", "trung hạng",
                 "có thể điểm cộng", "dễ chấp nhận", "không phải là vấn đề",
                 "không phản đối", "không quá đáng kể", "không gây bất ngờ", "không tạo ấn tượng", "có thể chấp nhận",
                 "không gây sốc", "tương đối tốt", "không thay đổi", "không quá phức tạp", "không đáng kể",
                 "chấp nhận", "có thể dễ dàng thích nghi", "không quá cầu kỳ", "không cần thiết", "không yêu cầu nhiều", "không gây hại",
                 "không có sự thay đổi đáng kể", "không rõ ràng", "không quá phê bình", "không đáng chú ý", "không đặc biệt",
                 "không quá phức tạp", "không gây phiền hà", "không đáng kể", "không gây kích thích"]

negative_words = [
    "kém", "tệ", "đau", "xấu", "bị","rè", "ồn",
    "buồn", "rối", "thô", "lâu", "sai", "hư", "dơ", "không có"
    "tối", "chán", "ít", "mờ", "mỏng", "vỡ", "hư hỏng",
    "lỏng lẻo", "khó", "cùi", "yếu", "mà", "không thích", "không thú vị", "không ổn",
    "không hợp", "không đáng tin cậy", "không chuyên nghiệp", "nhầm lẫn"
    "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", "không đáng giá",
    "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp", "bị mở", "bị khui", "không đúng", "không đúng sản phẩm",
    "khó hiểu", "khó chịu", "gây khó dễ", "rườm rà", "khó truy cập", "bị bóc", "sai sản phẩm",
    "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ","không rõ ràng", "giảm chất lượng",
    "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng tiền", "chưa đẹp", "không đẹp"
]

positive_words = [
    "thích", "tốt", "xuất sắc","đúng", "tuyệt vời", "tuyệt hảo", "đẹp", "ổn"
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "thú vị", "nhanh"
    "tiện lợi", "dễ sử dụng", "hiệu quả", "ấn tượng",
    "nổi bật", "tận hưởng", "tốn ít thời gian", "thân thiện", "hấp dẫn",
    "gợi cảm", "tươi mới", "lạ mắt", "cao cấp", "độc đáo",
    "hợp khẩu vị", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "hấp dẫn", "an tâm", "không thể cưỡng_lại", "thỏa mãn", "thúc đẩy",
    "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "tận tâm",
    "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền"
]

# Danh sách các từ mang ý nghĩa phủ định
negation_words = ["không", "nhưng", "tuy nhiên", "mặc dù", "chẳng", "mà", 'kém', 'giảm']

positive_emojis = [
    "😄", "😃", "😀", "😁", "😆",
    "😅", "🤣", "😂", "🙂", "🙃",
    "😉", "😊", "😇", "🥰", "😍",
    "🤩", "😘", "😗", "😚", "😙",
    "😋", "😛", "😜", "🤪", "😝",
    "🤗", "🤭", "🥳", "😌", "😎",
    "🤓", "🧐", "👍", "🤝", "🙌", "👏", "👋",
    "🤙", "✋", "🖐️", "👌", "🤞",
    "✌️", "🤟", "👈", "👉", "👆",
    "👇", "☝️"
]

# Count emojis positive and negative
negative_emojis = [
    "😞", "😔", "🙁", "☹️", "😕",
    "😢", "😭", "😖", "😣", "😩",
    "😠", "😡", "🤬", "😤", "😰",
    "😨", "😱", "😪", "😓", "🥺",
    "😒", "🙄", "😑", "😬", "😶",
    "🤯", "😳", "🤢", "🤮", "🤕",
    "🥴", "🤔", "😷", "🙅‍♂️", "🙅‍♀️",
    "🙆‍♂️", "🙆‍♀️", "🙇‍♂️", "🙇‍♀️", "🤦‍♂️",
    "🤦‍♀️", "🤷‍♂️", "🤷‍♀️", "🤢", "🤧",
    "🤨", "🤫", "👎", "👊", "✊", "🤛", "🤜",
    "🤚", "🖕"
]

# Định nghĩa hàm tiền xử lý
def preprocess_input(input_text, emoji_dict, teen_dict, wrong_lst, neutral_words, negative_words, positive_words, negation_words, positive_emojis, negative_emojis, stopwords_lst):
    # Bước 1: Áp dụng hàm xử lý văn bản
    processed_text = process_text(input_text, emoji_dict, teen_dict, wrong_lst)

    # Bước 2: Chuyển đổi ký tự unicode
    processed_text = covert_unicode(processed_text)

    # Bước 3: Tính toán số lượng từ và emoji mang tính cảm xúc
    neutral_word_count = find_words(processed_text, neutral_words)[0]
    negative_word_count = find_words(processed_text, negative_words)[0]
    positive_word_count = max(find_words(processed_text, positive_words)[0] - find_words(processed_text, negation_words)[0],0)
    positive_emoji_count = find_words(processed_text, positive_emojis)[0]
    negative_emoji_count = find_words(processed_text, negative_emojis)[0]

    # Bước 4: Tách từ và loại bỏ stopwords
    tokenized_text = word_tokenize(processed_text, format="text")
    tokenized_text = remove_stopword(tokenized_text, stopwords_lst)

    # Bước 5: Áp dụng POS tagging
    tokenized_text = ViTokenizer.tokenize(tokenized_text)
    tokenized_text = re.sub(r'\.', '', tokenized_text)

    # Gom các thông tin đã xử lý vào một dictionary
    processed_data = {
        "processed_text": tokenized_text,
        "neutral_word_count": neutral_word_count,
        "negative_word_count": negative_word_count,
        "positive_word_count": positive_word_count,
        "positive_emoji_count": positive_emoji_count,
        "negative_emoji_count": negative_emoji_count
    }

    return processed_data

def predict_sentiment(user_input):
    # Bước 1: Tiền xử lý đầu vào từ người dùng
    processed_data = preprocess_input(user_input, emoji_dict, teen_dict, wrong_lst, neutral_words, negative_words, positive_words, negation_words, positive_emojis, negative_emojis, stopwords_lst)

    # Bước 2: Trích xuất các đặc trưng
    processed_text = processed_data['processed_text']
    feature_values = [
        processed_data['neutral_word_count'],
        processed_data['negative_word_count'],
        processed_data['positive_word_count'],
        processed_data['positive_emoji_count'],
        processed_data['negative_emoji_count']
    ]

    # Bước 3: Vector hóa văn bản đã được tiền xử lý
    text_vectorized = vectorizer.transform([processed_text])

    # Bước 4: Kết hợp đặc trưng văn bản với các đặc trưng bổ sung
    features_combined = hstack((text_vectorized, np.array(feature_values).reshape(1, -1)))

    # Bước 5: Viết hàm dự báo
    prediction = model.predict(features_combined)

    return prediction[0]

# Example usage:
# comment = "sản phẩm không chất lượng"

# prediction = predict_sentiment(comment)
# print(f"The predicted sentiment is: {prediction}")

# Thêm sidebar
st.sidebar.header('Sentiment analysis',divider='rainbow')

sidebar_option = st.sidebar.selectbox(
    "Lựa chọn nội dung",
    ("Giới thiệu", "🔍 Tìm kiếm theo sản phẩm", "Nhập bình luận để dự đoán")
)

st.sidebar.markdown(f"<h1 style='font-size:10px;'>😊 Tích cực</h1>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h1 style='font-size:10px;'>😐 Trung tính</h1>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h1 style='font-size:10px;'>😢 Tiêu cực</h1>", unsafe_allow_html=True)

if sidebar_option == "Giới thiệu":
    st.title("Giới thiệu về ứng dụng")
    st.header('Hướng dẫn sử dụng app:', divider='rainbow')
    st.subheader('1. Tìm kiếm sản phẩm:', divider='rainbow')
    st.write("   - Người dùng nhập vào sản phẩm cần tìm.")
    st.write("   - Bấm vào tên sản phẩm để xem comment.")
    st.write("   - Mô hình đọc và hỗ trợ dự báo comment.")
    st.write("   - Đối chiếu với comment thực tế.")
    st.write("   - Tránh trường hợp rating và comment không tương đồng.")

    st.subheader('2. Nhập bình luận để dự đoán:', divider='rainbow')
    st.write("   - Người dùng nhập vào một đoạn comment.")
    st.write("   - Mô hình đánh giá comment đó là tích cực, tiêu cực, hay trung tính.")

    st.header('Người thực hiện:', divider='rainbow')
    st.write("Mã Quốc Dũng")
    st.write("Nguyễn Thanh Trọng")

    st.header('Nguồn dữ liệu:', divider='rainbow')
    st.write("Dữ liệu được thu thập từ hai nguồn chính: Tiki và Sendo.")
    st.write("Dữ liệu tìm kiếm được cào trực tiếp từ trang sản phẩm của Tiki.")

    st.header('Phương pháp sử dụng:', divider='rainbow')
    st.write("Phương pháp Random Forest")
    st.write("Độ chính xác: gần 94%")
    st.image("https://editor.analyticsvidhya.com/uploads/74060RF%20image.jpg", caption="Mô hình Random Forest")

elif sidebar_option == "🔍 Tìm kiếm theo sản phẩm":
    st.title("Tìm kiếm nhận xét theo sản phẩm")

    search_query = st.text_input("Nhập thông tin:")
    if st.button("Tìm kiếm 🔍"):
        st.write(f"Kết quả tìm kiếm cho: {search_query}")

    # Lấy sản phẩm products
    import requests

    # Define the URL of the API
    url = "https://tiki.vn/api/v2/products?q=" + search_query

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes

        data = response.json()

    except requests.exceptions.RequestException as e:
        st.write(f"An error occurred: {e}")
        data = {"data": []}
    except json.JSONDecodeError as e:
        st.write(f"Error decoding JSON: {e}")
        data = {"data": []}

    # Extracting the necessary information and creating a DataFrame
    review_link = 'https://tiki.vn/api/v2/reviews?product_id='
    # Extracting the necessary information and creating a list of dictionaries
    product_list = []
    for item in data["data"]:
        product = {
            "product_id": str(item["id"]),
            "name": item["name"],
            "sold": item.get("quantity_sold", {}).get("value", "N/A"),
            "price": item.get("price", "N/A"),
            "image": item.get("thumbnail_url", "N/A"),
            "review_count": item.get("review_count", "N/A"),
            "rating_average": item.get("rating_average", "N/A"),
            "link_comment": review_link + str(item["id"])
        }
        product_list.append(product)
    products = product_list


    current_product_id = None
    # Lấy comments dựa vào products và cột link_comment
    comments = []
    for product in products:
        review_url = product['link_comment']
        try:
            review_response = requests.get(review_url, headers=headers)
            review_response.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes

            review_data = review_response.json()

            comments_productid = [
                {
                    "product_id": product["product_id"],
                    "username": review.get("created_by", {}).get("name", "Anonymous"),
                    "rating": review.get("rating", 0),
                    "content": review.get("content", "No content"),
                }
                for review in review_data.get("data", [])
            ]
            comments.extend(comments_productid)
        except requests.exceptions.RequestException as e:
            st.write(f"An error occurred: {e}")
        except json.JSONDecodeError as e:
            st.write(f"Error decoding JSON: {e}")

    st.session_state.comments = comments


    # Display all the comments or process them as you need
    # for comment in st.session_state.comments:
    #    st.write(f"Product ID: {comment['product_id']}")
    #    st.write(f"Username: {comment['username']}")
    #    st.write(f"Rating: {comment['rating']} stars")
    #    st.write(f"Content: {comment['content']}")

    # Debug: print the number of comments fetched
    st.write(f"Number of comments fetched: {len(comments)}")

    def display_comments(product_id):
        product_comments = [comment for comment in st.session_state.comments if comment["product_id"] == product_id]
        for comment in product_comments:
            st.write(f"Tên: {comment['username']}")
            st.write(f"Đánh giá: {comment['rating']} sao")
            st.write(f"Nội dung: {comment['content']}")

            prediction = predict_sentiment(comment['content'])
            if prediction == 'positive':
                st.markdown(f"<h1 style='font-size:20px;'>Predict: 😊</h1>", unsafe_allow_html=True)
            elif prediction == 'neutral':
                st.markdown(f"<h1 style='font-size:20px;'>Predict: 😐</h1>", unsafe_allow_html=True)
            else: # Assuming 'negative' is the only other value returned by the predict function
                st.markdown(f"<h1 style='font-size:20px;'>Predict: 😢</h1>", unsafe_allow_html=True)

    cols = st.columns(2)
    for i, product in enumerate(products):
        with cols[i % 2]:
            st.image(product["image"])
            if st.button(product["name"], key=f"product_{product['product_id']}"):
                current_product_id = product["product_id"]
                display_comments(product["product_id"])
            # Điều chỉnh cỡ chữ và định dạng chữ cho các trường 'sold' và 'price'
            st.markdown(f"<p style='font-size: small;'>Đã bán: {product['sold']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: small;'>Reviews: {product['review_count']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: small;'>Rate: {product['rating_average']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: larger; font-weight: bold;'>Giá: {int(product['price']):,} đ</p>", unsafe_allow_html=True)

elif sidebar_option == "Nhập bình luận để dự đoán":
    st.title("Đánh giá nhận xét của người dùng")
    user_input = st.text_area("Nhập bình luận của bạn ở đây")
    if st.markdown("""
        <style>
        .custom-button {
            font-size: 20px;
            height: 50px;
            width: 200px;
            border: 2px solid white;
            border-radius: 20px;
            text-align: center;
            line-height: 50px; /* This will vertically center the text */
            padding-left: 1px; /* This will horizontally align the text to the left */
        }
        </style>
        <button class="custom-button" onclick="handleClick()">🔍 Dự đoán</button>
        <script>
            function handleClick() {
                // Logic to handle button click
            }
        </script>
        """, unsafe_allow_html=True):
        prediction = predict_sentiment(user_input)
        if prediction == 'positive':
            st.markdown(f"<h1 style='font-size:30px;'>Predict: 😊</h1>", unsafe_allow_html=True)
        elif prediction == 'neutral':
            st.markdown(f"<h1 style='font-size:30px;'>Predict: 😐</h1>", unsafe_allow_html=True)
        else: # Assuming 'negative' is the only other value returned by the predict function
            st.markdown(f"<h1 style='font-size:30px;'>Predict: 😢</h1>", unsafe_allow_html=True)
