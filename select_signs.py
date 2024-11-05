import os
import pickle

def get_messages(lang):
    if lang.upper() == 'VN':
        return {
            'file_not_found': "Không tìm thấy tệp '{file_name}'.",
            'key_not_found': "Không tìm thấy khóa 'words' trong dữ liệu.",
            'enter_words': "Vui lòng nhập 10 từ (từng từ một hoặc tách biệt bởi dấu phẩy):",
            'word_not_found': "Không tìm thấy từ: {word}",
            'word_chosen': "{word} đã chọn với {count} biến thể: {variants}",
            'more_words_needed': "Cần thêm {count} từ nữa.",
            'duplicates_found': "Các từ trùng lặp đã bị loại bỏ: {words}",
            'selected_words_variants': "Bạn đã chọn các từ và biến thể sau:",
            'confirm_all_correct': "Tất cả đều đúng chứ? (có/không):",
            'variants_written': "Các biến thể đã được ghi vào '{file_name}'.",
            'process_terminated': "Quá trình đã bị hủy bỏ.",
            'choose_language': "Chọn ngôn ngữ: EN/VN?",
            'invalid_input': "Đầu vào không hợp lệ. Vui lòng nhập 'EN' hoặc 'VN'.",
            'yes': 'có',
        }
    else:
        return {
            'file_not_found': "File '{file_name}' not found.",
            'key_not_found': "Key 'words' not found in the data.",
            'enter_words': "Please enter 10 words (either one by one or separated by commas):",
            'word_not_found': "Word not found: {word}",
            'word_chosen': "{word} chosen with {count} variants: {variants}",
            'more_words_needed': "{count} more word(s) needed.",
            'duplicates_found': "Duplicate words removed: {words}",
            'selected_words_variants': "You have selected the following words and variants:",
            'confirm_all_correct': "Are all these correct? (yes/no):",
            'variants_written': "Variants written to '{file_name}'.",
            'process_terminated': "Process terminated.",
            'choose_language': "Choose language: EN/VN?",
            'invalid_input': "Invalid input. Please enter 'EN' or 'VN'.",
            'yes': 'yes',
        }

def load_data(file_path, messages):
    if not os.path.exists(file_path):
        print(messages['file_not_found'].format(file_name=file_path))
        exit(1)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    if 'words' not in data:
        print(messages['key_not_found'])
        exit(1)
    return data['words']

def get_user_words(words_data, messages):
    words_list = []
    duplicates = set()
    print(messages['enter_words'])
    while len(words_list) < 10:
        user_input = input().strip()
        if ',' in user_input:
            words = [w.strip() for w in user_input.split(',')]
            is_line_by_line = False
        else:
            words = [user_input.strip()]
            is_line_by_line = True

        for word in words:
            if word in words_list:
                duplicates.add(word)
                continue
            if word in words_data:
                variants = words_data[word]['variants']
                if is_line_by_line:
                    print(messages['word_chosen'].format(
                        word=word,
                        count=len(variants),
                        variants=', '.join(variants)
                    ))
                words_list.append(word)
            else:
                print(messages['word_not_found'].format(word=word))

        if duplicates:
            print(messages['duplicates_found'].format(words=', '.join(duplicates)))
            duplicates.clear()

        remaining = 10 - len(words_list)
        if remaining > 0:
            print(messages['more_words_needed'].format(count=remaining))

    return words_list[:10], is_line_by_line

def confirm_words(valid_words, is_line_by_line, messages):
    confirmed = True
    if is_line_by_line:
        # Already confirmed during input
        pass
    else:
        print(messages['selected_words_variants'])
        for word, variants in valid_words:
            print(f"{word}: {', '.join(variants)}")
        response = input(messages['confirm_all_correct']).strip().lower()
        if response != messages['yes']:
            confirmed = False
    return confirmed

def write_variants_to_file(valid_words, file_name, messages):
    with open(file_name, 'w') as f:
        for _, variants in valid_words:
            for variant in variants:
                f.write(f"{variant}\n")
    print(messages['variants_written'].format(file_name=file_name))

def main():
    while True:
        lang = input(get_messages('EN')['choose_language']).strip()
        if lang.strip().upper() in ['EN', 'VN']:
            messages = get_messages(lang)
            break
        else:
            print(get_messages('EN')['invalid_input'])
    words_data = load_data('evaluation_pack.pkl', messages)
    words_list, is_line_by_line = get_user_words(words_data, messages)
    valid_words = [(word, words_data[word]['variants']) for word in words_list]
    if confirm_words(valid_words, is_line_by_line, messages):
        write_variants_to_file(valid_words, 'pending_training.txt', messages)
    else:
        print(messages['process_terminated'])

if __name__ == "__main__":
    main()