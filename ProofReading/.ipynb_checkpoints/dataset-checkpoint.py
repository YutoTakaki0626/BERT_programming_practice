
import unicodedata

def create_dateset(data_df):

	tokenizer = SC_tokenizer.from_pretrained(MODEL_NAME)

	def check_token_count(row):
		'''
		誤変換の文章と正しい文章でトークンに対応がつくかどうかを判定
		条件:(トークン数が同じ・異なるトークンが２個以内)
		'''

		wrong_text_tokens = tokenizer.tokenize(row['wrong_text'])
		correct_text_tokens = tokenizer.tokenize(row['correct_text'])
		if len(wrong_text_tokens) != len(correct_text_tokens):
			return False

			diff_count = 0
			threshold_count = 2
			for wrong_text_token, correct_text_token in zip(wrong_text_tokens, correct_text_tokens):
				if wrong_text_token != correct_text_token:
					diff_count += 1
					if diff_count > threshold_count:
						return False

		return True

	def normalize(text):
		'''
		文字列の正規化
		'''
		text = text.strip()
		text = unicodedata.normalize('NFKC', text)
		return text

	# 漢字の誤変換データのみ抜き出す
	category_type = 'kanji-conversion'
	data_df.query('category == @category_type', inplace=True)
    data_df.rename(columns={'pre_text': 'wrong_text', 'post_text': 'correct_text'}, inplace=True)
    data_df['wrong_text'] = data_df['wrong_text'].map(normalize) 
    data_df['correct_text'] = data_df['correct_text'].map(normalize)
    kanji_conversion_num = len(data_df)
    data_df = data_df[data_df.apply(check_token_count, axis=1)]
    same_tokens_count_num = len(data_df)
    print(
        f'- 漢字誤変換の総数：{kanji_conversion_num}',
        f'- トークンの対応関係のつく文章の総数: {same_tokens_count_num}',
        f'  (全体の{same_tokens_count_num/kanji_conversion_num*100:.0f}%)',
        sep = '\n'
    )
    return data_df[['wrong_text', 'correct_text']].to_dict(orient='records')