from transformers import BertJapaneseTokenizer, BertForMaskedLM
import torch
import unicodedata

class SC_tokenizer(BertJapaneseTokenizer):

	def encode_plus_tagged(self, wrong_text, correct_text, max_length=128):
		'''
		符号化(wrong sentences & correct sentences)
		'''

		encoding = self(
			wrong_text,
			max_length=max_length,
			padding='max_length',
			truncation=True
			)

		encoding_correct = self(
			correct_text,
			max_length=max_length,
			padding='max_length',
			truncation=True
			)

		encoding['labels'] = encoding_correct['input_ids']

		return encoding

	def encode_plus_untagged(self, text, max_length=None, return_tensors=None):
		'''
		文章を符号化 & それぞれのトークンの位置特定
		'''

		# 文章のトークン化 & トークンと文字列を対応づける
		tokens = [] #トークン
		tokens_original = [] #トークンに対応する文字列
		words = self.word_tokenizer.tokenize(text)
		for word in words:
			tokens_word = self.subword_tokenizer.tokenize(word)
			tokens.extend(tokens_word)
			if tokens_word[0] == '[UNK]': #未知語対策
				tokens_original.append(word)
			else:
				tokens_original.extend([
					token.replace('##','') for token in tokens_word
				])

		# 各トークンの文章中での位置
		position = 0
		spans = []
		for token in tokens_original:
			l = len(token)
			while 1:
				if token != text[position:position + l]:
					position += 1
				else:
					spans.append([position, position + l])
					position += l
					break

		# 符号化
		input_ids = self.convert_tokens_to_ids(tokens)
		encoding = self.prepare_for_model(
			input_ids,
			max_length=max_length,
			padding='max_length' if max_length else False,
			truncation=True if max_length else False
			)
		sequence_length = len(encoding['input_ids'])
		# [CLS]
		spans = [[-1, -1]] + spans[:sequence_length-2]
		# [SEP][PAD]
		spans = spans + [[-1, -1]] * (sequence_length - len(spans))

		if return_tensors == 'pt':
			encoding = { k: torch.tensor([v]) for k, v in encoding.items()}

		return encoding, spans

	def convert_bert_output_to_text(self, text, labels, spans):
		'''
		文章・各トークンのラベルの予測値・文章中の位置 ----> 予測文章
		'''

		assert len(spans) == len(labels)

		# 特殊トークンを除く
		labels = [label for label, span in zip(labels, spans) if span[0]!=-1]
		spans = [span for span in spans if span[0]!=-1]

		# 文章作成
		predicted_text = ''
		position = 0
		for label, span in zip(labels, spans):
			start, end = span
			if position != start: #空白の処理
				predicted_text += text[position:start]
			predicted_token = self.convert_ids_to_tokens(label)
			predicted_token = predicted_token.replace('##', '')
			predicted_token = unicodedata.normalize(
				'NFKC', predicted_token
				)
			predicted_text += predicted_token
			position = end

		return predicted_text
