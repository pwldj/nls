import sentencepiece as spm
import os

dir = 'raw_data'

files = os.listdir(dir)
files = ','.join([dir+'/'+x for x in files])
spm.SentencePieceTrainer.train(input='raw_data/0_wiki_en.txt', model_prefix='30k_spm', vocab_size=32000,
                               input_sentence_size=3000000,shuffle_input_sentence=True,
                               unk_piece='[UNK]', pad_piece='[PAD]', user_defined_symbols=['[CLS]', '[SEP]', '[MASK]'])
