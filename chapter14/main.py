from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import PreTokenizer

#class MiniPreTokenizer(PreTokenizer):
class MiniPreTokenizer:
  def __init__(self):
    pass

  def pre_tokenize(self, pretok):
    return pretok.split(lambda s: self.tokenize(s))


  def pre_tokenize_str(self, sequence):
    return [self.tokenize(v) for v in sequence]

  def tokenize(self, text):
    return "".join(filter(str.isalnum, text.lower())).split()


t = MiniPreTokenizer()

# # 加载预训练的分词器
# tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
# tokenizer.pre_tokenizer = MiniPreTokenizer()
