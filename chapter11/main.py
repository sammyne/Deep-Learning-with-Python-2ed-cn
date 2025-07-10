import string
import re

import tokenizers

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace, Sequence,Punctuation


# 加载预训练的分词器
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.enable_padding()
tokenizer.normalizer = Lowercase()
tokenizer.pre_tokenizer = Sequence(
    [
        Whitespace(),
        Punctuation(behavior='removed')
    ]
)
trainer = WordLevelTrainer(special_tokens=["[UNK]"])

data = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]

tokenizer.train_from_iterator(data, trainer=trainer, length=len(data))

# print(tokenizer.get_vocab())

test_sentence = "I write, rewrite, and still rewrite again"

encoded_sentence = tokenizer.encode(test_sentence)
print(encoded_sentence.ids)

inverse_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
print(inverse_vocab)
decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence.ids)
print(decoded_sentence)
