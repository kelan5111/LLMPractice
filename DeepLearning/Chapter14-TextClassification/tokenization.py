import re


class WordTokenizer:
    def __init__(self, vocabulary=None, max_size=20000):
        if vocabulary is None:
            self.vocabulary = {0: '[UNK]'}
        else:
            self.vocabulary = vocabulary

        self.max_size = max_size
        self.unk_id = ['|UNK|']

    def __call__(self, sequence):
        self.build_vocab(sequence)
        tokenized_sequence = self.tokenize(sequence)
        return tokenized_sequence

    def build_vocab(self, sequence, max_size=20000):
        tokens = self._standardize(self._split(sequence))

        assert len(self.vocabulary) <= self.max_size, "Vocabulary exceeds max size!"

        for token in tokens:
            if token not in self.vocabulary.values():
                self.vocabulary[len(self.vocabulary)] = token

    def tokenize(self, sequence):
        tokenized_sequence = []
        split_sequence = self._standardize(self._split(sequence))
        for word in split_sequence:
            token = self._index(word)
            tokenized_sequence.append(token)
        return tokenized_sequence

    @staticmethod
    def _split(sequence):
        pattern = r'\w+|[!,.,"]'
        return re.findall(pattern, sequence)

    @staticmethod
    def _standardize(sequence):
        return [word.lower() for word in sequence]

    def _index(self, word):
        return [index for index, token in self.vocabulary.items() if word == token][0]


word_tokenizer = WordTokenizer()
print(word_tokenizer("Hello my friends, my name is Kelan's"))