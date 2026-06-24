import re
import PyPDF2 as pdf
import collections


class WordTokenizer:
    def __init__(self, vocabulary=None, max_size=20000):
        if vocabulary is None:
            self.vocabulary = {0: '[UNK]'}
        else:
            self.vocabulary = vocabulary

        self.max_size = max_size
        self.unk_id = ['[UNK]']

    def __call__(self, sequence):
        self.build_vocab(sequence)
        tokenized_sequence = self.tokenize(sequence)
        return tokenized_sequence

    def build_vocab(self, sequence, max_size=20000):
        tokens = self._standardize(self._split(sequence))
        word_counts = collections.Counter()
        assert len(self.vocabulary) <= self.max_size, "Vocabulary exceeds max size!"

        word_counts.update(tokens)

        most_common = word_counts.most_common(max(0, max_size - len(self.vocabulary)))
        for token, count in most_common:
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


class CharacterTokenizer:
    def __init__(self, vocabulary=None, max_size=20000):
        if vocabulary is not None:
            self.vocabulary = {0: "[UNK]"}
        else:
            self.vocabulary = {}

        self.max_size = max_size
        self.unk_id = '[UNK]'

    def __call__(self, sequence):
        self.build_vocab(sequence)
        tokenized_seq = self.tokenize(sequence)
        return tokenized_seq

    def build_vocab(self, sequence):
        tokens = self._standardize(self._split(sequence))

        assert len(self.vocabulary) <= self.max_size, "Vocabulary is too full."

        for token in tokens:
            if token not in self.vocabulary.values():
                index = len(self.vocabulary)
                self.vocabulary[index] = token

    def tokenize(self, sequence):
        tokens = self._standardize(self._split(sequence))
        tokenized_seq = []
        for token in tokens:
            token_id = self._index(token)
            tokenized_seq.append(token_id)
        return tokenized_seq

    @staticmethod
    def _split(sequence):
        pattern = r'\S'
        return re.findall(pattern, sequence)

    @staticmethod
    def _standardize(sequence):
        return [word.lower() for word in sequence]

    def _index(self, token):
        return [index for index, t in self.vocabulary.items() if token == t][0]


# Loading Romeo and Juliet pdf
romeo_juliet_text = pdf.PdfReader("book_pdf/romeo-and-juliet.pdf")


def word_tok_test():
    word_tokenizer = WordTokenizer()

    tokenized_book = []
    num_tokens = 0

    for i in range(len(romeo_juliet_text.pages)):
        page = romeo_juliet_text.pages[i].extract_text()
        tokenized_page = word_tokenizer(page)

        tokenized_book.append(tokenized_page)
        num_tokens += len(tokenized_page)

    print(word_tokenizer.vocabulary.values())

    return tokenized_book, num_tokens


def char_tok_test():
    character_tokenizer = CharacterTokenizer()

    tokenized_book = []
    num_tokens = 0

    for i in range(len(romeo_juliet_text.pages)):
        page = romeo_juliet_text.pages[i].extract_text()
        tokenized_page = character_tokenizer(page)

        tokenized_book.append(tokenized_page)
        num_tokens += len(tokenized_page)

    return tokenized_book, num_tokens


word_tok_book, word_tok_size = word_tok_test()
char_tok_book, char_tok_size = char_tok_test()

print(f"Word Tokenizer input tokens: {word_tok_size}\nCharacter Tokenizer input tokens: {char_tok_size}")

print(word_tok_book)
