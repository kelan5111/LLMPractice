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


class SubWordTokenizer:
    def __init__(self, vocabulary=None, max_size=20000):
        if vocabulary is not None:
            self.vocabulary = vocabulary
        else:
            self.vocabulary = {0: "[UNK]"}

        self.max_size = max_size

    def __call__(self, sequence):
        tokens = self._standardize_split(sequence)
        self.byte_pair_encode(tokens)

    def byte_pair_encode(self, tokens):
        while len(self.vocabulary.values()) < self.max_size:
            pair = self._find_common_pair(tokens)

            if not pair:
                break

            merged_tokens = ''.join(pair)
            self.vocabulary[len(self.vocabulary)] = merged_tokens

            tokens = self._merge_pair(tokens, pair)

        return tokens

    @staticmethod
    def _merge_pair(tokens, common_pair):
        new_tokens = []
        for token in list(tokens):
            new_token = []
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])

                if pair == common_pair:
                    new_token.append(token[i] + token[i+1])
                else:
                    new_token.append(token[i])

            new_tokens.append(new_token)
        return new_tokens

    @staticmethod
    def _find_common_pair(tokens):
        common_pair = collections.Counter()

        for token in tokens:
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                common_pair[pair] += 1

        if not common_pair:
            return None
        return common_pair.most_common(1)[0][0]

    @staticmethod
    def _standardize_split(sequence):
        standardize_seq = []
        for word in sequence.split():
            split_word = list(word.lower())
            standardize_seq.append(split_word)
        return standardize_seq

    def _index(self, word):
        return [index for index, token in self.vocabulary.items() if word == token][0]


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


def token_count_test():
    word_tok_book, word_tok_size = word_tok_test()
    char_tok_book, char_tok_size = char_tok_test()

    print(f"Word Tokenizer input tokens: {word_tok_size}\nCharacter Tokenizer input tokens: {char_tok_size}")


def sub_word_tok_test():
    sub_word_tokenizer = SubWordTokenizer()
    sub_word_tokenizer("Hello my name, is kelan.")


def main():
    sub_word_tok_test()


if __name__ == '__main__':
    main()
