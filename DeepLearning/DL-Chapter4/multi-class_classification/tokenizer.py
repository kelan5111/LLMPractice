import string


class Tokenizer:
    def __init__(self):
        self.word_index = {}
        self.tokens = []

        self.input_token = 0

    def __call__(self, sequence):
        self.build_vocab(sequence)
        self.tokenize(sequence)
        self.input_token = len(self.tokens)
        return self.tokens

    def tokenize(self, sequence):
        self.tokens.clear()

        for word in sequence.lower().split():
            punc_check = [letter for letter in word if letter in string.punctuation]
            if punc_check:
                clean_word = ''.join(letter for letter in word if letter not in string.punctuation)
                token_word = next((token for token, value in self.word_index.items() if clean_word == value), None)
                token_punc = next((token for token, value in self.word_index.items() if ''.join(punc_check) == value),
                                  None)

                if token_word is not None and token_punc is not None:
                    self.tokens.append(token_word)
                    self.tokens.append(token_punc)
                else:
                    raise ValueError(f"Word {word} does not exist in vocab.")
            else:
                token = next((token for token, value in self.word_index.items() if word == value), None)
                if token is not None:
                    self.tokens.append(token)
                else:
                    raise ValueError(f"Word {word} does not exist in vocab.")

    def build_vocab(self, tokens):
        for token in tokens.lower().split():
            # Check any punct that are joined to the word itself
            punc = [letter for letter in token if letter in string.punctuation]
            if punc:
                clean_token = "".join(letter for letter in token if letter not in string.punctuation)
                if clean_token not in self.word_index.values():
                    self.word_index[len(self.word_index)] = clean_token
                punc_str = "".join(punc)
                if punc_str not in self.word_index.values():
                    self.word_index[len(self.word_index)] = punc_str
            else:
                if token not in self.word_index.values():
                    self.word_index[len(self.word_index)] = token

    @property
    def input_tokens(self):
        return self.input_token

    @property
    def num_words(self):
        return len(self.word_index)

    def get_word_index(self):
        return self.word_index
