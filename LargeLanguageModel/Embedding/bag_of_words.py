def bag_of_words(s, vocab):
    counts = dict.fromkeys(vocab, 0)
    s = remove_punc(s).lower()

    for word in s.split():
        counts[word] += 1

    return counts


def remove_punc(s):
    punc = ['!', ',', '.', '£', '"', '(', ')']
    new_s = ""
    for char in s:
        if char not in punc:
            new_s += char

    return "".join(new_s)


def build_vocab(docs):
    vocab = []
    for sen in docs:
        sen = remove_punc(sen).lower()

        for word in sen.split():
            if word not in vocab:
                vocab.append(word)

    return vocab


def to_vector(counts):
    try:
        return list(counts.values())

    except TypeError:
        print("Input not a dict.")

##################################
# 1. Each document through BoW is given a
# digital fingerprint which can be used to compare mathematically.
# 2. Each vector represents a straight line which can be measure for similarity
# using cosine which outputs 0-1 for BoW:
#             0 = no shared words (perpendicular, unrelated)
#             1 = identical words (same angle, very similar)
########################################################################################


def main():
    doc = [
        "The dog walked over the bridge, and barked at a cat!",
        "The cat slept on the mat all afternoon.",
        "A dog and a cat became unlikely friends.",
        "The cat watched the dog chase a squirrel.",
        "Dogs bark, cats purr, and birds sing."
    ]
    vocab = build_vocab(doc)
    for d in doc:
        vect = to_vector(bag_of_words(d, vocab))
        print(vect)


if __name__ == '__main__':
    main()
