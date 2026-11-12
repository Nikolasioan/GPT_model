class BPETokenizer:
    "Byte-pair-encoder style tokenizer."

    def __init__(self):
        self.merges = []        # ordered list of pairs we merged.
        self.vocab = {}         # Dictionary: token → id
        self.inv_vocab = {}     # Dictionary: id → token:

    def _word_freqs(self, text):
        "For each word in a text make it a tuple of its characters characters and in the end put </w>. </w> is the convention for end of word. Return a dictionary "
        freqs = {}
        for w in text.split(): # Split all words in the text.
            symbols = tuple(list(w) + ["</w>"]) # For each word create a tuple with the characters and '</w>'.
            freqs[symbols] = freqs.get(symbols, 0) + 1 # Count how many times each tuple appears. This is a dictionary with tuples as keys.
        return freqs

    def _count_pairs(self, word_freqs):
        "Across the text count all pair counts of all the words."
        pairs = {}
        for symbols, f in word_freqs.items(): 
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + f
        return pairs

    def _merge(self, pair, word_freqs):
        "Takes one chosen pair and glues it into a single token everywhere it appears across all the words"
        a, b = pair # Split the pair.
        merged = {}
        for symbols, f in word_freqs.items():
            new, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new.append(a + b)     
                    i += 2
                else:
                    new.append(symbols[i])
                    i += 1
            merged[tuple(new)] = f
        return merged

    def train(self, text, num_merges): # num_merges is how many merge rules to learn.
        "Byte-pair-encoder style tokenizer algorithm."
        word_freqs = self._word_freqs(text)
        for _ in range(num_merges):
            pairs = self._count_pairs(word_freqs) # count all adjacent pairs
            if not pairs:
                break
            best = max(pairs, key=pairs.get)  # Pick the most frequent pair
            self.merges.append(best) # Append the most frequent pair.
            word_freqs = self._merge(best, word_freqs)
        self._build_vocab(text)

    def _build_vocab(self, text):
        "After training, it gathers every possible token and gives each one an integer ID."
        tokens = set()
        for w in text.split():
            tokens.update(w)               
        tokens.add("</w>")
        for a, b in self.merges:         
            tokens.add(a + b)
        self.vocab = {t: i for i, t in enumerate(sorted(tokens))}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    def encode(self, text):
        "Give a text and return a list of IDs."
        ids = []
        for w in text.split():
            symbols = list(w) + ["</w>"]
            for a, b in self.merges:       
                new, i = [], 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                        new.append(a + b)
                        i += 2
                    else:
                        new.append(symbols[i])
                        i += 1
                symbols = new
            ids.extend(self.vocab[s] for s in symbols)
        return ids

    def decode(self, ids):
        "Give a list of IDs and return a list of tokens."
        text = "".join(self.inv_vocab[i] for i in ids)
        return text.replace("</w>", " ").strip()