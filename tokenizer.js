// Simple BERT tokenizer for MiniLM
// Loads vocab + applies WordPiece tokenization.

class BertTokenizer {
    constructor(vocab) {
        this.vocab = vocab;
        this.inv = Object.fromEntries(
            Object.entries(vocab).map(([k, v]) => [v, k])
        );
    }

    tokenize(text) {
        text = text.toLowerCase().replace(/[^\w\s]/g, "");
        const tokens = text.split(/\s+/);

        let out = [];
        for (let word of tokens) {
            if (this.vocab[word]) {
                out.push(word);
                continue;
            }

            // WordPiece splitting
            let chars = word.split("");
            let sub = [];
            let start = 0;

            while (start < chars.length) {
                let end = chars.length;
                let found = null;

                while (start < end) {
                    let piece = chars.slice(start, end).join("");
                    if (start > 0) piece = "##" + piece;

                    if (this.vocab[piece]) {
                        found = piece;
                        break;
                    }
                    end--;
                }

                if (!found) {
                    sub.push("[UNK]");
                    break;
                }

                sub.push(found);
                start = end;
            }
            out.push(...sub);
        }
        return out;
    }

    encode(text, maxLen = 128) {
        let tokens = ["[CLS]", ...this.tokenize(text), "[SEP]"];

        let ids = tokens.map(t => this.vocab[t] ?? this.vocab["[UNK]"]);
        let mask = ids.map(() => 1);

        while (ids.length < maxLen) {
            ids.push(0);
            mask.push(0);
        }

        return { ids: new BigInt64Array(ids.map(x => BigInt(x))), mask };
    }
}
