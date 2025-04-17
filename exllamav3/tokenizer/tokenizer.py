from __future__ import annotations
import torch
import os, json, re
from tokenizers import Tokenizer as HFTokenizer, models
from exllamav3.util import synchronized
from exllamav3.util.file import maybe_read_json
from exllamav3.models import Config
from functools import lru_cache

class Tokenizer:

    def __init__(
        self,
        config: Config,
    ):
        self.config = config

        # Defaults
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = ""
        self.newline_token = "\n"
        self.space_token = " "
        self.special_delimiters = None
        self.unspecial_delimiters = None
        self.extended_piece_to_id = {}
        self.extended_id_to_piece = {}
        self.unspecial_piece_to_id = {}
        self.unspecial_id_to_piece = {}
        self.vocab = None

        # Regex
        self.ord_exp = re.compile(r"^<0x([0-9A-Fa-f]+)>$")

        # Files
        self.path_tokenizer_json = os.path.join(self.config.directory, "tokenizer.json")
        self.path_tokenizer_config_json = os.path.join(self.config.directory, "tokenizer_config.json")
        self.path_added_tokens_json = os.path.join(self.config.directory, "added_tokens.json")
        self.tokenizer = HFTokenizer.from_file(self.path_tokenizer_json)
        self.tokenizer_config_dict = maybe_read_json(self.path_tokenizer_config_json)
        self.added_tokens_dict = maybe_read_json(self.path_added_tokens_json)

        # Disable truncation
        self.tokenizer.no_truncation()

        # Deduce placeholders used for space and newline chars in raw vocabulary
        self.space_char_ = " "
        self.newline_char_ = "\n"
        m = self.tokenizer.model
        if isinstance(m, models.BPE):
            def deduce_char_map(input_char):
                char_id = self.tokenizer.encode(input_char, add_special_tokens = False).ids[-1]
                char_str = self.tokenizer.id_to_token(char_id)
                match = self.ord_exp.match(char_str)
                if match:
                    h = match.group(1)
                    o = int(h, 16)
                    char_str = chr(o)
                else:
                    char_str = char_str[-1]
                return char_str
            self.space_char_ = deduce_char_map(" ")  # "Ġ"
            self.newline_char_ = deduce_char_map("\n")  # "Ċ"

        # Add tokens from added_tokens.json if present, assume they're all special
        self.extended_piece_to_id.update(self.added_tokens_dict)

        # Add special tokens from tokenizer_config.json
        atd = self.tokenizer_config_dict.get("added_tokens_decoder", {})
        for (k, v) in atd.items():
            if not v["special"]:
                continue
            token_id = int(k)
            token_str = v["content"]
            self.extended_piece_to_id[token_str] = token_id

        # Remove unspecial added tokens that exist in the base tokenizer already, but only if they decode correctly
        # see https://github.com/huggingface/tokenizers/issues/1392
        ok_tokens = []
        for p, i in self.unspecial_piece_to_id.items():
            try:
                itp = self.tokenizer.decode([i])
                if itp == p: ok_tokens.append(p)
            except IndexError:
                pass
        for t in ok_tokens: del self.unspecial_piece_to_id[t]

        # Invert extended dictionaries
        self.extended_id_to_piece = {v: k for k, v in self.extended_piece_to_id.items()}
        self.unspecial_id_to_piece = {v: k for k, v in self.unspecial_piece_to_id.items()}

        # Get control token IDs
        ut = self.tokenizer.model.unk_token
        self.unk_token_id = None if ut is None else self.tokenizer.token_to_id(ut)
        self.eos_token_id = self.config.eos_token_id
        self.bos_token_id = self.config.bos_token_id
        self.pad_token_id = self.config.pad_token_id

        # If model config doesn't specify BOS and EOS tokens, try to load from tokenizer config
        def get_default_token_id(config_key: str, current: int | None, default: int | None):
            if current is not None: return current
            if self.tokenizer_config_dict is not None and config_key in self.tokenizer_config_dict:
                st = self.tokenizer_config_dict[config_key]
                if st is None: return None
                if isinstance(st, dict):
                    stc: str | None = st.get("content", None)
                    if stc is None:
                        return None
                    else:
                        return self.tokenizer.token_to_id(stc)
                elif isinstance(st, str):
                    return self.tokenizer.token_to_id(st)
                else:
                    return None
            else:
                return default
        self.pad_token_id = get_default_token_id("pad_token", self.pad_token_id, None)
        self.bos_token_id = get_default_token_id("bos_token", self.bos_token_id, 1)
        self.eos_token_id = get_default_token_id("eos_token", self.eos_token_id, 2)

        # Get control token strings
        self.unk_token = self.tokenizer.model.unk_token
        self.bos_token = None if self.bos_token_id is None else \
            (self.extended_id_to_piece.get(self.bos_token_id) or self.tokenizer.id_to_token(self.bos_token_id))
        self.eos_token = None if self.eos_token_id is None else \
            (self.extended_id_to_piece.get(self.eos_token_id) or self.tokenizer.id_to_token(self.eos_token_id))

        # Use "<pad>" or BOS token as fallback for padding token
        if self.pad_token_id is None:
            pad_test = self.tokenizer.token_to_id("<pad>")
            if pad_test:
                self.pad_token_id = pad_test
            elif self.eos_token_id != self.bos_token_id:
                self.pad_token_id = self.eos_token_id
            else:
                self.pad_token_id = -1

        # Special case if <unk> and <pad> have the same ID
        if self.unk_token_id == self.pad_token_id:
            self.unk_token = self.pad_token

        # Make sure extended vocab contains control tokens, but avoid empty pieces
        if self.unk_token:
            self.extended_piece_to_id[self.unk_token] = self.unk_token_id
            self.extended_id_to_piece[self.unk_token_id] = self.unk_token
        if self.bos_token:
            self.extended_piece_to_id[self.bos_token] = self.bos_token_id
            self.extended_id_to_piece[self.bos_token_id] = self.bos_token
        if self.eos_token:
            self.extended_piece_to_id[self.eos_token] = self.eos_token_id
            self.extended_id_to_piece[self.eos_token_id] = self.eos_token

        self.actual_vocab_size = 1 + max(
            list(self.extended_id_to_piece.keys()) + \
            list(self.unspecial_id_to_piece.keys()) + \
            [self.tokenizer.get_vocab_size() - 1]
        )

        # Useful token IDs
        try:
            self.newline_token_id = self.tokenizer.encode(self.newline_token, add_special_tokens = False).ids[-1]
        except:
            self.newline_token_id = None
        try:
            self.space_token_id = self.tokenizer.encode(self.space_token, add_special_tokens = False).ids[-1]
        except:
            self.space_token_id = None

        # Dictionaries and lists
        self.id_to_ord = None
        self.id_to_piece = None
        self.id_to_piece_with_special = None
        self.piece_to_id = None
        self.get_id_to_ord_list()
        self.get_id_to_piece_list(True)
        self.get_id_to_piece_list(False)
        self.get_piece_to_id_dict()


    # Get single token

    def single_token(self, token_id: int) -> torch.Tensor:
        """
        Get single token as tensor

        :param token_id:
            Token ID

        :return:
            LongTensor of shape (1, 1) of token ID
        """
        return torch.tensor([[token_id]], dtype = torch.long)

    def single_id(self, token: str) -> int:
        """
        Get the ID of a single token from exact string match

        :param token:
            Token

        :return:
            int
        """
        tid = self.extended_piece_to_id.get(token, self.get_piece_to_id_dict().get(token))
        return tid

    # Encode string with added, unspecial tokens

    def encode_unspecial(self, text: str) -> list[int]:
        if not self.unspecial_piece_to_id:
            return self.tokenizer.encode(text, add_special_tokens = False).ids

        if self.unspecial_delimiters is None:
            self.unspecial_delimiters = re.compile(
                "(" + "|".join(map(re.escape, self.unspecial_piece_to_id.keys())) + ")")

        split = self.unspecial_delimiters.split(text)
        encoded = []

        i = 0
        while i < len(split):
            if split[i] != "": encoded += self.tokenizer.encode(split[i], add_special_tokens = False).ids
            if i + 1 < len(split): encoded += [self.unspecial_piece_to_id[split[i + 1]]]
            i += 2

        return encoded

    # Encode string with special tokens

    def encode_special(self, text: str) -> list[int]:
        # if self.special_delimiters is None:
        #     self.special_delimiters = re.compile("(" + "|".join(map(re.escape, self.extended_piece_to_id.keys())) + ")")
        #
        # split = self.special_delimiters.split(text)
        # encoded = []
        #
        # i = 0
        # while i < len(split):
        #     if split[i] != "": encoded += self.tokenizer.encode(split[i], add_special_tokens = False).ids
        #     if i + 1 < len(split): encoded += [self.extended_piece_to_id[split[i + 1]]]
        #     i += 2

        # TODO: Test if the above is actually no longer needed (was written for SentencePiece lib)
        encoded = self.tokenizer.encode(text, add_special_tokens = False).ids
        return encoded

    def encode_special_or_unspecial(
        self,
        text: str,
        special: bool,
        # embeddings: list[ExLlamaV2MMEmbedding]
    ):
        out_parts = []

        # if embeddings:
        #     aliases = {e.text_alias: e for e in embeddings}
        #     split_pattern = r"(" + "|".join(re.escape(k) for k in aliases.keys()) + ")"
        #     in_parts = re.split(split_pattern, text)
        # else:
        aliases = {}
        in_parts = [text]

        for text in in_parts:
            if text in aliases:
                out_parts += aliases[text].get_ids()
            else:
                if special:
                    out_parts += self.encode_special(text)
                else:
                    out_parts += self.encode_unspecial(text)

        return out_parts

    # Encode string or list of strings

    def encode(
        self,
        text: str | list[str],
        add_bos: bool = False,
        add_eos: bool = False,
        encode_special_tokens: bool = False,
        return_offsets: bool = False,
        # embeddings: list[ExLlamaV2MMEmbedding] | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        """
        Encode string or list of strings

        :param text:
            str or list[str]: Input text

        :param add_bos:
            Prepend BOS token before each sequence

        :param add_eos:
            Append EOS token to each sequence

        :param encode_special_tokens:
            Encode any special tokens that appear in the input as such. If False, substrings like "<bos>"
            will be encoded as text.

        :param return_offsets:
            Also return a tensor of padding lengths

        # :param embeddings:
        #     List of ExLlamaV2MMEmbeddings. If present, aliases in the input will be replaced with token ID ranges.

        :return:
            Tensor of shape (batch_size, max_seq_len), optionally offsets Tensor of shape (batch_size)
        """

        # if embeddings is None:
        embeddings = []

        if isinstance(text, list):

            # text is a list of strings

            list_ids = [self.encode_special_or_unspecial(t, encode_special_tokens) for t in text]

            if add_bos and self.bos_token_id is not None:
                for ids in list_ids: ids.insert(0, self.bos_token_id)
            if add_eos and self.eos_token_id is not None:
                for ids in list_ids: ids.append(self.eos_token_id)

            max_length = max([len(ids) for ids in list_ids])

            padded_ids = []
            offsets = []
            for ids in list_ids:
                padding_length = max_length - len(ids)
                padding = torch.full((padding_length,), self.pad_token_id)
                padded_ids.append(torch.cat((padding, torch.tensor(ids)), dim=0))
                offsets.append(-padding_length)

            stacked_ids = torch.stack(padded_ids, dim=0)

            if return_offsets:
                return stacked_ids, torch.tensor(offsets, dtype=torch.int)
            else:
                return stacked_ids

        else:

            # text is a single string

            # ids = self.encode_special(text) if encode_special_tokens else self.encode_unspecial(text)
            ids = self.encode_special_or_unspecial(text, encode_special_tokens)
            if add_bos and self.bos_token_id is not None:
                ids.insert(0, self.bos_token_id)
            if add_eos and self.eos_token_id is not None:
                ids.append(self.eos_token_id)

            ids = torch.tensor(ids).to(torch.long).unsqueeze(0)
            if return_offsets:
                return ids, torch.tensor([0], dtype=torch.int)
            else:
                return ids

    # Decode sequence with added, unspecial tokens

    def decode_unspecial(self, seq):

        if not self.unspecial_id_to_piece:
            return self.tokenizer.decode(seq)

        text = ""
        start = 0
        end = 0
        while end < len(seq):
            if seq[end] in self.unspecial_id_to_piece:
                if end > start: text += self.tokenizer.decode(seq[start: end])
                text += self.unspecial_id_to_piece[seq[end]]
                end += 1
                start = end
            else:
                end += 1
        if end > start: text += self.tokenizer.decode(seq[start: end])
        return text

    # Decode sequence with or without special tokens

    def decode_(self, seq, decode_special_tokens):

        if not decode_special_tokens:

            max_token = self.tokenizer.get_vocab_size()
            seq = [t for t in seq if (t != self.pad_token_id and t < max_token and t != self.eos_token_id)]
            if self.eos_token_id in seq: seq = seq[:seq.index(self.eos_token_id)]
            return self.decode_unspecial(seq)

        else:

            max_token = self.tokenizer.get_vocab_size()
            seq = [t for t in seq if (t != self.pad_token_id and t < max_token)]
            text = ""
            start = 0
            end = 0
            while end < len(seq):
                if seq[end] in self.extended_id_to_piece:
                    if end > start: text += self.tokenizer.decode(seq[start: end])
                    text += self.extended_id_to_piece[seq[end]]
                    end += 1
                    start = end
                else:
                    end += 1
            if end > start: text += self.decode_unspecial(seq[start: end])

        return text

    # Decode IDs, or a list of IDs

    def decode(
        self,
        ids: torch.Tensor | list[torch.Tensor],
        decode_special_tokens: bool = False
    ) -> str | list[str]:
        """
        Decode IDs or list of IDs

        :param ids:
            Tensor of ids, shape (batch_size, max_seq_len)

        :param decode_special_tokens:
            Decode special tokens in the input to their text representation. If False, tokens such as
            "<bos>" will become empty strings in the output.

        :return:
            str if batch_size == 1, else list[str]
        """

        if isinstance(ids, list):

            texts = []
            for i in ids:
                texts.append(self.decode(i, decode_special_tokens))
            return texts

        assert isinstance(ids, torch.Tensor), "ids must be Tensor"

        if ids.dim() > 1:

            texts = []
            for i in range(ids.shape[0]):
                seq = ids[i].tolist()
                texts.append(self.decode_(seq, decode_special_tokens))
            return texts

        else:

            ids = ids.tolist()
            text = self.decode_(ids, decode_special_tokens)
            return text

    # Create padding mask

    def padding_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask corresponding to IDs tensor

        :param ids:
            Token IDs

        :return:
            Additive bias Tensor of -inf where ids == pad_token_id, 0 elsewhere
        """

        mask_bool = (ids == self.pad_token_id)
        mask = mask_bool.int()
        mask *= -65505 * 2
        mask = mask.half()
        return mask

    def num_tokens(self, text):
        """
        Measure tokenized length of text

        :param text:
            Input text

        :return:
            Tokenized length of text
        """

        ids = self.tokenizer.encode(text)
        return len(ids)

    # Get ordinals of single-byte tokens

    @synchronized
    def get_id_to_ord_list(self):
        if self.id_to_ord is not None: return self.id_to_ord

        self.id_to_ord = []
        for idx in range(self.tokenizer.get_vocab_size()):
            p = self.tokenizer.id_to_token(idx)
            if not p: p = ""
            self.id_to_ord.append(self.tokenizer.token_to_id(p))

        def clean_special_chars(p):
            p = p.replace(self.space_char_, " ")
            p = p.replace(self.newline_char_, "\n")
            return p

        def piece_to_ord(p):
            match = self.ord_exp.match(p)
            if match:
                h = match.group(1)
                return int(h, 16)
            if len(p) == 1:
                p = clean_special_chars(p)
                o = ord(p)
                if o <= 255: return o
            return -1

        i = self.tokenizer.get_vocab_size()
        while True:
            if i in self.extended_id_to_piece:
                self.id_to_ord.append(piece_to_ord(self.extended_id_to_piece[i]))
            elif i in self.unspecial_id_to_piece:
                self.id_to_ord.append(piece_to_ord(self.unspecial_id_to_piece[i]))
            elif i < self.actual_vocab_size:
                self.id_to_ord.append(-1)
            else:
                break
            i += 1

        return self.id_to_ord

    # Copy vocabulary from model

    @synchronized
    def get_id_to_piece_list(self, include_special_tokens = False):

        def enumerate_tokens():
            if self.vocab is not None: return enumerate(self.vocab)
            self.vocab = []

            test_enc = self.tokenizer.encode(" t", add_special_tokens=False)
            test_count = len(test_enc.ids)
            assert test_count > 0, "Tokenizer error, test string encodes to zero tokens"
            test_id = test_enc.ids[0]
            test_piece = self.tokenizer.decode([test_id])

            if test_count == 1 and len(test_piece) == len(" t"):
                for i in range(self.tokenizer.get_vocab_size()):
                    d = self.tokenizer.decode([i])
                    self.vocab.append(d)
            else:
                prefix_id = self.tokenizer.encode(" ", add_special_tokens=False).ids[0]
                prefix_piece = self.tokenizer.decode([prefix_id])
                prefix_len = len(prefix_piece)
                for i in range(self.tokenizer.get_vocab_size()):
                    dt = self.tokenizer.decode([prefix_id, i])
                    d = dt[prefix_len:]
                    self.vocab.append(d)

            return enumerate(self.vocab)

        if include_special_tokens:
            if self.id_to_piece_with_special is not None: return self.id_to_piece_with_special

            id_to_piece_extended = self.get_id_to_piece_list().copy()
            for k, v in self.extended_id_to_piece.items():
                id_to_piece_extended[k] = v

            self.id_to_piece_with_special = id_to_piece_extended
            return self.id_to_piece_with_special

        if self.id_to_piece is not None: return self.id_to_piece

        self.id_to_piece = [""] * self.tokenizer.get_vocab_size()
        for idx, p in enumerate_tokens():
            self.id_to_piece[idx] = p

        i = self.tokenizer.get_vocab_size()
        while True:
            if i in self.extended_id_to_piece:
                self.id_to_piece.append(self.extended_id_to_piece[i])
            elif i in self.unspecial_id_to_piece:
                self.id_to_piece.append(self.unspecial_id_to_piece[i])
            elif i < self.actual_vocab_size:
                self.id_to_piece.append("��_undefined_token_��")
            else:
                break
            i += 1

        return self.id_to_piece

    @synchronized
    def get_piece_to_id_dict(self):
        if self.piece_to_id is not None: return self.piece_to_id
        all_pieces = self.get_id_to_piece_list()
        self.piece_to_id = {piece: idx for idx, piece in enumerate(all_pieces)}
        return self.piece_to_id

    @staticmethod
    def from_config(config: Config):
        return Tokenizer(config)

    @lru_cache(1000)
    def get_tokens_with_prefix_string(self, prefix: str):
        """
        Return list of token IDs for pieces that start with the given prefix
        """
        id_to_piece = self.get_id_to_piece_list()
        tokens = [idx for idx, piece in enumerate(id_to_piece) if piece.startswith(prefix)]
        return tokens

    @lru_cache(1000)
    def get_tokens_with_prefix_id(self, prefix_id: int):
        """
        Return list of token IDs for pieces that start with the given prefix
        """
        id_to_piece = self.get_id_to_piece_list()
        prefix = id_to_piece[prefix_id]
        return self.get_tokens_with_prefix_string(prefix)
