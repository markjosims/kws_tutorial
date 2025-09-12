import pynini
import numpy as np
import graphviz
import pydot
from typing import *
import torch
import os

class FstFactory():
    """
    Provides convenient wrappers for saving, decoding and visualizing WFSTs
    in Pynini using a set SymbolTable
    """

    def __init__(self, symbols: pynini.SymbolTable):
        self.symbols = symbols

    def set_symbols(self, f: pynini.Fst) -> pynini.Fst:
        """
        Set input and output symbols for a FST `f` to the
        user-defined symbol table.
        """
        f=f.set_input_symbols(self.symbols)
        f=f.set_output_symbols(self.symbols)
        return f

    def fsa(
            self,
            acceptor_str: Union[str, List[str]],
            weight: Optional[pynini.WeightLike]=None
        ):
            """
            Create a Finite State Acceptor of the given string using
            the symbols table.
            """
            if type(acceptor_str) is list:
                acceptor_str = ' '.join(acceptor_str)
            f=pynini.accep(acceptor_str, weight=weight, token_type=self.symbols)
            f=self.set_symbols(f)
            f=f.optimize()
            return f

    def print_fst(self, f):
        tmp_path = 'tmp/tmp.dot'
        f.draw(tmp_path, portrait=True)
        with open(tmp_path) as file:
            return graphviz.Source(file.read())
        
    def draw_svg(self, fst: pynini.Fst, filepath: str = 'tmp/tmp.svg', title: Optional[str]=None):
        """
        Saves .dot and .svg representations of `fst`, with an optionally specified `title`
        (defaults to `filepath`).
        """
        stem = os.path.splitext(filepath)[0]
        fst = self.set_symbols(fst)
        dotfile = stem+'.dot'
        fst.draw(
            source=dotfile,
            show_weight_one=True,
            isymbols=fst.input_symbols(),
            osymbols=fst.output_symbols(),
            portrait=True,
            title=title or stem,
        )
        graph = pydot.graph_from_dot_file(dotfile)[0]
        graph.write_svg(filepath)
        
class LexicalFstFactory(FstFactory):
    """
    Subclass of `FstFactory` that takes a dict mapping words
    to phonemes on init and constructs a symbol table.
    Provides helpers for creating WFSTs mapping between words
    and phonemes.
    """
    def __init__(
            self,
            lexicon: Dict[str, Union[str, List[str]]],
            phonemes: Optional[List[str]]=None,
            word_boundary: str = '|',
            blank: str = '-',
        ):
        """
        Arguments:
            lexicon:    dict of shape `{word: phoneme_seq}`, where `word` is a str
                        and `phoneme_seq` is a str of shape `p1 p2 p3...` or a list
                        of strs ['p1', 'p2', 'p3']
        """
        make_phonemes = False
        if phonemes is None:
            make_phonemes = True
            phonemes = set()
        words = set()
        for word, phoneme_seq in lexicon.items():
            words.add(word)
            if not make_phonemes:
                continue
            if type(phoneme_seq) is str:
                phonemes.update(phoneme_seq.split())
            else:
                phonemes.update(phoneme_seq)

        phonemes = list(phonemes)
        words = list(words)

        symbols = pynini.SymbolTable()
        symbols.add_symbol('<eps>')
        for word in words:
            symbols.add_symbol(word)
        for phoneme in phonemes:
            symbols.add_symbol(phoneme)
        if word_boundary not in phonemes:
            symbols.add_symbol(word_boundary)
        if blank not in phonemes:
            symbols.add_symbol(blank)

        self.blank = blank
        self.word_boundary = word_boundary
        self.lexicon = lexicon
        self.symbols = symbols
        self.phonemes = phonemes
        self.words = words

        self.words2phonemes = self._get_allwords2phonemes()
        self.phonemes2words = pynini.invert(self.words2phonemes)
        self.word_decoder = self._get_word_decoder_fst()
        self.phone_encoder = self._get_phone_encoder_fst()

    def make_phone_probabilities(
            self,
            cost_matrix: np.ndarray,
        ) -> pynini.Fst:
        """
        Convert a `cost_matrix` of probabilities over the alphabet of phonemes into a WFST.
        """
        phoneme_probs = self.fsa('')
        for timestep_vector in cost_matrix:
            timestep_probs = []
            for i, phoneme in enumerate(self.phonemes):
                phoneme_cost = timestep_vector[i]
                phoneme_prob_fsa = self.fsa(phoneme, weight=phoneme_cost)
                timestep_probs.append(phoneme_prob_fsa)
            timestep_probs = pynini.union(*timestep_probs)
            phoneme_probs = phoneme_probs + timestep_probs
        return phoneme_probs.optimize()
    
    def decode_phone_probabilities(self, cost_matrix: np.ndarray) -> str:
        phone_prob_fsa = self.make_phone_probabilities(cost_matrix)
        phone_prob_composed = phone_prob_fsa@self.word_decoder
        return self.get_top_string(phone_prob_composed)

    def get_top_string(self, f: pynini.Fst) -> str:
        shortest_path = pynini.shortestpath(f)
        return self.fst_string(shortest_path)

    def fst_string(self, f) -> str:
        return f.string(token_type=self.symbols)
    
    def phoneme_fsa(
            self,
            phoneme_seq: Union[str, List[str]],
            weight: Optional[pynini.WeightLike] = None,
        ):
        phoneme_fsa = self.fsa(self.blank, weight=weight).closure()
        if type(phoneme_seq) is str:
            phoneme_seq = phoneme_seq.split()
        for phoneme in phoneme_seq:
            phoneme_fsa += self.fsa(phoneme).plus + self.fsa(self.blank).closure()
        phoneme_fsa += self.fsa(self.word_boundary).plus + self.fsa(self.blank).closure()
        phoneme_fsa.optimize()
        return phoneme_fsa
    
    def get_word2phonemes_fst(self, word: str, weight: Optional[pynini.WeightLike]=None):
        phonemes = self.lexicon[word]
        phoneme_fsa = self.phoneme_fsa(phonemes, weight)
        word_fsa = self.fsa(word)
        word2phonemes_fst = pynini.cross(word_fsa, phoneme_fsa)
        return word2phonemes_fst

    def get_phonemes2word_fst(self, word: str, weight: Optional[pynini.WeightLike]=None):
        word2phonemes_fst = self.get_word2phonemes_fst(word, weight)
        phonemes2word_fst = pynini.invert(word2phonemes_fst)
        return phonemes2word_fst
    
    def _get_allwords2phonemes(self):
        allwords2phonemes = self.fsa('')
        for word in self.lexicon.keys():
            allwords2phonemes = allwords2phonemes | self.get_word2phonemes_fst(word)
        allwords2phonemes.optimize()
        return allwords2phonemes
    
    def _get_word_decoder_fst(self):
        return pynini.closure(self.phonemes2words).optimize()

    def _get_phone_encoder_fst(self):
        return pynini.closure(self.words2phonemes).optimize()
    
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
            emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
        returns:
            str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1) # [num_seq, ]
        indices_deduplicated = torch.unique_consecutive(indices, dim=-1)
        indices_noblanks = [i for i in indices_deduplicated if i != self.blank]
        decoded_str = "".join(self.labels[i] for i in indices_noblanks)
        return decoded_str

class WFSTDecoder(torch.nn.Module):
    def __init__(
            self,
            labels,
            lexicon: Union[List[str], Dict[str, Union[str, List[str]]]],
            blank=0,
            word_boundary=1,
        ):
        super().__init__()
        if type(lexicon) is list:
            lexicon = WFSTDecoder._lexicon_to_dict(lexicon)

        self.labels = labels
        self.blank = blank
        self.lexicon = lexicon
        self.fst_factory = LexicalFstFactory(
            lexicon=lexicon,
            phonemes=labels,
            blank=labels[blank],
            word_boundary=labels[word_boundary],
        )

    def _lexicon_to_dict(lexicon: List[str]):
        return {word: list(word) for word in lexicon}
    
    def prepare_logits_for_wfst(self, logits: torch.Tensor):
        probs = logits.softmax(dim=-1)
        inverse_probs = 1-probs
        return inverse_probs.numpy()

    def forward(self, emission: torch.Tensor) -> str:
        return self.fst_factory.decode_phone_probabilities(self.prepare_logits_for_wfst(emission))