from fst import *
import pytest
from typing import *


phoneme_inputs = [
    ("foo", "f o o", ["- - f o - o - - |", "- f - - o o - o | | - -"]),
    ("bar", "b a r", ["b b b - - - a a - r r | - - -", "- - - - - b a r |"]),
    ("baz", "b a z", ["- - - - b - - - a a - z z z z - - - - | | - - -", "b a a a z | - -"]),
]

@pytest.fixture
def sample_lexicon_fst():
    lexicon = {
        "foo": "f o o",
        "bar": "b a r",
        "baz": "b a z",
    }
    return LexicalFstFactory(lexicon=lexicon)

@pytest.mark.parametrize("word,phoneme_seq,input_strs", phoneme_inputs)
def test_phoneme_fsa(word: str, phoneme_seq: str, input_strs: List[str], sample_lexicon_fst: LexicalFstFactory):
    phoneme_fsa = sample_lexicon_fst.phoneme_fsa(phoneme_seq)
    for input_str in input_strs:
        composed_fsa = sample_lexicon_fst.fsa(input_str)@phoneme_fsa
        assert sample_lexicon_fst.fst_string(composed_fsa) == input_str

@pytest.mark.parametrize("word,phoneme_seq,input_strs", phoneme_inputs)
def test_phoneme2word_fsa(word: str, phoneme_seq: str, input_strs: List[str], sample_lexicon_fst: LexicalFstFactory):
    phoneme2word = sample_lexicon_fst.get_phonemes2word_fst(word)
    for input_str in input_strs:
        composed_fst = sample_lexicon_fst.fsa(input_str)@phoneme2word
        assert sample_lexicon_fst.fst_string(composed_fst) == word

@pytest.mark.parametrize("word,phoneme_seq,input_strs", phoneme_inputs)
def test_word2phoneme_fa(word: str, phoneme_seq: str, input_strs: List[str], sample_lexicon_fst: LexicalFstFactory):
    word2phoneme = sample_lexicon_fst.get_word2phonemes_fst(word)
    composed_fst_w_word = sample_lexicon_fst.fsa(word)@word2phoneme
    assert composed_fst_w_word.start() != pynini.NO_STATE_ID
    for input_str in input_strs:
        composed_fst_w_phonemes = composed_fst_w_word@sample_lexicon_fst.fsa(input_str)
        assert sample_lexicon_fst.fst_string(composed_fst_w_phonemes) == input_str

@pytest.mark.parametrize("word,phoneme_seq,input_strs", phoneme_inputs)
def test_word_decoder(word: str, phoneme_seq: str, input_strs: List[str], sample_lexicon_fst: LexicalFstFactory):
    word_decoder = sample_lexicon_fst.word_decoder
    for input_str in input_strs:
        composed_fst = sample_lexicon_fst.fsa(input_str)@word_decoder
        assert sample_lexicon_fst.fst_string(composed_fst) == word