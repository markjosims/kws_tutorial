"""
Given a built split of LibriPhrase, make the following files:
- ${LibriSpeech_split}_sentences.csv
    - Columns: sentence_index,file,speaker,text
- ${LibriPhrase_split}_keywords.csv
    - Columns: word_index,file,speaker,text,class
    - Words are union of all unique anchors and comparisons from LibriPhrase

"""

import os
import pandas as pd
from tqdm import tqdm
from typing import *

tqdm.pandas()

# input files
DATASETS = os.environ['DATASETS']

LIBRISPEECH_ROOT = os.path.join(DATASETS, "LibriSpeech")
LIBRISPEECH_SPLIT = "train-clean-100"
LIBRISPEECH_AUDIO = os.path.join(LIBRISPEECH_ROOT, LIBRISPEECH_SPLIT)

LIBRIPHRASE_ROOT = os.path.join(DATASETS, 'LibriPhrase')
LIBRIPHRASE_DATA = os.path.join(LIBRIPHRASE_ROOT, 'data')
LIBRIPHRASE_SPLIT = "libriphrase_diffspk_all"
LIBRIPHRASE_AUDIO = os.path.join(LIBRIPHRASE_DATA, LIBRISPEECH_SPLIT)

# output files
SENTENCES_CSV = os.path.join(LIBRIPHRASE_DATA, f"{LIBRISPEECH_SPLIT}_sentences.csv")
WORDS_CSV = os.path.join(LIBRIPHRASE_DATA, f"{LIBRIPHRASE_SPLIT}_keywords.csv")

def get_unique_keywords(keyword_df: Optional[pd.DataFrame]=None) -> List[str]:
    if keyword_df is None:
        keyword_df = pd.read_csv(WORDS_CSV)
    return list(keyword_df['keyword'].unique())

def get_keyword_tokens(
        keyword: str,
        keyword_df: Optional[pd.DataFrame]=None
    ) -> pd.DataFrame:
    """
    Arguments:
        keyword:    String representing keyword/phrase to query
        keyword_df: Optional Pandas DataFrame containing keyword data.
                    If not passed, load from $WORDS_CSV file.
    Returns:
        pandas.DataFrame, subset of `keyword_df` corresponding to the given keyword
    """
    if keyword_df is None:
        keyword_df = pd.read_csv(WORDS_CSV)
    keyword_mask = keyword_df['keyword']==keyword
    return keyword_df[keyword_mask]

def get_random_keyword_token(
        keyword: str,
        keyword_df: Optional[pd.DataFrame]=None,
    ) -> pd.Series:
    """
    Arguments:
        keyword:    String representing keyword/phrase to query
        keyword_df: Optional Pandas DataFrame containing keyword data.
                    If not passed, load from $WORDS_CSV file.
    Returns:
        pandas.Series, randomly sampled row of `keyword_df`
        corresponding to the given keyword
    """
    keyword_tokens = get_keyword_tokens(keyword, keyword_df)
    return keyword_tokens.sample().iloc[0]
    
def get_keyword_sentences(
        keyword: str,
        sentence_df: Optional[pd.DataFrame]=None,
    ) -> pd.DataFrame:
    """
    Arguments:
        keyword:        String representing keyword/phrase to query
        sentences_df:   Optional Pandas DataFrame containing sentence data.
                        If not passed, load from $SENTENCES_CSV file.
    Returns:
        pandas.DataFrame, subset of `sentences_df` containing to the given keyword
    """
    if sentence_df is None:
        sentence_df = pd.read_csv(SENTENCES_CSV)
    keyword_mask = sentence_df['sentence'].str.contains(keyword)
    return sentence_df[keyword_mask]

def get_random_keyword_sentence_pair(
        keyword: str,
        keyword_df: Optional[pd.DataFrame]=None,
        sentences_df: Optional[pd.DataFrame]=None,
        same_source: bool=False,
        same_speaker: bool=False,
    ) -> Tuple[pd.Series, pd.Series]:
    """
    Arguments:
        keyword:        String representing keyword/phrase to query
        keyword_df:     Optional Pandas DataFrame containing keyword data.
                        If not passed, load from $WORDS_CSV file.
        sentences_df:   Optional Pandas DataFrame containing sentence data.
                        If not passed, load from $SENTENCES_CSV file.
        same_source:    bool indicating whether the keyword token should
                        originate from the same audio as the sentence.
        same_speaker:   bool indicating whether the keyword token should
                        originate from the same speaker as the sentence.

    Returns:
        (keyword_row, sentence_row), randomly sampled row of `keyword_df`
        corresponding to the given keyword
    """
    keyword_row = get_random_keyword_token(keyword, keyword_df)
    keyword_sentences = get_keyword_sentences(keyword, sentences_df)
    source_mask = keyword_sentences['file']==keyword_row['sentence_file']
    speaker_mask = keyword_sentences['speaker']==keyword_row['speaker']
    if same_source:
        return keyword_row, keyword_sentences[source_mask].iloc[0]
    if same_speaker:
        return keyword_row, keyword_sentences[speaker_mask].sample().iloc[0]
    return keyword_row, keyword_sentences[~speaker_mask].sample().iloc[0]