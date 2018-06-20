import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for i in range(test_set.num_items):
      best_prob, best_guess_word = None, None
      i_word_probabilities = {}
      i_sequences, i_lengths = test_set.get_item_Xlengths(i)
      for word, model in models.items():
          try:
              i_word_probabilities[word] = model.score(i_sequences, i_lengths)
          except Exception as e:
              i_word_probabilities[word] = float("-inf")
          if(best_prob == None or i_word_probabilities[word] > best_prob):
              best_prob = i_word_probabilities[word]
              best_guess_word = word
          continue
      probabilities.append(i_word_probabilities)
      guesses.append(best_guess_word)
    return probabilities, guesses
