#-----------------------------------------------------
# Baseline model for medical embeddings using Word2Vec
# Hathaitorn Rojnirun (hr346@cornell.edu)
# Oluseye Bankole (ob97@cornell.edu) 
#-----------------------------------------------------
from gensim.models import Word2Vec
import argparse
import json
import os.path

def convert_phrase_to_word(phrase):
    return phrase.replace(' ', '_')

#-----------------------------------------------------
# Returns a list of triplets in the following format:
# (pharase_1, phrase_2, occurrences)
#-----------------------------------------------------
def load_data(path):
    print('Loading data from %s' % path)
    lines = [line.strip() for line in open(path)]

    print('Number of pairs', len(lines))
    
    data = []
    for line in lines:
        phrase_1, phrase_2, occurrences = line.split('\t')
        data.append((phrase_1, phrase_2, occurrences))
    return data

class PairIterator(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = load_data(self.data_path)
 
    def __iter__(self):
        for i, triplet in enumerate(self.data):
            if i % 10000 == 0:
                print('Processing the %dth word pair.' % i)

            occurrences = int(triplet[2])
            sentence = [convert_phrase_to_word(triplet[0]), convert_phrase_to_word(triplet[1])]
            yield sentence * occurrences

#------------------------------------------------------
# Get the Word2Vec model.
# If the model is already on the disk, load it.
# Otherwise, train the model on the input data.
#------------------------------------------------------
def get_word2vec_model(data_path):
    word2vec_model_file_name = 'word2vec_baseline_using_iterator.bin'
    if os.path.isfile(word2vec_model_file_name):
        print('Word2Vec model already exists. Loading the saved version.')
        return Word2Vec.load(word2vec_model_file_name)
    else:
        print('Generating Word2Vec model.')
        pair_iterator = PairIterator(data_path)
        model = Word2Vec(pair_iterator, size=300, window=2, min_count=1, workers=8)
        model.save(word2vec_model_file_name)
        print('Model saved to %s.' % word2vec_model_file_name)
        return model

def main(options):
    data_path = '../data/finale.txt'
    word2vec = get_word2vec_model(data_path)
    
    if options.similarity:
        concept_1, concept_2 = options.similarity
        similarity = word2vec.wv.similarity(concept_1, concept_2)
        print('Similarity value between %s and %s is %f.' % (concept_1, concept_2, similarity))
    
    if options.search:
        query = options.search[0]
        print('query = ', query)
        similar_words = word2vec.wv.most_similar([query], topn=10)
        print('Medical concepts most similar to "%s" are' % query, similar_words)

# Options
# =======
# --similarity <concept1> <concept2> - find the similarity value between two medical concepts
# --search <concept> - look up modical concepts that are most similar to the input word

# Example
# =======
# python v1.py --search difficulty_moving --similarity ibuprofen penicillin
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--similarity', default=['tobacco', 'burning_sensation'], nargs=2)
    parser.add_argument('--search', default=['tobacco'], nargs=1)
    options = parser.parse_args()
    print(json.dumps(options.__dict__, sort_keys=True, indent=4))
    
    main(options)