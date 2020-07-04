from nlp import load_dataset
import spacy
import neuralcoref

# dataset = load_dataset('trivia_qa', 'rc')

spacy_nlp = spacy.load('en_core_web_lg')

# load NeuralCoref and add it to the pipe of SpaCy's model
coref = neuralcoref.NeuralCoref(spacy_nlp.vocab)
spacy_nlp.add_pipe(coref, name='neuralcoref')

# You're done. You can now use NeuralCoref the same way you usually manipulate a SpaCy document and it's annotations.
doc = spacy_nlp(u'My sister, Genevieve, has a dog named Hugo. She loves him.')

print(doc)
