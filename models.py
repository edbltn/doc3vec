import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

class TaggedDocumentIterator(object):
    
    def __init__(self, articles):
        self.articles = articles
        
    def __iter__(self):
        for a in self.articles:
            yield TaggedDocument(a[-1], [a[0]])
            
class Doc3Vec(object):
    
    assert gensim.models.doc2vec.FAST_VERSION > -1
    
    def __init__(self,
                corpus,
                epochs,
                min_count,
                vector_size,
                window=5,
                negative=5,
                compute_loss=False,
                dm=0,
                workers=5):
        
        self.corpus = corpus
        self.epochs = epochs
        self.min_count = min_count
        self.vector_size = vector_size
        self.window = window
        self.negative = negative
        self.compute_loss = compute_loss
        self.dm = dm
        self.workers = workers
        
    @property
    def model(self):
        model = Doc2Vec(epochs=self.epochs,
                       min_count=self.min_count,
                       vector_size=self.vector_size,
                       window=self.window,
                       negative=self.negative,
                       compute_loss=self.compute_loss,
                       dm=self.dm,
                       workers=self.workers)
        model.build_vocab(self.corpus)
        model.train(self.corpus,
                   total_examples=model.corpus_count,
                   epochs=model.epochs)
        return model
