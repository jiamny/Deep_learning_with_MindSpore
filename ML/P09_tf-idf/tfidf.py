import mindspore as ms
from mindspore import context, Tensor, nn

from nltk.tokenize import word_tokenize


ms.set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class TF_IDF:
    """
    TF - Term Frequency:  count of specific word in document / total no. of words in document
    IDF - Inverse Document Frequency: Log ratio of (Total no. of documents / no. of documents containing words)
    """
    def text_processing(self, X):
        """
        text processing: We clean our text by removing special character and keeping them as lower case and each
        line sentence is converted into list of words and then finding the total number of unqiue words in the all the
        documents combined together.
        :param X: List of documents
        :return: Unique words (Vocabulary), all docu/home/hhj/.condarcments [[d_1], [d_2], ..[d_n]]
        """
        documents = []
        vocabulary = []
        for document in X:
            document_words = [word.lower() for word in word_tokenize(document) if word.isalpha()]
            documents.append(document_words)
            for word in document_words:
                if word not in vocabulary:
                    vocabulary.append(word)

        vocabulary = set(vocabulary)
        return vocabulary, documents

    def strtoint(self, vocabulary):
        """
        :param vocabulary: all unique in the documents
        :return: mapping words to integer such as {'the': 1}
        """
        wordToInt = {}
        for i, vocab in enumerate(vocabulary):
            wordToInt[vocab] = i

        return wordToInt

    def vocab_frequency(self, vocabulary, documents):
        """
        :param vocabulary: all unique in the documents
        :param documents: all the documents
        :return: Frequency of word in all the documents combined together
        """
        word_frequency = {}
        for word in vocabulary:
            word_frequency[word] = 0
            for document in documents:
                if word in document:
                    word_frequency[word] += 1

        return word_frequency

    def tf(self, input_document, word):
        """
        Calculating term_frequency
        :param input_document: test document
        :param word: each word in the test document
        :return: tf value (refer the formula above)
        """
        num_words = len(input_document)
        word_frequency = len([token for token in input_document if token==word])
        return word_frequency/num_words

    def idf(self, word, word_frequency, documents):
        """
        :param word: words of the test input document
        :param word_frequency: word frequency w.r.t all the documents available.
        :param documents: all the documents
        :return: idf value
        """
        try:
            word_frequency = word_frequency[word] + 1
        except:
            word_frequency = 1

        return float(ms.ops.log(Tensor(len(documents)*1.0/word_frequency, dtype=ms.float32)).item())

    def fit_tranform(self, document, vocabulary, wordToInt, word_frequency, documents):
        """
        :param document: test input document
        :param vocabulary: all unique words
        :param wordToInt: word to int mapping
        :param word_frequency: each word frequency throughout all the documents
        :param documents: all the documents
        :return: tf_idf vector for test input document
        """
        tfidf_vector = ms.ops.zeros((len(vocabulary), ), dtype=ms.float32)
        print('tfidf_vector: ', tfidf_vector)


        for word in document:
            tf = self.tf(document, word)
            idf = self.idf(word, word_frequency, documents)
            tfidf_values = tf * idf
            tfidf_vector[wordToInt[word]] = tfidf_values
            print('word: ', word, ' tf: ', tf, ' idf: ', idf,
                  ' tfidf_values: ', tfidf_values, ' wordToInt[word]: ', wordToInt[word])
        return tfidf_vector

if __name__ == '__main__':

    vectors = []
    documents = ['Hi, how are you?',
                 'What are you doing?',
                 'what is your name?',
                 'who are you?']

    tfidf_vectorizer = TF_IDF()
    vocabulary, processed_documents = tfidf_vectorizer.text_processing(documents)
    print(vocabulary)
    print(processed_documents)
    wordToInt = tfidf_vectorizer.strtoint(vocabulary)
    print('wordToInt: ', wordToInt)
    vocab_frequency = tfidf_vectorizer.vocab_frequency(vocabulary, processed_documents)
    print('vocab_frequency: ', vocab_frequency)
    _, new_document = tfidf_vectorizer.text_processing([documents[0]])
    print('document: ', new_document[0])
    v = tfidf_vectorizer.fit_tranform(new_document[0],vocabulary, wordToInt, vocab_frequency, documents)
    print(v)
    print(ms.ops.log(Tensor(1.0)))
    print(ms.ops.log(Tensor(2.0)))
    print(ms.ops.log(Tensor(3.0)))
    exit(0)
