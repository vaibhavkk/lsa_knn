REDUCEDCOMPONENTS = 50

from sklearn.datasets import load_files
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import PorterStemmer
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from numpy.linalg import svd
from numpy import *
import pylab
import numpy
from sklearn.datasets import load_files
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from os import walk
import matplotlib.pyplot as plt
import glob



newgroupdata = load_files("../dataset_bbc/",
			description = None, load_content = True,
			encoding='latin1', decode_error='strict', shuffle=False,
			random_state=42)
				
stemmer = PorterStemmer()
class StemmedCountVectorizer(CountVectorizer):
		def build_analyzer(self):
			analyzer = super(StemmedCountVectorizer,self).build_analyzer()
			return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

count_vectorizer = StemmedCountVectorizer(min_df=3, analyzer="word", stop_words=text.ENGLISH_STOP_WORDS)

frequency_matrix = count_vectorizer.fit_transform(newgroupdata.data)


vocabulary = count_vectorizer.fit(newgroupdata.data)
filenameslist = newgroupdata.filenames

path = '../dataset_bbc /'

feat_names = count_vectorizer.get_feature_names()

for x in range(0, len(filenameslist)):
	filenameslist[int(x)] = filenameslist[int(x)].replace(path, '')

tfidf = TfidfTransformer()
tfidf_matrix = tfidf.fit_transform(frequency_matrix)
#print 'freq matrix shape is' + frequency_matrix.shape
print tfidf_matrix.shape

'''
cosine_matrix = cosine_similarity(tfidf_matrix)

with open('cosine_matrix.csv', 'w') as f:
		writer = csv.DictWriter(f, ['']+filenameslist.tolist())
		writer.writeheader()
		filectr = 0
		for sublist in cosine_matrix:
			
			f.write(filenameslist[filectr])
			filectr = filectr+1
			i = 0
			
			f.write(',')
			for item in sublist:
					if i != (len(sublist)-1) :
						f.write(str(item)+',')
						i = i+1
					else :
						f.write(str(item)+'\n')
				
	#f.write('\n')
'''

#LSA Learning phase 

svd = TruncatedSVD(n_components = REDUCEDCOMPONENTS)
lsa = make_pipeline(svd, Normalizer(copy=False))
bbc_lsa = lsa.fit_transform(tfidf_matrix)
# writing terms and respective weight of training file in csv
f = open('terms_wts.csv','w')
#To print major features for the componentssvd
#for compNum in range(0,1000):
for compNum in range(0,int(len(svd.components_))):
	comp = svd.components_[compNum]
	indeces = numpy.argsort(comp).tolist()
	indeces.reverse()
	#term = [feat_names[weightIndex] for weightIndex in indeces[0:10]]
	term = [feat_names[weightIndex] for weightIndex in indeces[0:21]]
	weights = [comp[weightIndex] for weightIndex in indeces[0:21]]
	term.reverse()
	weights.reverse()
#	print term
#	print weights
	i = 0 

	for t in term :
		f.write(t.encode())
		f.write(','+str(weights[i]))
		f.write(','+str(compNum)+'\n')


		i = i + 1




#explained_variance = svd.explained_variance_ratio_.sum()
#print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


'''
		# Graphs creation	
		positions = pylab.arange(10)+.5	
		pylab.figure(compNum)
		pylab.barh(positions, weights, align='center')
		pylab.yticks(positions, term)
		pylab.xlabel('Weight')
		pylab.title('Strongest terms in SVD component %d' % (compNum))
		pylab.grid(True)
		pylab.savefig('weight_graph'+str(compNum)+'.png')	

'''
f.close()

#cluster all LSA data using KMeans
from sklearn.cluster import KMeans
import numpy as np
#print("%d documents" % len(newgroupdata.data))
print("Clustering Data using k-means algorithm \n with %d categories" % len(newgroupdata.target_names))
print()
labels = newgroupdata.target
true_k = np.unique(labels).shape[0]
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
km.fit(bbc_lsa)
from sklearn import metrics
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"% metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(bbc_lsa, km.labels_, sample_size=1000))
print()
print "Top 10 terms in a cluster"
original_space_centroids = svd.inverse_transform(km.cluster_centers_) 
order_centroids = original_space_centroids.argsort()[:, ::-1]
for i in range(true_k):
        print("Cluster %d:" % i)
        for ind in order_centroids[i, :10]:
            print(' %s' % feat_names[ind])
        print()



'''
# LSA Testing 


testdata = load_files("../test_bbc/",
			description = None, load_content = True,
			encoding='latin1', decode_error='strict', shuffle=False,
			random_state=42)
				
stemmer = PorterStemmer()
'''
'''
class StemmedCountVectorizer(CountVectorizer):
		def build_analyzer(self):
			analyzer = super(StemmedCountVectorizer,self).build_analyzer()
			return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
''' 

'''


count_vectorizer_test = StemmedCountVectorizer(min_df=3, analyzer="word", stop_words=text.ENGLISH_STOP_WORDS)

frequency_matrix_test = count_vectorizer.fit_transform(testdata.data)


vocabulary_test = count_vectorizer_test.fit(testdata.data)
filenameslist_test = testdata.filenames
path = '../all_in_one_bbc/'

feat_names_test = count_vectorizer.get_feature_names()

for x in range(0, len(filenameslist_test)):
	filenameslist_test[int(x)] = filenameslist_test[int(x)].replace(path, '')

tfidf_test = TfidfTransformer()
tfidf_matrix_test = tfidf_test.fit_transform(frequency_matrix)
#print 'freq matrix shape is' + frequency_matrix.shape
print tfidf_matrix_test.shape

svd_test = TruncatedSVD(n_components = REDUCEDCOMPONENTS)
lsa_test = make_pipeline(svd_test, Normalizer(copy=False))
test_lsa = lsa_test.fit_transform(tfidf_matrix_test)
# Perform training and testing using classifier 

#creating Pipeline
linear_SVC_classifier = Pipeline([ ('vect', CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS)),
                                ('tfidf', TfidfTransformer()),
                                ('clf', LinearSVC(random_state=111))
                        ])
print "LinearSVC Classifier intialized"


linear_SVC_classifier.fit(newgroupdata.data, newgroupdata.target)
print str(len(newgroupdata.filenames))+" training files loaded."
#Predicting classes from testdata
predicted_classes = linear_SVC_classifier.predict(testdata.data)

#Calculating accuracy
print "\nThe accuracy of classifer is - "+str((metrics.accuracy_score(testdata.target, predicted_classes))*100) + " %"

#Confusion Matrix
cm = confusion_matrix(testdata.target, predicted_classes)
print "\nConfusion Matrix:\n"
print cm

# plotting cm 
#Code for confusion matrix graph
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
'''
