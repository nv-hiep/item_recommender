import os, sys
import sklearn

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

from scipy.sparse           import csr_matrix
from sklearn.decomposition  import TruncatedSVD
from sklearn.neighbors      import NearestNeighbors



# Read data
book  = pd.read_csv( 'data/Books.csv', sep=';', error_bad_lines=False, encoding='latin-1' )
book.columns = ['ISBN', 'Book_Title', 'Book_Author', 'Year_Of_Publication', 'Publisher',
                'Image_URL_S', 'Image_URL_M', 'Image_URL_L']
# books.shape  (271360, 8)



user  = pd.read_csv( 'data/Users.csv', sep=';', error_bad_lines=False, encoding='latin-1' )
user.columns = ['User_ID', 'Location', 'Age']
# users.shape (278858, 3)



rating = pd.read_csv( 'data/Ratings.csv', sep=';', error_bad_lines=False, encoding='latin-1' )
rating.columns = ['User_ID', 'ISBN', 'Book_Rating']
# rating.shape 1149780, 3






_plot = False
if(_plot):
	plt.figure( figsize=(10,10) )
	rating['Book_Rating'].value_counts(sort=False).plot(kind='bar')
	plt.title('')
	plt.xlabel('Ratings')
	plt.ylabel('Counts')
	# plt.savefig('', bbox_inches='tight')
	plt.show()
	plt.close()


	plt.figure( figsize=(10,10) )
	user['Age'].hist(bins=[0, 10, 20, 30, 40, 50, 100])
	plt.title('Histogram')
	plt.xlabel('Age')
	plt.ylabel('Count')
	plt.savefig('', bbox_inches='tight')
	plt.show()
	plt.close()
# End - if





# Find most rated books
rating_counts = rating.groupby('ISBN')['Book_Rating'].\
                count().\
                reset_index().\
                rename( columns = {'Book_Rating': 'total_ratings'} )\
                [['ISBN', 'total_ratings']]



# Top 10 rated books
top_ISBN = rating_counts.sort_values('total_ratings', ascending=False)['ISBN'].tolist()[:11]
print('\n----')
print('* Top ten rated books: ')
print( book.loc[book['ISBN'].isin(top_ISBN)]['Book_Title'] )




# Merge book and rating to find popular books
book_rating = pd.merge( rating, book, on='ISBN' )
book_rating = book_rating.drop( ['Book_Author', 'Image_URL_S', 'Image_URL_M', 'Image_URL_L', 'Year_Of_Publication', 'Publisher'], axis=1 )



# Remove missing values.
book_rating = book_rating.dropna(axis=0, subset=['Book_Title'])



# Group by book titles and create a new column for total rating count.
book_rating_count = (book_rating.\
                groupby(by = ['Book_Title'])['Book_Rating'].\
                count().\
                reset_index().\
                rename(columns = {'Book_Rating': 'Total_Rating_Count'})\
                [['Book_Title', 'Total_Rating_Count']] ).\
                sort_values('Total_Rating_Count', ascending=False)

print('\n----')
print('* Book with Total Rating Count: ')
print( book_rating_count.head() )





# Merge rating with total rating count
rating_total_rating = book_rating.merge(book_rating_count,
	                                   left_on = 'Book_Title',
	                                   right_on = 'Book_Title',
	                                   how = 'left' # use only keys from left frame
	                                   ).sort_values('Total_Rating_Count', ascending=False)
print('\n----')
print('* Book rating + Total Rating Count: ')
print( rating_total_rating.head() )

# Some stats
print('\n----')
print('* Statistics of Total_Rating_Count: ')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(book_rating_count['Total_Rating_Count'].describe())

print('\n----')
print('* Statistics of Top of the Total_Rating_Count: ')
print(book_rating_count['Total_Rating_Count'].quantile(np.arange(.8, 1., .01)))
# About 1% of the books received 50+ ratings




thresh = 50 # rates
popular_books = rating_total_rating.query('Total_Rating_Count >= @thresh')

print('\n----')
print('* Books with Total_Rating_Count > 50: ')
print( popular_books.head() )



# Users in UK, USA and Russia only
pop_books_users = popular_books.merge(user,
	                           left_on = 'User_ID',
	                           right_on = 'User_ID',
	                           how = 'left'
	                           ).sort_values('Total_Rating_Count', ascending=False)

URUK_pop_books_users = pop_books_users[
                                        pop_books_users['Location'].str.\
                                        contains('usa|russia|united kingdom')
                                        ]
URUK_pop_books_users.head()

print('\n----')
print('* Popular books for Users in UK, USA and Russia only: ')
print( URUK_pop_books_users.columns)
print( URUK_pop_books_users[['User_ID', 'Book_Rating', 'Book_Title', 'Total_Rating_Count']].head(10) )







# Using kNN, find the nearest neighbours
URUK_pop_books_users    = URUK_pop_books_users.drop_duplicates(['User_ID', 'Book_Title'])
URUK_pop_books_users_pv = URUK_pop_books_users.pivot(
	                           index = 'Book_Title',
	                           columns = 'User_ID', 
	                           values = 'Book_Rating').fillna(0)

print('\n----')
print('* After filling missing values: ')
print( URUK_pop_books_users_pv.values )

URUK_pop_books_users_rating_matrix = csr_matrix(URUK_pop_books_users_pv.values)

print('\n----')
print('* Books-users rating matrix: ')
print( URUK_pop_books_users_rating_matrix)





# kNN model
xmodel = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
xmodel.fit(URUK_pop_books_users_rating_matrix)




# Give recommendations
ind = np.random.choice(URUK_pop_books_users_pv.shape[0])
dist, indx = xmodel.kneighbors(
	                 URUK_pop_books_users_pv.iloc[ind, :].values.reshape(1, -1),
	                 n_neighbors = 11
	                 )



print('\n----')
for i in range(len(dist.flatten())):
    if (i == 0):
        print('* Recommendations for {0}:\n'.format(
        	URUK_pop_books_users_pv.index[ind])
        )
    else:
        print('{0}: {1}, with distance of {2}:'.format(
        	i,
        	URUK_pop_books_users_pv.index[indx.flatten()[i]],
        	np.round( dist.flatten()[i], 3) )
        )