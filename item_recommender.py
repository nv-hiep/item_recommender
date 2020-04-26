import os, sys
import sklearn

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns

from scipy.sparse           import csr_matrix
from sklearn.neighbors      import NearestNeighbors

import warnings



# Read Amazon data
data = pd.read_csv('data/amazon_ratings.csv',
	                names=['userId', 'productId','Rating','timestamp']
	               )
# print(data.shape) # (7824482, 4)
data.columns = ['user_id', 'item_id', 'rating', 'timestamp']

# Remove timestamp column
data = data.drop('timestamp', axis=1)

print('\n' + '-'*30)
print('* Data header: ')
print( data.head(10) )

print('\n' + '-'*30)
print('* Check the datatypes: ')
print( data.dtypes )

print('\n' + '-'*30)
print('* Print data infor: ')
print( data.info() )


print('\n' + '-'*30)
print('* Missing Values: \n', data.isnull().sum())


_plot = False
if(_plot):
	plt.figure( figsize=(10,10) )

	# data.hist('rating', bins=10)

	data['rating'].value_counts().sort_index().plot(kind='bar')
	plt.title('')
	plt.xlabel('Ratings')
	plt.ylabel('Counts')
	# plt.savefig('', bbox_inches='tight')
	plt.show()
	plt.close()
# End - if

_print = False
if(_print):
	print('\n' + '-'*30)
	print('\n# of ratings :', data.shape[0])
	print('# of users   :', len(np.unique(data.user_id)))
	print('# of items  :', len(np.unique(data.item_id)))




# Find users who highly rate items
rating_counts_by_user = data.groupby('user_id').count()
                        # sort_values('rating', ascending=False)

print('\n' + '-'*30)
print('* Items with Rating Counts by users: ')
print( rating_counts_by_user.head(10) )

top_users_id = rating_counts_by_user[ 
                                 rating_counts_by_user['rating'] > 100
                                 ].index

top_users_ratings = data[
                        data['user_id'].isin(top_users_id)
                        ].sort_values('rating', ascending=False)

print('\n' + '-'*30)
print('* Top-user ratings: ')
print( top_users_ratings.head(10) )




# Find most rated items
rating_counts_by_item = data.groupby('item_id').count()

print('\n' + '-'*30)
print('* Items with Rating Counts by items: ')
print( rating_counts_by_item.head(10) )

top_items_id = rating_counts_by_item[
                                  rating_counts_by_item['rating'] > 100
                                 ].index

top_ratings_user_item = top_users_ratings[
                          top_users_ratings['item_id'].isin(top_items_id)
                          ].sort_values('rating', ascending=False)

print('\n' + '-'*30)
print('* Top rated items: ')
print( top_ratings_user_item.head(10) )


# Some stats
print('\n' + '-'*30)
print('* Statistics of top_ratings_user_item: ')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(top_ratings_user_item['rating'].describe())


thresh = 4.
popular_items = top_ratings_user_item.query('rating >= @thresh')

print('\n' + '-'*30)
print('* Items with rating > 4: ')
print( popular_items.head(10) )

print('\n' + '-'*30)
print('* Counts: items with rating > 4: ')
print( popular_items.count() )





# Using kNN, find the nearest neighbours
popular_items    = popular_items.drop_duplicates(['user_id', 'item_id'])
popular_items_pv = popular_items.pivot(
	                           index = 'item_id',
	                           columns = 'user_id', 
	                           values = 'rating').fillna(0)

print('\n' + '-'*30)
print('* After filling missing values: ')
print( popular_items_pv.values )


popular_items_rating_matrix = csr_matrix(popular_items_pv.values)

print('\n' + '-'*30)
print('* Items-users rating matrix: ')
print( popular_items_rating_matrix)



# kNN model
xmodel = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
xmodel.fit(popular_items_rating_matrix)


# Give recommendations
ind = np.random.choice(popular_items_pv.shape[0])
dist, indx = xmodel.kneighbors(
	                 popular_items_pv.iloc[ind, :].values.reshape(1, -1),
	                 n_neighbors = 11
	                 )



print('\n' + '-'*30)
for i in range(len(dist.flatten())):
    if (i == 0):
        print('* Recommendations for << {0} >> :\n'.format(
        	popular_items_pv.index[ind])
        )
    else:
        print('{0}: {1}, with distance of {2}:'.format(
        	i,
        	popular_items_pv.index[indx.flatten()[i]],
        	np.round( dist.flatten()[i], 3) )
        )