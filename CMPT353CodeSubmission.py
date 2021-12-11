import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import difflib
import math
import geopandas
import matplotlib.pyplot as plt
import sklearn
from pandas.io.json import  json_normalize
from shapely.geometry import Polygon, LineString, Point
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None

MATH_PI = 0.017453292519943295
EARTH_RADIUS_BY_2 = 12742000 # in metre 

data = pd.read_json("amenities-vancouver.json.gz", lines=True)
data = data.drop(['tags', 'timestamp'], axis = 1)
census_data = pd.read_csv('CensusLocalAreaProfiles2016.csv', encoding = 'unicode_escape')
yvr_boundary = geopandas.read_file('local-area-boundary.geojson')


# Group similar categories of amenities into their own dataframes.
financial = pd.concat([data[data.amenity == 'atm'], data[data.amenity == 'atm;bank'], data[data.amenity == 'bank'], data[data.amenity == 'money_transfer'], 
            data[data.amenity == 'bureau_de_change'], data[data.amenity == 'payment_terminal']]).reset_index()
financial['category'] = 'financial'

education = pd.concat([data[data.amenity == 'college'], data[data.amenity == 'school'], data[data.amenity == 'language_school'], 
            data[data.amenity == 'library'], data[data.amenity == 'music_school'], data[data.amenity == 'public_bookcase'], 
            data[data.amenity == 'university'], data[data.amenity == 'cram_school']]).reset_index()
education['category'] = 'education'

food_drink = pd.concat([data[data.amenity == 'restaurant'], data[data.amenity == 'bistro'], data[data.amenity == 'cafe'], data[data.amenity == 'fast_food'], 
            data[data.amenity == 'food_court'], data[data.amenity == 'ice_cream'], data[data.amenity == 'juice_bar'], data[data.amenity == 'vending_machine'],
            data[data.amenity == 'drinking_water'], data[data.amenity == 'internet_cafe']]).reset_index()
food_drink['category'] = 'food_drink'

transportation = pd.concat([data[data.amenity == 'bus_station'], data[data.amenity == 'seaplane terminal'], data[data.amenity == 'trolley_bay'], 
            data[data.amenity == 'ferry_terminal'], data[data.amenity == 'taxi']]).reset_index()
transportation['category'] = 'transportation'

alcohol = pd.concat([data[data.amenity == 'bar'], data[data.amenity == 'pub'], data[data.amenity == 'biergarten']]).reset_index()
alcohol['category'] = 'alcohol'

parking = pd.concat([data[data.amenity == 'parking'], data[data.amenity == 'parking_entrance'], data[data.amenity == 'parking_space'], 
            data[data.amenity == 'motorcycle_parking']]).reset_index()
parking['category'] = 'parking'

health = pd.concat([data[data.amenity == 'Pharmacy'], data[data.amenity == 'chiropractor'], data[data.amenity == 'clinic'], data[data.amenity == 'dentist'], 
            data[data.amenity == 'doctors'], data[data.amenity == 'first_aid'], data[data.amenity == 'healthcare'], data[data.amenity == 'hospital'], 
            data[data.amenity == 'pharmacy'], data[data.amenity == 'veterinary']]).reset_index()
health['category'] = 'health'

nightlife = pd.concat([data[data.amenity == 'nightclub'], data[data.amenity == 'stripclub']]).reset_index()
nightlife['category'] = 'nightlife'

waste_management = pd.concat([data[data.amenity == 'waste_transfer_station'], data[data.amenity == 'waste_disposal'], data[data.amenity == 'waste_basket'], 
            data[data.amenity == 'vacuum_cleaner'], data[data.amenity == 'scrapyard'], data[data.amenity == 'trash'], data[data.amenity == 'sanitary_dump_station'],
            data[data.amenity == 'recycling']]).reset_index()
waste_management['category'] = 'waste_management'

science = pd.concat([data[data.amenity == 'science'], data[data.amenity == 'ATLAS_clean_room'], data[data.amenity == 'Observation Platform'], data[data.amenity == 'research_institute']]).reset_index()
science['category'] = 'science'

spiritual = pd.concat([data[data.amenity == 'meditation_centre'], data[data.amenity == 'monastery'], data[data.amenity == 'place_of_worship']]).reset_index()
spiritual['category'] = 'spiritual'

community = pd.concat([data[data.amenity == 'events_venue'], data[data.amenity == 'family_centre'], data[data.amenity == 'arts_centre'], data[data.amenity == 'childcare'],
            data[data.amenity == 'townhall'], data[data.amenity == 'courthouse'], data[data.amenity == 'public_building'], data[data.amenity == 'social_centre'], data[data.amenity == 'social_facility']]).reset_index()
community['category'] = 'community'

children = pd.concat([data[data.amenity == 'nursery'], data[data.amenity == 'kindergarten'], data[data.amenity == 'playground']]).reset_index()
children['category'] = 'children'

bicycle = pd.concat([data.amenity[data.amenity == 'bicycle_parking'], data[data.amenity == 'bicycle_rental'],  data[data.amenity == 'bicycle_repair_station']]).reset_index()
bicycle['category'] = 'bicycle'

electric_car = pd.concat([data[data.amenity == 'EVSE'], data[data.amenity == 'charging_station']]).reset_index()
electric_car['category'] = 'electric_car'

car_services = pd.concat([data[data.amenity == 'car_rental'], data[data.amenity == 'car_rep'], data[data.amenity == 'car_sharing'], data[data.amenity == 'car_wash'], data[data.amenity == 'fuel'],
            data[data.amenity == 'motorcycle_rental']]).reset_index()
car_services['category'] = 'car_services'

entertainment = pd.concat([data[data.amenity == 'casino'], data[data.amenity == 'cinema'], data[data.amenity == 'gambling'], data[data.amenity == 'leisure'], data[data.amenity == 'lounge'],
            data[data.amenity == 'photo_booth'], data[data.amenity == 'spa'], data[data.amenity == 'theatre']]).reset_index()
entertainment['category'] = 'entertainment'

fitness = pd.concat([data[data.amenity == 'dojo'], data[data.amenity == 'gym'], data[data.amenity == 'training']]).reset_index()
fitness['category'] = 'fitness'

shopping = pd.concat([data[data.amenity == 'marketplace'], data[data.amenity == 'shop|clothes']]).reset_index()
shopping['category'] = 'shopping'

business = pd.concat([data[data.amenity == 'office|financial'] , data[data.amenity == 'conference_centre'], data[data.amenity == 'workshop']]).reset_index()
business['category'] = 'business'

nature = pd.concat([data[data.amenity == 'park'], data[data.amenity == 'hunting_stand']]).reset_index()
nature['category'] = 'nature'

safety = pd.concat([data[data.amenity == 'police'], data[data.amenity == 'fire_station'], data[data.amenity == 'ranger_station']]).reset_index()
safety['category'] = 'safety'

postal = pd.concat([data[data.amenity == 'letter_box'], data[data.amenity == 'post_box'], data[data.amenity == 'post_depot'], data[data.amenity == 'post_office']]).reset_index()
postal['category'] = 'postal'

hygiene = pd.concat([data[data.amenity == 'showers'], data[data.amenity == 'toilets']]).reset_index()
hygiene['category'] = 'hygiene'

storage = pd.concat([data[data.amenity == 'storage'], data[data.amenity == 'luggage_locker'], data[data.amenity == 'storage_rental']]).reset_index()
storage['category'] = 'storage'

water = pd.concat([data[data.amenity == 'water_point'], data[data.amenity == 'watering_place'], data[data.amenity == 'fountain']]).reset_index()
water['category'] = 'water'

misc = pd.concat([data[data.amenity == 'animal shelter'], data[data.amenity == 'bench'], data[data.amenity == 'clock'], data[data.amenity == 'compressed_air'], data[data.amenity == 'construction'],
            data[data.amenity == 'disused:restaurant'], data[data.amenity == 'loading_dock'], data[data.amenity == 'shelter'], data[data.amenity == 'telephone'], data[data.amenity == 'housing co-op'],
            data[data.amenity == 'lobby'], data[data.amenity == 'smoking_area'], data[data.amenity == 'studio'], data[data.amenity == 'boat_rental']]).reset_index()
misc['category'] = 'misc'


data = pd.concat([misc, water, storage, hygiene, postal, safety, nature, business, shopping, fitness, entertainment,
                           car_services, electric_car, bicycle, children, community, spiritual, science, waste_management,
                           nightlife, health, parking, alcohol, transportation, food_drink, education, financial]).drop(['index', 0], axis=1)




def return_near_by_structures(lat, lon, radius, input_data):
    
    input_data['distance'] = input_data.apply(lambda row: distance_between_2_points(lat, lon, row['lat'], row['lon']), axis=1)
        
    input_data = input_data[input_data['distance'] < radius].reset_index()
    
    return input_data


def distance_between_2_points(lat1, lon1, lat2, lon2):
    c = np.cos
    a = 0.5 - c((lat2 - lat1) * MATH_PI)/2 + c(lat1 * MATH_PI) * c(lat2 * MATH_PI) * (1 - c((lon2 - lon1) * MATH_PI)) / 2
    
    return EARTH_RADIUS_BY_2 * np.arcsin(np.sqrt(a))


## The score for a given category is equal to the number of amenities of that category in the radius, divided by the total number of 
## amenities in that category in the dataset times the average distance to amenities in the category in the radius.
## Assume that the data provided has a column of categories.
def generate_scores(lat, lon, radius, data):
    
    nearby_amenities = return_near_by_structures(lat, lon, radius, data)
    num_per_category_in_radius = nearby_amenities.groupby('category').size()
    num_per_category_in_data = data.groupby('category').size()
    average_distance_to_amenity = nearby_amenities.groupby('category').mean().drop(['lat', 'lon', 'index'], axis=1)
    
    categories = ['misc', 'water', 'storage', 'hygiene', 'postal', 'safety', 'nature', 'business', 'shopping', 'fitness', 'entertainment', 'car_services', 'electric_car',
                  'bicycle', 'children', 'community', 'spiritual', 'science', 'waste_management', 'nightlife', 'health', 'parking', 'alcohol', 'transportation', 'food_drink', 
                  'education', 'financial']
    categories.sort()
    category_scores = {}
    for category in categories:
        if category in num_per_category_in_radius:
            category_scores[category] = 100000*num_per_category_in_radius.loc[category]/(num_per_category_in_data.loc[category]*max(1, average_distance_to_amenity.loc[category, 'distance']))
        else:
            category_scores[category] = 0
    return category_scores



def neighbourhood_scores(data, boundaries):
  total_categories = data.groupby('category').size()
  neighbourhood_scores_data = pd.DataFrame(columns=['name', 'alcohol', 'bicyle', 'business', 'car_services', 'children', 'community', 'education', 'electric_car', 'entertainment', 'financial', 'fitness', 'food_drink', 'health',
                                    'hygiene', 'misc', 'nature', 'nightlife', 'parking', 'postal', 'safety', 'science', 'shopping', 'spiritual', 'storage', 'transportation', 'waste_management', 'water'])
  for row in boundaries.iterrows():
    name = row[1]['name']
    geo = row[1]['geometry']
    data['bool'] = data.apply(lambda r: geo.contains(Point(r.iloc[1], r.iloc[0])), axis=1)
    points_in_geo = data[data['bool'] == True].drop('bool', axis=1)
    per_category = points_in_geo.groupby('category').size()
    scores = 1000*per_category/total_categories
    scores = scores.fillna(0)
    neighbourhood_scores_data.loc[row[0]] = [name, *scores.tolist()] 
    
  
  return neighbourhood_scores_data


# Given a point by lat and lon, and neighbourhood name, return all the structures in the neighbourhood if the point is inside the neighbourhood, 
# Return None if the point is not in the neighbourhood. 

# The idea of detecting points within boundary is inspired from: 
# https://stackoverflow.com/questions/58288587/get-values-only-within-the-shape-geopandas

def return_near_by_structures_boundary(lat, lon, neighbourhood_name, input_data):
    
    point_list_from_input_data = [Point(xy) for xy in zip(input_data.lon, input_data.lat)]
    
    points_df = geopandas.GeoDataFrame(input_data, geometry = point_list_from_input_data)
    
    
    my_point = Point(lon, lat)
    
    neighbourhood_boundary = yvr_boundary[yvr_boundary['mapid'] == neighbourhood_name].geometry.values[0]
    
    if neighbourhood_boundary.contains(my_point):
        
        res_points = points_df[points_df.geometry.within(neighbourhood_boundary)].reset_index()
        
        return res_points
    
    else:
        return None
        
    



# Generate random points within a given boundary
# Give neighbourhood name and number of points,  return a list of randomly generated points.
def get_random_points_in_neighbourhood(neighbourhood_name, num_point):
    
    neighbourhood_boundary = yvr_boundary[yvr_boundary['mapid'] == neighbourhood_name].geometry.values[0]
    
    min_x, min_y, max_x, max_y = neighbourhood_boundary.bounds
    
    x = np.random.uniform(min_x, max_x, num_point)
    y = np.random.uniform(min_y, max_y, num_point)
    
    points = geopandas.GeoSeries(geopandas.points_from_xy(x, y))
    
    points = points[points.within(neighbourhood_boundary)]
    
    return points




# Separate points in the input_data into each neighbourhood; return a dataframe with a row for each neighbourhood.
def points_into_neighbourhood(input_data, neighbourhood_data):
    
    point_list_from_input_data = [Point(xy) for xy in zip(input_data.lon, input_data.lat)]

    res_df = neighbourhood_data[['mapid', 'name']]
    res_df['numpoints'] = 0
    
    
    
    
    for i in range(0, len(neighbourhood_data)):
        
        boundary = neighbourhood_data[neighbourhood_data['mapid'] == neighbourhood_data.iloc[i].mapid].geometry.unary_union
        
        temp_df = geopandas.GeoDataFrame(input_data, geometry = point_list_from_input_data)
        
        res_df.at[i, 'numpoints'] = len(temp_df[temp_df.geometry.within(boundary)])
        
    
    return res_df


# Use scores of each amenities in each neighbourhood to predict the income in each neighbourhood

census_data_income = census_data.iloc[[1882]].drop(['ID', 'Variable'], axis=1).reset_index().drop('index', axis=1).drop(['Vancouver CSD ', 'Vancouver CMA '], axis=1).transpose()
census_data_income.index = census_data_income.index.map(lambda s: s.rstrip().lstrip())


neighbourhood_data_low_income = census_data.iloc[[2093]].drop(['ID', 'Variable'], axis=1).reset_index().drop('index', axis=1).drop(['Vancouver CSD ', 'Vancouver CMA '], axis=1).transpose()
neighbourhood_data_low_income.index = neighbourhood_data_low_income.index.map(lambda s: s.rstrip().lstrip())



# Calculates the neighbourhood scores
scores = neighbourhood_scores(data, yvr_boundary)
scores['name'] = scores['name'].apply(lambda s: s.rstrip().lstrip())  
scores = scores.sort_values(by=['name'], ascending=True)
scores = scores.set_index('name') 


# Prepares the median income data for the regression 
scores_and_income = scores
scores_and_income['income'] = census_data_income




# Train a regression model to predict the income per neighbourhood using all the category scores as features
X = scores_and_income.drop('income', axis=1)
y = scores_and_income['income']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=32)
model_income = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
print(model_income.score(X_test, y_test))
print(X_test, model_income.predict(X_test))
print(model_income.coef_)


# Train a regression model to predict the income per neighbourhood using the "community" and "education" categories as features
X = scores_and_income[['community', 'education']]
y = scores_and_income['income']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=32)
model_income = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
print(model_income.score(X_test, y_test))
print(X_test, model_income.predict(X_test))
print(model_income.coef_)



# Prepares the low income data for regression analysis
scores_and_low_income = scores
scores_and_low_income['low_income'] = neighbourhood_data_low_income
print(scores_and_low_income)


# Train a regression model to predict the percentage of people in a low income category per neighbourhood using the "community" and "education" categories as features
X = scores_and_low_income[['community', 'education']]
y = scores_and_low_income['low_income']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=32)
model_income = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
print(model_income.score(X_test, y_test))
print(X_test, model_income.predict(X_test))
print(model_income.coef_)



# Using the entertainment amenities to predict the number of single person households


single_person_household = census_data[census_data['Variable'] == '  1 person']
single_person_household = single_person_household.drop(['Variable', 'Vancouver CSD ','Vancouver CMA ', 'ID'], axis = 1)
single_person_household = single_person_household.T
single_person_household.columns = single_person_household.columns.map(str)
single_person_household = single_person_household.rename(columns={'160':'num_single_person_household'})
single_person_household.index = single_person_household.index.map(lambda s: s.rstrip().lstrip())

yvr_boundary_sort = yvr_boundary.sort_values(by = ['name']).reset_index()

test = neighbourhood_scores(data, yvr_boundary)

entertainments = test[['name','entertainment', 'transportation', 'shopping', 'postal']]



test2 = points_into_neighbourhood(data, yvr_boundary)
test2 = test2.sort_values(by = 'mapid').reset_index().drop('index', axis = 1)


entertainments['num_single_household'] = single_person_household['num_single_person_household'].values
entertainments['number_of_amenity'] = test2['numpoints'].values

entertainments = entertainments.reset_index().drop('index', axis = 1)
X = entertainments.drop(['num_single_household', 'name'], axis = 1)
y = entertainments[['num_single_household']]

X_train, X_test, y_train, y_test = train_test_split(X, y)
lingre = LinearRegression()
lingre.fit(X_train, y_train)
y_pred = lingre.predict(X_test)

lingre.score(X_test, y_test)


# Use number of amenity to predict the number of single person households
X = entertainments[['number_of_amenity']]
y = entertainments[['num_single_household']]

X_train, X_test, y_train, y_test = train_test_split(X, y)

lingre = LinearRegression()

lingre.fit(X_train, y_train)

y_pred = lingre.predict(X_test)

lingre.score(X_test, y_test)

# Plot the data to see why we have a low score
plt.scatter(y = entertainments['num_single_household'],x =  entertainments['shopping'])