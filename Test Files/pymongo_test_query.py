# Get the database using the method we defined in pymongo_test_insert file
from pymongo_get_database import get_database
from pandas import DataFrame
dbname = get_database()
 
# Retrieve a collection named "user_1_items" from database
collection_name = dbname["user_1_items"]
 
# item_details = collection_name.find()
item_details = collection_name.find({"category" : "food"})
# print(item_details)
items_df = DataFrame(item_details)
print(items_df)

# for item in items_df:
#    print(item)
#    # convert the dictionary objects to dataframe
#    items_df = DataFrame(item_details)
#    # see the magic
#    print(items_df)  
#    # # This does not give a very readable output
#    # print(item['item_name'])a
