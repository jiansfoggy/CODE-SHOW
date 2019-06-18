1 Generate Data

We use the same methods to generate data in MongoDB part. So we don't repeat here.

2 Load Data

Here, we use mongoose to load data into pizza database. The code is saved as create_schemas_insert_data.js under the Load Data folder.

We also save the create order code as create_order.js for 50000 orders under this folder.

3 Tracking Order Function

We create a function called Tracking_Order here and save as Tracking_Order_MongoDB.js here. It works well.

I also write the mongoose function which is saved as Tracking_Order_Mongoose.js, but I am still tuning it.

4 Query Order

After creating 50000 orders, we are trying to query the order by Recipe Name.

I save the query code before adding index and add index code and query code after adding index in the file called query_code.js.

The query results before adding index is saved as result before indexing.txt and the query results after adding index is saved as result after indexing.txt

5 Compare

The comparing part and my own understanding for this MongoDB part is saved as Comparison_MongoDB.txt.