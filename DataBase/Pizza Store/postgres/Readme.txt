
1 Generate Data

We use Excel to generate Users, Inventory, I_R, Recipes datasets.
The code are stored as gen_users.txt, gen_rec_inv.txt under Generated Dataset folder.

And I_R is a kind of lookup table, which saves the recipes name, related ingredient and quantity for each pizza.

2 Insert Data

We write sql file to insert data. The codes are saved as insert_user.sql, insert_inventory.sql, insert_recipes.sql, insert_I_R.sql, create_order.sql under Load Data folder.

3 Design Tables

We design the table structure for Users, Inventory, I_R, Recipes and Orders table.
The code is saved as design tables.sql.

4 Track Orders

This is the main body function. It is consisting of keep_recipe rule(avoid changing the recipes), track_order( ) function(check if the quantity of ingredients are enough to create an order and insert the UserID, create time and recipe name into Order dataset), create index.
The code is saved as Tracking Oders.sql.

5 Query Statistic

We do this part twice. The first time, we query order without index, the second time, we create an index and query again.
The code is saved as query statistics.sql.

6 Result

We save the query results as two different files, Result for Nonindex Condition.txt and Result for Index Condition.txt.

7 Comparison

It is saved as Comparison_PostgreSQL.txt

