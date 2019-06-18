/*
Query Before Index
*/
db.Orders.find({Recipes_Name: 'R1'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R2'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R3'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R4'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R5'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R6'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R7'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R8'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R9'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R10'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R11'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R12'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R13'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R14'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R15'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R16'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R17'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R18'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R19'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R20'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R21'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R22'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R23'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R24'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R25'}).explain("executionStats").executionStats;

/*
Query After Index
*/
db.Order.ensureIndex(
{ Recipes_Name : 1 },
{ unique : true, dropDups : true }
)

db.Orders.find({Recipes_Name: 'R1'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R2'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R3'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R4'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R5'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R6'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R7'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R8'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R9'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R10'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R11'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R12'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R13'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R14'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R15'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R16'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R17'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R18'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R19'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R20'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R21'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R22'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R23'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R24'}).explain("executionStats").executionStats;
db.Orders.find({Recipes_Name: 'R25'}).explain("executionStats").executionStats;