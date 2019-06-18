
CREATE DATABASE BoB_Pizza;

/*
Creating Users
*/

CREATE TABLE Users (
	Email_Address text,
	UserID SERIAL NOT NULL,
	First_Name text NOT NULL,
	Last_Name text NOT NULL,
    Phone_Number varchar(10) NOT NULL,
    Address_Line1 text,
    Address_Line2 text,
    City text,
    State char(2),
    Zip varchar(9) CHECK (Zip <> ''),
    PRIMARY KEY ( UserID )
 );

/*
Tracking Inventory
*/

CREATE TABLE Inventory (
	Ingredient_Name text NOT NULL,
    Description text,
    Quantity numeric NOT NULL,
    PRIMARY KEY ( Ingredient_Name )
);

/*
Storing Recipes
*/

CREATE TABLE Recipes (
	Recipes_Name text NOT NULL,
    Description text,
    Cooking_instructions text NOT NULL,
    PRIMARY KEY ( Recipes_Name )
);

CREATE TABLE I_R (
	Recipes_Name text REFERENCES Recipes,
    Ingredient_Name text REFERENCES Inventory,
    R_Quantity integer NOT NULL
);

/*
Tracking Orders
*/

CREATE TABLE Orders (
	ID SERIAL NOT NULL,
	UserID  varchar(10) REFERENCES Users NOT NULL,
	Create_Time timestamp NOT NULL DEFAULT CURRENT_DATE,
    Recipes_Name text NOT NULL REFERENCES Recipes,
    PRIMARY KEY ( ID )
);