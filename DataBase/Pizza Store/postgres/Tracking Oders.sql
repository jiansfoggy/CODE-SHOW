/*
Rule that stop updating Recipes
*/

CREATE RULE keep_recipe AS ON UPDATE TO Recipes
     DO INSTEAD
     UPDATE Recipes SET Description = OLD.Description
     WHERE Recipes_Name = OLD.Recipes_Name;

/*
SERVER-SIDE Function
*/

CREATE OR REPLACE FUNCTION track_order(Username integer, RRSS varchar(50)) RETURNS void AS $$ 

DECLARE
  UpdateQuant integer;
  Ingre_List RECORD;
  Inven_List RECORD;
  I_R_Content RECORD;
  Update_Time TIMESTAMP:= LOCALTIMESTAMP;

BEGIN

    FOR I_R_Content IN SELECT * FROM I_R WHERE Recipes_Name = RRSS
    
    LOOP

        FOR Inven_List IN (
        SELECT * FROM Inventory I JOIN I_R
	    ON I_R.Ingredient_Name =  I.Ingredient_Name
	    WHERE I_R.Recipes_Name = RRSS )
	    LOOP
            IF (Inven_List.Quantity < Inven_List.R_Quantity) THEN
                RAISE EXCEPTION 'There is not enough %', Inven_List.Ingredient_Name;
            ELSE
                UPDATE Inventory SET Quantity = Inven_List.Quantity - Inven_List.R_Quantity
		      	WHERE Ingredient_Name = Inven_List.Ingredient_Name;
            END IF;
	    END LOOP;

    END LOOP;

    INSERT INTO Orders (UserID, Create_Time, Recipes_Name) 
    VALUES (Username, Update_Time, RRSS);
    RAISE NOTICE 'Your Pizza Will Be Ready Soon!';
    RETURN;
END;
$$ 
LANGUAGE plpgsql;

/*
Create Index For Each Client
*/

CREATE INDEX Index_Rec 
ON Orders USING hash (Recipes_Name);
