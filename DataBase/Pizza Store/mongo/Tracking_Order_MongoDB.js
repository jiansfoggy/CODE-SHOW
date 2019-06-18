db.system.js.save({_id: "Tracking_Order", value: function(username, reci) {
	var uu = db.users.findOne({'UserID': username});
	var rr = db.recipes.findOne({'Recipes_Name': reci});
	var ii = db.inventories;

	if (rr == null) {
		throw "Recipe does not exist! No substitutions, our recipes are pefect!";
	} else if (uu == null) {
		throw "User unknown. Please enter a valid user id.";
	} else {
		var ingre_num = rr.Ingredient.length;
		var count = 0;

		//Check ingredient amounts
		Array.prototype.forEach.call(rr.Ingredient,function(ingre){
			var I_N = ingre.Ingre_Name;
			var QNT = ingre.QNTY;

			var Stock_Amount = ii.findOne({'Ingredient_Name': I_N}).Quantity;
			if (QNT < Stock_Amount) {count = count+1;}
		});

		if (ingre_num-count > 1) {
			//raise exception
			throw "There Is Not Enough Ingredients to Make One Pizza! Try Another Flavor";
		} else {
			//db.update to change value
			Array.prototype.forEach.call(rr.Ingredient,function(ingre){
				var I_N = ingre.Ingre_Name;
                var QNT = ingre.QNTY;
                var Stock_Amount = ii.findOne({Ingredient_Name: I_N}).Quantity;
                var Left = Stock_Amount - QNT;
				ii.update({'Ingredient_Name': I_N}, {$set:{'Quantity': Left}});
			});

			//create order
			var ct = new Date();
            var C_T = ct.toString();

			db.Orders.insert({
				'UserID': username,
				'Create_Time': C_T,
				'Recipes_Name': reci,
				'Status': 'Pending'
				
			});

			return "You Order Will Be Ready Soon!";
		}
	}
}});