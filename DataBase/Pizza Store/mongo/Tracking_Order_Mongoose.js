/*
Tracking Orders Function
*/

/*
SERVER-SIDE Function
*/


var t_o = mongoose.Schema({ 
  username: Number, 
  reci: String
});
var Ordersc = mongoose.Schema({ 
  UserID:String,
  Create_Time: Date, 
  Recipes_Name: String,
  Status: String
});
var order = mongoose.model('order',Ordersc,'pizza');

t_o.methods.tracking_order = function(){
  Users.findOne({UserID: this.username}, function(err, data){
    if (err) console.log(err);
    console.log(data);
    if (data == null) {
    throw "Unknown User ID. Don't Naughty. Input Valid One!";
  }
  });
  Recipes.findOne({Recipes_Name: this.reci}, function(err, data){
    if (err) console.log(err);
    console.log(data);
    var ii = Inventory.find();
    var oo = Orders
    if (rr == null) {
    throw "Unexisted Recipe! Don't Substite Our Perfect Recipes!";
  } else {
    var ingre_num = rr.Ingredient.Ingre_Name.length;
    var count = 0;

    //Check Ingredient Quantity
    rr.Ingredient.forEach(function(ingre){
      var I_N = ingre.Ingre_Name;
      var QNT = ingre.QNTY;

      var Stock_Amount = ii.findOne({Ingredient_Name: I_N}).Quantity;
      if (QNT <= Stock_Amount) { count++; }
      }, function(err, data){
        if (err) console.log(err); console.log(data);
    });  
  
    if (count != ingre_num) {
      //raise exception
      throw "There Is Not Enough Ingredients to Make One Pizza!";
      } else {
      //update to change value
        rr.Ingredient.forEach(function(ingre){
          var I_N = ingre.Ingre_Name;
          var QNT = ingre.QNTY;
          var Stock_Amount = ii.findOne({Ingredient_Name: I_N}).Quantity;
          var Left = Stock_Amount - QNT;
          ii.update({Ingredient_Name: I_N}, {$set:{Quantity: Left}},function(error){
            if(error) {
              console.log(error);
            } else {
              console.log('Update Success!');
            }
          });
        });

        //create order
        var CT = new Date();
        var C_T = CT.toString();
        OO.collection.insert({UserID:this.username, Create_Time:C_T, Recipes_Name:this.reci, Status: 'Pending'}, function (err, docs) {
          if (err){
            return console.error(err);
          } else {
            console.log("Multiple documents inserted to Collection");
          }
        });
        var orderr = new order({ UserID:username, Create_Time:C_T, Recipes_Name:reci,Status: 'Pending' });
 
    // save model to database
    orderr.save(function (err, book) {
      if (err) return console.error(err);
      console.log(orderr.UserID + " saved to bookstore collection.");
    });
        return "You Order Will Be Ready Soon!";
    }
  }

  });
  
};