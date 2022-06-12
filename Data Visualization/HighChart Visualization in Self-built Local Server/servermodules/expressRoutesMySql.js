var mysql      = require('mysql');
module.exports = function(app) {
 //HighCharts api calls to database
 app.get('/api/highchartarea', function(req,res) {
 var connection = mysql.createConnection({
    host     : 'localhost',
    user     : 'root',
    password : 'Blade=787',
    database : 'charts'
  });
 connection.connect();
console.log('coming here to get high chart area');
connection.query('SELECT * from horsepowerandprice',
      function(err, rows, fields) {
        if (!err)
          console.log('The solution is: ', rows);
       else
          console.log('Error while performing Query.', err);
       res.send(rows);

});
connection.end();

});

app.get('/api/highchartline', function(req,res) {
 var connection = mysql.createConnection({
    host     : 'localhost',
    user     : 'root',
    password : 'Blade=787',
    database : 'charts'
  });
 connection.connect();
console.log('coming here to get high chart line');
connection.query('SELECT * from PM25valueChina',
      function(err, rows, fields) {
        if (!err)
          console.log('The solution is: ', rows);
       else
          console.log('Error while performing Query.', err);
       res.send(rows);

});
connection.end();

});

app.get('/api/highchartcolumn', function(req,res) {
 var connection = mysql.createConnection({
    host     : 'localhost',
    user     : 'root',
    password : 'Blade=787',
    database : 'charts'
  });
 connection.connect();
console.log('coming here to get high chart column');
connection.query('SELECT * from GEDtesternum',
      function(err, rows, fields) {
        if (!err)
          console.log('The solution is: ', rows);
       else
          console.log('Error while performing Query.', err);
       res.send(rows);

});
connection.end();

});

 // D3 related api calls to database
 /*app.get('/api/columnchart', function(req,res) {
	 var connection = mysql.createConnection({
	 		host     : 'localhost',
	 		user     : 'root',
	 		password : 'Blade=787',
	 		database : 'charts'
	  });
	 connection.connect();
		var numofcolleges = req.query.numofcolleges;
    console.log('number of colleges : ', numofcolleges);

		if(numofcolleges == 'all') {
			connection.query('SELECT * from cold3',
				function(err, rows, fields) {
					if (!err)
						console.log('The solution is: ', rows);
				 else
						console.log('Error while performing Query.', err);
				 res.send(rows);

			 });
		}else {
				connection.query('SELECT * from cold3 where numcolleges > ?',numofcolleges,
				function(err, rows, fields) {
					if (!err)
						console.log('The solution is: ', rows);
				 else
						console.log('Error while performing Query.', err);
				 res.send(rows);

			 });
		}

		connection.end();

	});

	app.get('/api/linechart', function(req,res) {
			var connection = mysql.createConnection({
					host     : 'localhost',
					user     : 'root',
					password : 'Blade=787',
					database : 'charts'
			 });
			 connection.connect();
       var stockSymbol = req.query.stockSymbol;
       if(stockSymbol == 'all' || stockSymbol == ''){
         connection.query('SELECT * from multilinechart',
             function(err, rows, fields) {
               if (!err)
                 console.log('The line charts is: ', rows);
               else
                 console.log('Error while performing Query.', err);
               res.send(rows);
         });
       }else {
         connection.query('SELECT * from multilinechart where symbol = ?',stockSymbol,
  		 				function(err, rows, fields) {
  		 					if (!err)
  		 						console.log('The line charts is: ', rows);
  		 				  else
  		 						console.log('Error while performing Query.', err);
  		 				  res.send(rows);
  				});
       }

		    connection.end();
	});
//connection.query('SELECT * FROM users WHERE id = ?', [userId], function(err, results) {
  // ...
//});
	app.get('/api/areachart', function(req,res) {
			var connection = mysql.createConnection({
					host     : 'localhost',
					user     : 'root',
					password : 'Blade=787',
					database : 'charts'
			 });
      var startDate = req.query.startDate;
      var endDate = req.query.endDate;
      var year = req.query.year;
      console.log('value of startDate ', startDate);
      console.log('value of endDate ', endDate);
      console.log('value of year ', year);
			 connection.connect();
       if(year == 'all' || year == ''){
         connection.query('SELECT * from areachartstockprice',

         function(err, rows, fields) {
                if (!err)
                console.log('The area charts is: ');
                else
                  console.log('Error while performing Query.', err);
                res.send(rows);

          });
       }else {
         connection.query('SELECT * from areachartstockprice where years >= ?',year,

        function(err, rows, fields) {
               if (!err)
               console.log('The area charts is: ');
               else
                 console.log('Error while performing Query.', err);
               res.send(rows);

         });
       }

				connection.end();
	});

	app.get('/api/scatterplot', function(req,res) {
			var connection = mysql.createConnection({
					host     : 'localhost',
					user     : 'root',
					password : 'Blade=787',
					database : 'charts'
			 });
			 connection.connect();
			 connection.query('SELECT * from scatterplot',
						function(err, rows, fields) {
							if (!err)
								console.log('The scatter plot charts is: ', rows);
							else
								console.log('Error while performing Query.', err);
							res.send(rows);
				});
				connection.end();
	});*/

	// application -------------------------------------------------------------
	app.get('*', function(req, res) {
		res.sendfile('./public/index.html'); // load the single view file (angular will handle the page changes on the front-end)
	});
}
