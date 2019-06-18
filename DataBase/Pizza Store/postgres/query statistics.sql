-- statistics for each Recipes
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Egibhiibaaii';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Hfifbbigdbad';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ghdehbhibcdi';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Beeigedcbhhc';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Edacigbgecea';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ihiaacbhhfed';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Degdagedhhfh';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ffhceabaebfe';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Difefdfiidbb';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Gifiaicafhah';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Cdicgbbaddab';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Eeedegehaiac';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Iigegicffcbh';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ibgighebgfbh';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Fbibbaeffadi';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Affghbihheac';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ibagcifiaddd';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ihbbeadghfea';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Aifbabaigbdi';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ggghadaecbaa';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Eahffbbcfdch';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Iidefddibcbg';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Bgdfegcciecc';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Caiffeaaedgc';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ccdghbbhdgid';

-- statistics for each Recipes

CREATE INDEX Index_Rec 
ON Orders USING hash (Recipes_Name);

EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Egibhiibaaii';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Hfifbbigdbad';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ghdehbhibcdi';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Beeigedcbhhc';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Edacigbgecea';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ihiaacbhhfed';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Degdagedhhfh';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ffhceabaebfe';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Difefdfiidbb';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Gifiaicafhah';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Cdicgbbaddab';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Eeedegehaiac';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Iigegicffcbh';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ibgighebgfbh';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Fbibbaeffadi';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Affghbihheac';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ibagcifiaddd';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ihbbeadghfea';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Aifbabaigbdi';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ggghadaecbaa';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Eahffbbcfdch';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Iidefddibcbg';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Bgdfegcciecc';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Caiffeaaedgc';
EXPLAIN SELECT * FROM Orders WHERE Recipes_Name = 'Ccdghbbhdgid';
