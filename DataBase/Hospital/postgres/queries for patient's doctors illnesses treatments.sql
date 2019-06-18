
######################################################
# query patient's doctors, illnesses, and treatments #
######################################################

SELECT JH.DName, JH.IName, PTT.TName
FROM P_T_T PTT
LEFT JOIN 
(SELECT P.PName , P.DName, DTI.IName
FROM P_T_I DTI 
LEFT JOIN Patients P
ON P.PName = DTI.PName) JH
ON PTT.PName = JH.PName
WHERE JH.PName = 'P88'
GROUP BY JH.DName, JH.IName, PTT.TName
ORDER BY JH.DName, JH.IName, PTT.TName;

/*
Query result

 dname | iname | tname 
-------+-------+-------
 D90   | I105  | T130
 D90   | I105  | T152
 D90   | I105  | T175
 D90   | I105  | T185
 D90   | I105  | T19
 D90   | I105  | T261
 D90   | I105  | T320
 D90   | I105  | T38
 D90   | I105  | T4
 D90   | I105  | T410
 D90   | I105  | T413
 D90   | I105  | T462
 D90   | I105  | T519
 D90   | I105  | T643
 D90   | I105  | T679
 D90   | I105  | T70
 D90   | I682  | T130
 D90   | I682  | T152
 D90   | I682  | T175
 D90   | I682  | T185
 D90   | I682  | T19
 D90   | I682  | T261
 D90   | I682  | T320
 D90   | I682  | T38
 D90   | I682  | T4
 D90   | I682  | T410
 D90   | I682  | T413
 D90   | I682  | T462
 D90   | I682  | T519
 D90   | I682  | T643
 D90   | I682  | T679
 D90   | I682  | T70
*/