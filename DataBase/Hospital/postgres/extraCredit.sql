##########################################
# How many doctors are treating doctors? #
##########################################

SELECT COUNT(DISTINCT EC.DName) AS DOCTOR_COUNT
FROM (SELECT JH.PName, JH.DName, JH.IName, PTT.TName
FROM P_T_T PTT
LEFT JOIN 
(SELECT P.PName, P.DName, DTI.IName
FROM P_T_I DTI 
LEFT JOIN Patients P
ON P.PName = DTI.PName) JH
ON PTT.PName = JH.PName) EC
WHERE EC.PName LIKE 'D%' AND 
EC.IName != 'Healthy';

/*
67 doctors are treating doctors.

 doctor_count 
--------------
           67
(1 row)
*/

####################################################################
# What's the count of how many patients have each kind of illness? #
####################################################################

SELECT EC.IName, COUNT(DISTINCT EC.PName) AS PAT_NUM
FROM (SELECT JH.PName, JH.DName, JH.IName, PTT.TName
FROM P_T_T PTT
LEFT JOIN 
(SELECT P.PName, P.DName, DTI.IName
FROM P_T_I DTI 
LEFT JOIN Patients P
ON P.PName = DTI.PName) JH
ON PTT.PName = JH.PName) EC
WHERE EC.IName != 'Healthy'
GROUP BY EC.IName;

#############################################
# What's the doctor with the most patients? #
#############################################

SELECT EC.DName, COUNT(DISTINCT EC.PName) AS TREATED_PAT
FROM (SELECT JH.PName, JH.DName, JH.IName, PTT.TName
FROM P_T_T PTT
LEFT JOIN 
(SELECT P.PName, P.DName, DTI.IName
FROM P_T_I DTI 
LEFT JOIN Patients P
ON P.PName = DTI.PName) JH
ON PTT.PName = JH.PName) EC
WHERE EC.IName != 'Healthy'
GROUP BY EC.DName
ORDER BY COUNT(DISTINCT EC.PName) DESC
LIMIT 1;

/*
The doctor D96 treats most patients, which is 260.

 dname | treated_pat 
-------+-------------
 D96   |         260
(1 row)
*/

####################################################################
# Which doctor is treating the largest number of unique illnesses? #
####################################################################

SELECT EC.DName, COUNT(DISTINCT EC.IName) AS TREATED_ILL
FROM (SELECT JH.PName, JH.DName, JH.IName, PTT.TName
FROM P_T_T PTT
LEFT JOIN 
(SELECT P.PName, P.DName, DTI.IName
FROM P_T_I DTI 
LEFT JOIN Patients P
ON P.PName = DTI.PName) JH
ON PTT.PName = JH.PName) EC
WHERE EC.IName != 'Healthy'
GROUP BY EC.DName
ORDER BY TREATED_ILL DESC
LIMIT 1;

/*
The doctor D19 is treating the largest number of unique illnesses, which is 411.

 dname | treated_ill 
-------+-------------
 D19   |         411
(1 row)
*/

###############################################################################
# What illness is being treated with the largest number of unique treatments? #
###############################################################################

SELECT EC.IName, COUNT(DISTINCT EC.TName) AS TREAT_NUM
FROM (SELECT JH.PName, JH.DName, JH.IName, PTT.TName
FROM P_T_T PTT
LEFT JOIN 
(SELECT P.PName, P.DName, DTI.IName
FROM P_T_I DTI 
LEFT JOIN Patients P
ON P.PName = DTI.PName) JH
ON PTT.PName = JH.PName) EC
WHERE EC.IName != 'Healthy'
GROUP BY EC.IName
ORDER BY TREAT_NUM DESC
LIMIT 1;


/*
The illness I615 is being treated with the largest number of unique treatments, which is 658.

 iname | treat_num 
-------+-----------
 I615  |       658
(1 row)
*/