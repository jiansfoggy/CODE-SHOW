DROP TABLE P_T_I;
DROP TABLE P_T_T;
DROP TABLE Patients;
DROP TABLE Treatment;
DROP TABLE Doctors;
DROP TABLE Illness;

CREATE TABLE Doctors (
    DName text NOT NULL,
    If_Patient boolean NOT NULL,
    PRIMARY KEY ( DName )
 )

CREATE TABLE Patients (
    PatID char(20) NOT NULL,
    PName text NOT NULL,
    DName text REFERENCES Doctors,
    PRIMARY KEY ( PatID )
 );

CREATE TABLE Illness (
    IName text NOT NULL,
    PRIMARY KEY (IName)
 );

CREATE TABLE P_T_I (
    PTIID char(20) NOT NULL,
    PName text NOT NULL,
    IName text REFERENCES Illness,
    PRIMARY KEY ( PTIID )
 );

CREATE TABLE Treatment (
    TName text NOT NULL,
    PRIMARY KEY ( TName )
 );

CREATE TABLE P_T_T (
    PName text NOT NULL,
    TName text REFERENCES Treatment
);