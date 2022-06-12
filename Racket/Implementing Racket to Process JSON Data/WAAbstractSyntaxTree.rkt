#lang racket

;; Jian Sun
;; 873397832

;; in this file, we defined the following systax, which is waiting for being used in JSONParser

(provide (all-defined-out))

(struct emptyVal () #:transparent)
(struct NullVal (value) #:transparent)
(struct TrueVal (value) #:transparent)
(struct FalseVal (value) #:transparent)
(struct StrVal (value) #:transparent)
(struct NumVal (value) #:transparent)
(struct EleVal (value) #:transparent)

(struct StrJSONPair (string json) #:transparent) 
(struct ObjVal (list-of-strjsonpairs) #:transparent)
(struct Array (list-of-json-elements) #:transparent)


; (ObjVal? (ObjVal (list StrVal "ID" NumVal 1)))
