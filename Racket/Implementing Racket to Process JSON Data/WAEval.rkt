#lang racket

;;;;;;;;;;;;;;;;;;;;;;;;;;
;; JSON Evaluation Part ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;

; Firstly, I called JSONParser and JSONAbstractSyntaxTree files
; Secondly, defined explanation function, empty-env, extend-env and apply-env.
; Thirdly, defined the main evaluation functions, JSON_eval and its helper, JSON_evalHelper
; Fourthly, JSON_eval converts JSON data formats into noraml data format expression.
; Fifthly, JSON_evalHelper offered specific design for 7 JSON data formats.

(require parser-tools/yacc
         (prefix-in lex: parser-tools/lex)
         "WALexer.rkt"
         "WAParser.rkt"
         "WAAbstractSyntaxTree.rkt")
(require test-engine/racket-tests)

(define (empty-env)
  (lambda (searchVar)
    (error "No Binding Found" searchVar)))

(define (extend-env savedVar savedVal savedEnv)
  (lambda (searchVar)
    (if (equal? searchVar savedVar)
        savedVal
        (apply-env savedEnv searchVar))))

(define (apply-env env searchVar)
  (env searchVar))

(define (JSON_eval expression)
  (JSON_evalHelper expression (empty-env)))

(define (JSON_evalHelper expression env)
  (match expression
    [(emptyVal )                                 '()]
    [(TrueVal #t)                                 #t]
    [(FalseVal #f)                                #f]
    [(NullVal '())                               '()]
    [(NumVal expr)                              expr]
    [(StrVal expr)                              expr]
    [(StrJSONPair expr1 expr2)                 (list expr1 (JSON_evalHelper expr2 env))] 
    [(list expr)                               (JSON_evalHelper expr env)]
    [(Array elems)                             (map (lambda (m) (JSON_evalHelper m env)) elems)]
    [(ObjVal members)                          (map (lambda (m) (JSON_evalHelper m env)) members)]
    ))

;;;;;;;;;;;;;;;;;;;;;
;;; Question Part ;;;
;;;;;;;;;;;;;;;;;;;;;

; Question 1

; function name: objectContains?
; type of function: JSON -> String -> Boolean
; description of function parameters: there are two parameters, JSONObject and field_name
; description of function result: This function is to check if the JSON object contains the certain field.
; If it has, the function returns true. Otherwise, return not.
; We made two tests and passed test.

(define (objectContains? JSONObject field_name)
  (if (ObjVal? JSONObject)
      (objectContains? (JSON_eval JSONObject) field_name)
      (if (list? JSONObject)
          (if (empty? JSONObject)
              #f
              (if (equal? (caar JSONObject) field_name)
                  #t
                  (objectContains? (cdr JSONObject) field_name)))
          (error "The input value must be JSON Object."))
      ))



; Question 2

; function name: getField
; type of function: JSON -> String -> JSON
; description of function parameters: there are two parameters, JSONObject and field_name
; description of function result: This function is to output the value corresponding to the field_name.
; If there isn't the field_name in the JSONObject, the function returns the JSON null value.
; We made two tests and passed test.

(define (getField JSONObject field_name)
  (letrec (
           [Split_JSONObj (lambda (val sym)
                          (match val
                            [(ObjVal rest_part) (Print_NewOb rest_part sym)]
                            [_ (error "The input value must be JSON Object.")]
                            ))]

           [Print_NewOb (lambda (v k)
                            (if (empty? v)
                                (NullVal v)
                                (match v
                                  [_ (key_match v k)]
                                  )))]
           )
    (Split_JSONObj JSONObject field_name)))

(define (key_match JSONList field_name)
  (if (empty? JSONList)
      (NullVal null)
      (match (car JSONList)
        [(StrJSONPair expr1 expr2) (if (equal? expr1 field_name)
                                       expr2
                                       (key_match (cdr JSONList) field_name))]
        )))



; Question 3

; function name: filterKeys, JSON_filter and check_length
; type of function: (String -> Bool) -> JSON -> JSON
; description of function parameters: For function filterKeys, there are two parameters, func1 and JSONObj.
; For function JSON_filter, there are two parameters, func1 and JSONList. And JSON_filter is a kind of helper
; function here.
; description of function result: This function will take a predicate function, and apply it to each (key, value)
; pair in the JSON object. If the predicate returns True, the pair is kept, else it is ignored.
; The function will return a JSON object containing only those keys for which the predicate returned True.
; We made two tests and passed test.

(define (filterKeys func1 JSONObj)
  (letrec (
           [Split_JSONObj (lambda (func1 val)
                          (match val
                            [(ObjVal rest_part) (Print_NewOb func1 rest_part)]
                            [_ (error "The input value must be JSON Object.")]
                            ))]

           [Print_NewOb (lambda (f1 JSONList)
                            (if (empty? JSONList)
                                (ObjVal JSONList)
                                (match JSONList
                                  [(NullVal null) (ObjVal (NullVal null))]
                                  [_ (ObjVal (JSON_filter f1 JSONList))]
                                  )))]
           )
    (Split_JSONObj func1 JSONObj)))

(define (JSON_filter func1 JSONList)
  (if (empty? JSONList)
      '()
      (match (car JSONList)
        [(StrJSONPair expr1 expr2) (if (func1 expr1)
                                       (cons (StrJSONPair expr1 expr2) (JSON_filter func1 (cdr JSONList)))
                                       (JSON_filter func1 (cdr JSONList))
                                       )])))

(define (check_length str)
  (if (> (string-length str) 4)
      #t
      #f))



; Question 4

; function name: keyCount
; type of function: JSON -> Integer
; description of function parameters: there is one parameter, JSONObject.
; description of function result: this function is to count the number of the key within the JSON Object.
; the return value is an integer.
; We made two tests and passed test.

(define (keyCount JSONObject)
  (if (ObjVal? JSONObject)
      (keyCount (JSON_eval JSONObject))
      (if (list? JSONObject)
          (if (empty? JSONObject)
              0
              (+ 1 (keyCount (rest JSONObject))))
          (error "The input value must be JSON Object."))))



; Question 5

; function name: keyList
; type of function: JSON -> [String]
; description of function parameters: there is one parameter, JSONObject.
; description of function result: This function returns all the keys within the Object and output a list.
; We made two tests and passed test.

(define (keyList JSONObject)
  (if (ObjVal? JSONObject)
      (keyList (JSON_eval JSONObject))
      (if (list? JSONObject)
          (if (empty? JSONObject)
              '()
              (cons (caar JSONObject) (keyList (rest JSONObject))))
          (error "The input value must be JSON Object."))))



; Question 6

; function name: arrayLength
; type of function: JSON -> Integer
; description of function parameters: there is one parameter, JSONArray
; description of function result: This function is to count the number of elements in the JSON Array.
; It returns an integer number.
; We made two tests and passed test.

(define (arrayLength JSONArray)
  (if (Array? JSONArray)
      (arrayLength (JSON_eval JSONArray))
      (if (list? JSONArray)
          (if (empty? JSONArray)
              0
              (+ 1 (arrayLength (cdr JSONArray)))
          )
          (error "The input value must be JSON Array."))
  ))



; Question 7

; function name: filterRange
; type of function: Integer -> Integer -> JSON -> JSON
; description of function parameters: there are three parameters, low, high, JSONArray
; description of function result: this function takes a low and a high value and a JSON
; Array and returns a JSON array only containing the range of elements between low and high.
; We made three tests and passed test.

(define (filterRange low high JSONArray)
  (if (or (< low 0) (> high (arrayLength JSONArray)))
      (error "Please input a bigger low or smaller high")
      (letrec (
               [Split_JSONArr (lambda (val)
                                (match val
                                  [(Array expr) (Print_NewAr expr)]
                                  ))]
               [Print_NewAr (lambda (JSONList)
                              (if (empty? JSONList)
                                  (Array (emptyVal))
                                  (match JSONList
                                    [ _ (Array (drop (take JSONList (+ high 1)) low))]
                                    )))]
               )
        (Split_JSONArr JSONArray))))



; Question 8

; function name: filterArray, Array_filter and extract_want_str
; type of function: (JSON -> Bool) -> JSON
; description of function parameters: For filterArray, there are two parameters, func1 (test function) and JSONArray (Array).
; For Array_filter, there are two parameters, func1 (test function) and JSONList (list without Array symbol).
; For extract_want_str, this is self-designed function to test filterArray function and it has one parameter, JSONList (list without Array symbol).
; description of function result: this function consumes a function of type (Any JSON Value -> Bool) and a
; JSON Array, and returns a JSON Array containing only those elements where the predicate function returns True.
; We made three tests and passed test.

(define (filterArray func1 JSONArray)
  (letrec (
           [Split_JSONArr (lambda (val)
                          (match val
                            [(Array expr) (Print_NewAr expr)]
                            [ _ (error "The input value must be JSON Array.")]
                            ))]
           [Print_NewAr (lambda (JSONList)
                             (if (empty? JSONList)
                                 (Array '())
                                 (match JSONList
                                   [(NullVal null) (Array '())]
                                   [ _ (Array (Array_filter func1 JSONList))]
                                   )))]
           )
    (Split_JSONArr JSONArray)))

(define (Array_filter func1 JSONList)
  (if (empty? JSONList)
      '()
      (if (extract_want_str (caar JSONList))
          (cons (car JSONList) (Array_filter func1 (cdr JSONList)))
          (Array_filter func1 (cdr JSONList))
          )))

(define (extract_want_str JSONElem)
  (match JSONElem
        [(StrVal expr) (if (> (string-length expr) 8)
                           #t
                           #f)]
        [ _ #f]
        ))



; Question 9

; function name: extractElements 
; type of function: JSON -> [Integer] -> [list]
; description of function parameters: there are two parameters, JSONArray and list_of_indices.
; description of function result: this function takes a JSON Array and a list of indices into
; the array, and returns a new array consisting only of those indices. If the indice is out of
; boundary, the program returns nothing and continues on recurrent.
; We made one test and passed test.

(define (extractElements JSONArray list_of_indices)
  (if (Array? JSONArray)
      (extractElements (JSON_eval JSONArray) list_of_indices)
      (if (list? JSONArray)
          (if (empty? list_of_indices)
              '()
              (if (or (>= (car list_of_indices) 0) (< (car list_of_indices) (arrayLength JSONArray)))
                  (cons (list-ref JSONArray (car list_of_indices)) (extractElements JSONArray (cdr list_of_indices)))
                  (extractElements JSONArray (cdr list_of_indices)))
              )
          (error "The input value must be JSON Array."))
      ))



; Question 10

; function name: increasingIncidents, extract_value, extractHelper10 and locate_field.

; type of function: For increasingIncidents,STRING -> [STRING];
; For extract_value, STRING -> STRING -> JSON; For extractHelper10, [list] -> [STRING];
; For locate_field, JSON -> STRING -> [list].

; description of function parameters: For increasingIncidents, there are two parameters, JSONFile and field_name.
; For extract_value, there are two parameters, JSONFile and field_name. For extractHelper10, there is one parameter,
; target_data. For locate_field, there are two parameters, file_parse and field_name.

; description of function result: this function processes the JSON file to find all the diseases that have had
; increasing numbers of incidents since 2013.
; Here, the purpose of locate_field is to find the string json pair within json object based on target key (field_name)
; , and output the value part.
; the purpose of extractHelper10 is to extract three needed elements, dis_name, num_2017, num_2013. And we predicate if one of them has null
; value. If not, we predicate if the disease counting in 2017 is bigger than or equal to that in 2013.
; If so, we print out the required sentences.
; the purpose of extract_value is to output the value part of string json pair when we input a json file.
; the purpose of increasingIncidents is to print a string list of the disease, which has more count number in 2017 than
; that in 2013.
; We imported the file cdc2018.json to test our code and passed the test.

(define (extract_value JSONFile field_name)
  (let ([js-parse (parsefile JSONFile)])
    (locate_field js-parse field_name))
    )

(define (extractHelper10 target_data)
  (let ([dis_name (list-ref target_data 8)]
        [num_2017 (list-ref target_data 17)]
        [num_2013 (list-ref target_data 25)])
    (if (or (empty? num_2017) (empty? num_2013))
        (printf "~s: We don't deal with null value \n" dis_name)
        (let ([num_17 (string->number (substring num_2017 1 (- (string-length num_2017) 1)))]
              [num_13 (string->number (substring num_2013 1 (- (string-length num_2013) 1)))])
          (if (>= num_17 num_13)
              (printf "~s: ~s cases in 2013, ~s cases in 2017 \n" dis_name num_13 num_17)
              (printf "~s: We don't see increase pattern from 2013 to 2017" dis_name))
          )
        )
    )
  )

(define (locate_field file_parse field_name)
      (if (ObjVal? file_parse)
          (locate_field (JSON_eval file_parse) field_name)
          (if (list? file_parse)
              (if (empty? file_parse)
                  '()
                  (if (equal? (caar file_parse) field_name)
                      (cdar file_parse)
                      (locate_field (cdr file_parse) field_name)))
              (error "The input value must be JSON Object."))
      ))

(define (increasingIncidents JSONFile field_name)
  (let ([df (car (extract_value JSONFile field_name))])
    (map extractHelper10 df)
  ))

;(define target_data (car (extract_value "cdc3.json" "\"data\"")))
; (increasingIncidents "cdc3.json" "\"data\"")
; (increasingIncidents "cdc2018.json" "\"data\"")



; Question 11

; function name: strictlyIncreasing, extractHelper11 locate_field extract_value

; type of function: For strictlyIncreasing, STRING -> [STRING];
; For extract_value, STRING -> STRING -> JSON; For extractHelper11, [list] -> [STRING];
; For locate_field, JSON -> STRING -> [list].

; description of function parameters: For strictlyIncreasing, there are two parameters, JSONFile and field_name.
; For extract_value, there are two parameters, JSONFile and field_name. For extractHelper11, there is one parameter,
; target_data. For locate_field, there are two parameters, file_parse and field_name.

; description of function result: this function returns only those diseases where the number of incidents has been
; monotonically increasing since 2013. 
; Here, the purpose of locate_field is to find the string json pair within json object based on target key (field_name)
; , and output the value part.
; the purpose of extractHelper11 is to extract six needed elements, dis_name, num_2017, num_2016, num_2015, num_2014, num_2013.
; And we predicate if one of them has null value. If not, we predicate if the disease counting in 2017 > that in 2016 > that in 2015
; > that in 2014 > that in 2013.
; If so, we print out the required sentences.
; the purpose of extract_value is to output the value part of string json pair when we input a json file.
; the purpose of strictlyIncreasing is to print a string list of the disease, which the disease counting in 2017 > that in 2016 > that in 2015
; > that in 2014 > that in 2013.
; We imported the file cdc2018.json to test our code and passed the test.

(define (extractHelper11 target_data)
  (let ([dis_name (list-ref target_data 8)]
        [num_2017 (list-ref target_data 17)]
        [num_2016 (list-ref target_data 19)]
        [num_2015 (list-ref target_data 21)]
        [num_2014 (list-ref target_data 23)]
        [num_2013 (list-ref target_data 25)])
    (if (or (empty? num_2017) (empty? num_2016) (empty? num_2015)
            (empty? num_2014) (empty? num_2013))
        (printf "~s: We don't deal with null value \n" dis_name)
        (let ([num_17 (string->number (substring num_2017 1 (- (string-length num_2017) 1)))]
              [num_16 (string->number (substring num_2016 1 (- (string-length num_2016) 1)))]
              [num_15 (string->number (substring num_2015 1 (- (string-length num_2015) 1)))]
              [num_14 (string->number (substring num_2014 1 (- (string-length num_2014) 1)))]
              [num_13 (string->number (substring num_2013 1 (- (string-length num_2013) 1)))])
          (if (and (> num_17 num_16) (> num_16 num_15) (> num_15 num_14) (> num_14 num_13))
              (printf "~s: ~s cases in 2013, ~s cases in 2014, ~s cases in 2015, ~s cases in 2016, ~s cases in 2017 \n"
                      dis_name num_13 num_14 num_15 num_16 num_17)
              (printf "~s doesn't have strictly increase pattern from 2013 to 2017. \n" dis_name))))
        )
    )

(define (strictlyIncreasing JSONFile field_name)
  (let ([df (car (extract_value JSONFile field_name))])
    (map extractHelper11 df))
  )

; (strictlyIncreasing "cdc3.json" "\"data\"")
; (strictlyIncreasing "cdc2018.json" "\"data\"")

;;;;;;;;;;;;;;;;;;;;
;;; Create Array ;;;
;;; and Object   ;;;
;;; examples     ;;;
;;;;;;;;;;;;;;;;;;;;

(define Ar1 (Array (list (list (NumVal 1)) (list (NumVal 2)) (NumVal 3))))

(define Ar2 (Array (list (list (StrVal "\"Now\"")) (list (StrVal "\"We Start Testing array\""))
                         (list (TrueVal #t)) (list (FalseVal #f)) (list (NullVal '())) (list (NumVal 2019))
                         (list (NumVal 114)) (list (Array (list (StrVal "\"Single Element\"")))))))

(define Ar3 (Array (list (list (StrVal "\"This\"")) (list (StrVal "\"is\"")) (list (StrVal "\"my\""))
                         (list (StrVal "\"final\"")) (list (StrVal "\"Weekly Homework.\""))
                         (list (StrVal "\"So much questions.\"")) (list (StrVal "\"Really\"")) (list (StrVal "\"tough.\""))
                         )))

(define Ar4 (Array (list (list (StrVal "\"There\"")) (list (StrVal "\"is\"")) (list (StrVal "\"another\""))
                         (list (StrVal "\"homework called hw 5.\"")) (list (StrVal "\"The due day is tomorrow.\""))
                         )))

(define Ob1 (ObjVal (list (StrJSONPair "\"No1\"" (NumVal 1)))))

(define Ob2 (ObjVal
 (list
  (StrJSONPair
   "\"Weird\""
   (ObjVal
    (list
     (StrJSONPair "\"This one is what I hate most in last homework \r\n, you see,\"" (Array (list (list (StrVal "\"Really So Stupid\"")) (list (StrVal "\"It is\"")) (TrueVal #t))))
     (StrJSONPair "\"What\"" (Array (list (list (FalseVal #f)) (list (StrVal "\"?\"")) (StrVal "\"Incredible, I quit.\""))))
     (StrJSONPair "\"No.2\"" (Array (list (list (StrVal "\"Testing Numeric Value,\"")) (list (StrVal "\"Today is\"")) (list (NumVal 2019)) (list (NumVal 114)) (list (NumVal 120)) (NumVal 121)))))))))
)

(define Ob3 (ObjVal
 (list
  (StrJSONPair
   "\"Weird\""
   (ObjVal
    (list
     (StrJSONPair "\"This one is what I hate most in last homework \r\n, you see,\"" (Array (list (list (StrVal "\"Really So Stupid\"")) (list (StrVal "\"It is\"")) (TrueVal #t))))
     (StrJSONPair "\"What\"" (Array (list (list (FalseVal #f)) (list (StrVal "\"?\"")) (StrVal "\"Incredible, I quit.\""))))
     (StrJSONPair "\"No.2\"" (Array (list (list (StrVal "\"Testing Numeric Value,\"")) (list (StrVal "\"Today is\"")) (list (NumVal 2019)) (list (NumVal 114)) (list (NumVal 120)) (NumVal 121)))))))
  (StrJSONPair "\"Finally\"" (Array (list (list (StrVal "\"Let's\"")) (Array (list (StrVal "\"Start\"")))))))))

(define Ob4 (ObjVal
 (list
  (StrJSONPair "\"name\"" (StrVal "\"sj\""))
  (StrJSONPair "\"id\"" (NumVal 8733))
  (StrJSONPair "\"Age\"" (NumVal 27))
  (StrJSONPair "\"major\"" (StrVal "\"CS\""))
  (StrJSONPair "\"Description\"" (NullVal '()))
  (StrJSONPair "\"Confident?\"" (TrueVal #t)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test all questions ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;

; Q1
(check-expect (objectContains? (ObjVal (list (StrJSONPair "\"id\"" (StrVal "\"1\"")))) "id") #f)
(check-expect (objectContains? Ob2 "\"Weird\"") #t)
; Q2
(check-expect (getField Ob2 "\"id\"") (NullVal '()))
(check-expect (getField Ob2 "\"Weird\"") (ObjVal
 (list
  (StrJSONPair "\"This one is what I hate most in last homework \r\n, you see,\"" (Array (list (list (StrVal "\"Really So Stupid\""))
                                                                                               (list (StrVal "\"It is\"")) (TrueVal #t))))
  (StrJSONPair "\"What\"" (Array (list (list (FalseVal #f)) (list (StrVal "\"?\""))
                                       (StrVal "\"Incredible, I quit.\""))))
  (StrJSONPair "\"No.2\"" (Array (list (list (StrVal "\"Testing Numeric Value,\""))
                                       (list (StrVal "\"Today is\"")) (list (NumVal 2019))
                                       (list (NumVal 114)) (list (NumVal 120)) (NumVal 121)))))))
; Q3
(check-expect (filterKeys check_length Ob3) (ObjVal
 (list
  (StrJSONPair
   "\"Weird\""
   (ObjVal
    (list
     (StrJSONPair "\"This one is what I hate most in last homework \r\n, you see,\"" (Array (list (list (StrVal "\"Really So Stupid\"")) (list (StrVal "\"It is\"")) (TrueVal #t))))
     (StrJSONPair "\"What\"" (Array (list (list (FalseVal #f)) (list (StrVal "\"?\"")) (StrVal "\"Incredible, I quit.\""))))
     (StrJSONPair "\"No.2\"" (Array (list (list (StrVal "\"Testing Numeric Value,\"")) (list (StrVal "\"Today is\"")) (list (NumVal 2019)) (list (NumVal 114)) (list (NumVal 120)) (NumVal 121)))))))
  (StrJSONPair "\"Finally\"" (Array (list (list (StrVal "\"Let's\"")) (Array (list (StrVal "\"Start\"")))))))))
(check-expect (filterKeys check_length Ob4) (ObjVal (list (StrJSONPair "\"name\"" (StrVal "\"sj\""))
                                                          (StrJSONPair "\"Age\"" (NumVal 27)) (StrJSONPair "\"major\"" (StrVal "\"CS\""))
                                                          (StrJSONPair "\"Description\"" (NullVal '())) (StrJSONPair "\"Confident?\"" (TrueVal #t)))))
; Q4
(check-expect (keyCount Ob2) 1)
(check-expect (keyCount Ob3) 2)
; Q5
(check-expect (keyList Ob2) '("\"Weird\""))
(check-expect (keyList Ob3) '("\"Weird\"" "\"Finally\""))
; Q6
(check-expect (arrayLength Ar1) 3)
(check-expect (arrayLength Ar2) 8)
; Q7
(check-expect (filterRange 2 6 Ar2)
              (Array (list (list (TrueVal #t)) (list (FalseVal #f)) (list (NullVal '()))
                           (list (NumVal 2019)) (list (NumVal 114)))))
(check-expect (filterRange 0 4 Ar2)
              (Array (list (list (StrVal "\"Now\"")) (list (StrVal "\"We Start Testing array\""))
                           (list (TrueVal #t)) (list (FalseVal #f)) (list (NullVal '())))))
(check-expect (filterRange 2 2 Ar2) (Array (list (list (TrueVal #t)))))
; Q8
(check-expect (filterArray extract_want_str Ar2) (Array (list (list (StrVal "\"We Start Testing array\"")))))
(check-expect (filterArray extract_want_str Ar3)
              (Array (list (list (StrVal "\"Weekly Homework.\"")) (list (StrVal "\"So much questions.\"")))))
(check-expect (filterArray extract_want_str Ar4)
              (Array (list (list (StrVal "\"another\"")) (list (StrVal "\"homework called hw 5.\""))
                           (list (StrVal "\"The due day is tomorrow.\"")))))
; Q9
(check-expect (extractElements Ar2 '(1 3 7 0 6 6 2 5 4)) '("\"We Start Testing array\"" #f ("\"Single Element\"")
                                                                                        "\"Now\"" 114 114 #t 2019 ())
)
; Q10
; Q11
(test)