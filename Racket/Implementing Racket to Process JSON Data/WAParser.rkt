#lang racket

;; Jian Sun
;; 873397832

;; comments
;; Firstly, we called different tokens defined at JSONLexer file, which are
;; lambda-def,sign-value,null-values,bool-values,parens,
;; bool-operators,end-of-file,names-and-value.
;; With these tokens, we defined the grammar of parse, which are consisting of 4 big parts.
;; expr, elems, member, members.
;; Within the expr, we defined the 7 data formats as required.
;; elems is a small recursive structure. We defined empty value, single element value, and multiple elements value (elems COMMA elems).
;; Within the member, we defined String JSON Pair (STRING COLON expr) as required.
;; members is similar to elems, which is another recursive structure.
;; The grammar is written within a big fucntion called myparser.
;; Out of myparser, there are three functions, parse, parsestr and parsefile.
;; parse and parsest are used to read text.
;; parsefile is used to read the testing json file.
;; We tried 16 tests and passed all of them. We also tested the json file,
;; cdc2018.json, and get the correct answer. But due to the file size, it is time-consuming.
;; So our code is ok to run.

(require parser-tools/yacc
         (prefix-in lex: parser-tools/lex)
         "WALexer.rkt"
         "WAAbstractSyntaxTree.rkt")
(require test-engine/racket-tests)

(provide (all-defined-out))

(define myparser
  (parser
      (start expr)
      (end EOF)
      (tokens lambda-def
              sign-value
              null-values
              bool-values
              parens
              bool-operators
              end-of-file
              names-and-value)
      (error (lambda (tok-ok? tok-name tok-value)
               (printf "Parser error: token ~a value ~a"
                           tok-name
                           tok-value)))
      (grammar
          (expr [(TRUE)                               (TrueVal true)]
                [(FALSE)                              (FalseVal false)]
                [(NULL)                               (NullVal null)]
                [(STRING)                             (StrVal $1)]
                [(NUMBER)                             (NumVal $1)]
                [(LEFTBRACKET elems RIGHTBRACKET)     (Array $2)]
                [(LEFTBRACE members RIGHTBRACE)       (ObjVal $2)])

          (elems [( )                                 (emptyVal )]
                 [(expr)                              (list $1)]
                 [(elems COMMA elems)                 (cons $1 $3)])

          (member [(STRING COLON expr)                (StrJSONPair $1 $3)]
                  [( )                                (emptyVal )])

          (members [(member)                          (list $1)]
                   [(member COMMA members)            (cons $1 $3)])
                   
          )


          ))

(define (parse in)
  (myparser (get-tokenizer in)))

(define (parsestr str)
  (let ([in (open-input-string str)])
    (parse in)))

(define (parsefile filename)
  (let ([in (open-input-file filename)])
    (parse in)))

(check-expect (parsestr "true") (TrueVal #t))
(check-expect (parsestr "false") (FalseVal #f))
(check-expect (parsestr "null") (NullVal '()))
(check-expect (parsestr "2019") (NumVal 2019))
(check-expect (parsestr "\"Love Programming Language\"") (StrVal "\"Love Programming Language\""))
(check-expect (parsestr "\"Fall 2019\"") (StrVal "\"Fall 2019\""))
(check-expect (parsestr "\"(true)\"") (StrVal "\"(true)\""))
(check-expect (parsestr "[true]") (Array (list (TrueVal #t))))

#|
;(parsefile "cdc2018.json")

(define json-array2 (list 1 2 3 4 5))
(define rjson2 "[{\"name\":\"Nate\",\"email\":\"nathanl@ccs.neu.edu\"}]")
(define json-array3 (list 1 "hi" false (list "foo" 10)))


json-array2
json-array3
rjson2
|#