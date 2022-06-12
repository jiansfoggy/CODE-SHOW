#lang racket

;; Jian Sun
;; 873397832

;; comments
;; Firstly, we defined different token names in program. For example, null,
;; true, false, eof, dot, comma, string, number, leftparen and so on with
;; define-empty-tokens and define-tokens.
;; Then we define those values in mylexser function.
;; true, false and null are easy
;; For string, it is consisting of quote, characters and quote.
;; For number, it has integer, fraction, exponent.
;; For array, it is made by brackets and true or
;; false or null or number or string and comma between each two values.
;; For object, it is made by braces and string and colon and any values.
;; Next we write function, get-tokenizer, lex, lexstr to help to read input value
;; and explain them.
;; We tried 16 tests and passed all of them. We also tested the json file,
;; cdc2018.json, and get the correct answer. But due to the file size,
;; it is time-consuming.
;; So our code is ok to run.

(require parser-tools/lex
         (prefix-in : parser-tools/lex-sre))
(require test-engine/racket-tests)

(provide (all-defined-out))

(define-empty-tokens null-values (NULL))
(define-empty-tokens end-of-file (EOF))
(define-empty-tokens sign-value (SIGN))
(define-empty-tokens bool-values (TRUE FALSE))
(define-empty-tokens lambda-def (LAMBDA DOT QUOTE COMMA COLON))
(define-empty-tokens bool-operators (AND OR NOT))
(define-empty-tokens parens (LEFTPAREN
                             RIGHTPAREN
                             LEFTBRACE
                             RIGHTBRACE
                             LEFTBRACKET
                             RIGHTBRACKET))

(define-tokens names-and-value (NUMBER STRING ARRAY OBJECT))

(define mylexer
  (lexer
   [(eof)                                                                                       (token-EOF)]
   [whitespace                                                                                  (mylexer input-port)]
   [#\.                                                                                         (token-DOT)]
   [#\,                                                                                         (token-COMMA)]
   [#\"                                                                                         (token-QUOTE)]
   [#\:                                                                                         (token-COLON)]
   [(:or #\λ "lambda")                                                                          (token-LAMBDA)]
   [#\[                                                                                         (token-LEFTBRACKET)]
   [#\]                                                                                         (token-RIGHTBRACKET)]
   [#\{                                                                                         (token-LEFTBRACE)]
   [#\}                                                                                         (token-RIGHTBRACE)]
   [#\(                                                                                         (token-LEFTPAREN)]
   [#\)                                                                                         (token-RIGHTPAREN)]

   ["true"                                                                                      (token-TRUE)]
   ["false"                                                                                     (token-FALSE)]
   ["null"                                                                                      (token-NULL)]

   [(:: (:: #\" (:+ (:or (complement (:: (:* any-string) #\" (:* any-string)))) (:: #\\ #\") ) #\"))
                                                                                                (token-STRING lexeme)]

   [(:or "" #\+ #\-)                                                                            (token-SIGN)]
   ;[(:or (char-range #\1 #\9))                                                                  (token-ONENINE (string->number lexeme))]    
   ;[(:or #\0 (char-range #\1 #\9))                                                              (token-DIGIT (string->number lexeme))]
   ;[(:+ (:or #\0 (char-range #\1 #\9)))                                                         (token-DIGITS (string->number lexeme))]

   ;[(:or "" (:: (:+ (:or #\0 (char-range #\1 #\9))) #\. (:+ (:or #\0 (char-range #\1 #\9)))))   (token-FRACTION (string->number lexeme))]
   ;[(:or (:or #\0 (char-range #\1 #\9)) (:: (char-range #\1 #\9) (:+ (:or #\0 (char-range #\1 #\9))))
   ;      (:: #\- (:or #\0 (char-range #\1 #\9))) (:: #\- (char-range #\1 #\9) (:+ (:or #\0 (char-range #\1 #\9)))))
   ;                                                                                             (token-INTEGER lexeme)]
   ;[(:or "" (:: "E" (:or "" #\+ #\-) (:+ (:or #\0 (char-range #\1 #\9))))
   ;         (:: "e" (:or "" #\+ #\-) (:+ (:or #\0 (char-range #\1 #\9)))))                      (token-EXPONENT lexeme)]
   [(:or (:or (:or #\0 (char-range #\1 #\9)) (:: (char-range #\1 #\9) (:+ (:or #\0 (char-range #\1 #\9))))
         (:: #\- (:or #\0 (char-range #\1 #\9))) (:: #\- (char-range #\1 #\9) (:+ (:or #\0 (char-range #\1 #\9)))))
         (:or "" (:: (:* (:or #\0 (char-range #\1 #\9))) #\. (:+ (:or #\0 (char-range #\1 #\9)))))
         (:or "" (:: (:* (:or #\0 (char-range #\1 #\9))) (:* (:: #\. (:+ (:or #\0 (char-range #\1 #\9))))) "E" (:or "" #\+ #\-) (:+ (:or #\0 (char-range #\1 #\9))))
                 (:: (:* (:or #\0 (char-range #\1 #\9))) (:* (:: #\. (:+ (:or #\0 (char-range #\1 #\9))))) "e" (:or "" #\+ #\-) (:+ (:or #\0 (char-range #\1 #\9))))))
                                                                                                (token-NUMBER (string->number lexeme))]
   ))

(define (get-tokenizer in)
  (lambda () (mylexer in)))

(define (lex in)
  (let ([tokenizer (get-tokenizer in)])
    (define (lex-function)
              (let ([tok (tokenizer)])
                   (cond
                      [(eq? tok (token-EOF)) null]
                      [else (cons tok (lex-function))])))
    (lex-function)))

(define (lexstr str)
  (lex (open-input-string str)))

(define (lexerfile filename)
  (let ([in (open-input-file filename)])
     (lex in)))

(define (lexfile filename)
  (lex (open-input-file filename)))

(check-expect (mylexer (open-input-string "+")) 'SIGN)
(check-expect (mylexer (open-input-string "false")) 'FALSE)
(check-expect (mylexer (open-input-string "true")) 'TRUE)
(check-expect (mylexer (open-input-string "1")) (token-NUMBER 1))
(check-expect (mylexer (open-input-string "2003")) (token-NUMBER 2003))
(check-expect (mylexer (open-input-string "\"he234sdaf\"")) (token-STRING "\"he234sdaf\""))
(check-expect (mylexer (open-input-string "\"11\"")) (token-STRING "\"11\""))
(check-expect (mylexer (open-input-string "null")) 'NULL)
;(string->number "123")