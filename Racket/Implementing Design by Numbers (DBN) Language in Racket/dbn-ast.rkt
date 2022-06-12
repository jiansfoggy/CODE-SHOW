#lang racket

;; this module contains all the structs needed to create the abstract
;; syntax of the DBN language
(provide (all-defined-out))

;;; these are the expressions for paper, pen and line
(struct paper-expr (value xsize ysize) #:transparent)
(struct pen-expr (value) #:transparent)
(struct line-expr (x1 y1 x2 y2) #:transparent)

;; numeric expressions
(struct numeric-expr (value) #:transparent)

;; var identifiers
(struct var-expr (name) #:transparent)

; paper location structs
(struct set-paper-loc (x y color) #:transparent)
(struct get-paper-loc (x y) #:transparent)
(struct antialias-expr (value) #:transparent)

; set, which works on variables and color references
(struct assignment-expr (e1 e2) #:transparent)

; iterations
(struct repeat-expr (var start end body) #:transparent)
(struct forever-expr (body) #:transparent)



; predicate expressions
(struct same-expr (e1 e2 body) #:transparent)
(struct not-same-expr (e1 e2 body) #:transparent)
(struct smaller-expr (e1 e2 body) #:transparent)
(struct not-smaller-expr (e1 e2 body) #:transparent)

; structs for defining functions and procedures,
; these have a name a list of parameters and a body
(struct command-fun (name params body) #:transparent)
(struct number-fun (name params body) #:transparent)

; this is the equivalent of a 'return v' in another language
(struct value-expr (value) #:transparent)

; just prints to the standard output
(struct print-expr (value) #:transparent)

; function application, empty for commands
(struct apply-expr (fun-name params) #:transparent)

; things related to the external world, time, mouse, etc
(struct mouse-expr (value) #:transparent)
(struct key-expr (value) #:transparent)
(struct time-expr (value) #:transparent)

; compound expressions
(struct add-expr (e1 e2) #:transparent)
(struct sub-expr (e1 e2) #:transparent)
(struct mult-expr (e1 e2) #:transparent)
(struct div-expr (e1 e2) #:transparent)

; loading
(struct load-expr (filename) #:transparent)

; represents an entire program
(struct program (statements) #:transparent)