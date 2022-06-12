#lang racket

(provide (all-defined-out))
; simple implementation of environments using references

; references in our symbol table
(struct memref (sym [value #:mutable]) #:transparent)

; dereferences the reference
(define (deref ref)
  (memref-value ref))

; sets the reference
(define (setref! ref val)
  (set-memref-value! ref val))

; extends the environment with a key/value pair,
; and returns the new environment
(define (extend-env env k v)
  (let ([ref (memref k v)])
    (cons (cons k ref) env)))

; creates a new environment
(define (empty-env) '())

; apply environment looks for var in the env
(define (apply-env env var)
  (let ([res (assoc var env)])
    (if res
        (cdr res)
        #f)))


;;; closures for functions
;;; we can store closures in addition to values in the environment,
;;; note, it's useful to store the name (sym) with the closure for errors and such
(struct closure (sym params body env) #:transparent)