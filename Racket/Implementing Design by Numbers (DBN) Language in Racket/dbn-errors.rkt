#lang racket

(define lexer-error (make-parameter #t))
(define parser-error (make-parameter #t))
(define eval-error (make-parameter #t))

(define (reset-errors!)
  (lexer-error #f)
  (parser-error #f)
  (eval-error #f))

(provide (all-defined-out))