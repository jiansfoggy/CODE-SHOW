#lang racket

(require parser-tools/yacc
         (prefix-in lex: parser-tools/lex)
         "dbn-errors.rkt"
         "dbn-lexer.rkt"
         "dbn-ast.rkt")

(provide (all-defined-out))

(define DEFAULT-PAPER-SIZE 100)


(define dbn-parser
  (parser
   (src-pos)
   (start prog)
   (end EOF)
   (tokens names-and-values end-of-file keywords math-operators parentheses)
   (error (lambda (tok-ok? tok-name tok-value start-pos end-pos)
            ; indicate a parsing error occurred
            (parser-error #t)
            ; and print an error
            ; (unless (and (eq? tok-ok? #t) (eq? tok-name 'EOF))
            (printf "Parsing error at line ~a, col ~a: token: ~a, value: ~a, tok-ok? ~a\n"
                    (lex:position-line start-pos)
                    (lex:position-col start-pos)
                    tok-name
                    tok-value
                    tok-ok?)))
   ; because our function calls are really ID ID ... ID, we make identifiers right
   ; associative, so that they attach more identifiers to the right by default
   (precs (right IDENTIFIER))
   (grammar
    (prog
     [(statements) (program (flatten (filter (λ (el) (not (null? el))) $1)))])

    ; lists of statements
    (statements     
     [(statement maybe-newlines) (list $1)]
     [(statement maybe-newlines statements) (cons $1 $3)])

    ; used to recognize the possibility of some newlines
    (maybe-newlines
     [() null]
     [(NEWLINE maybe-newlines) $2])

    (statement
     ; a statement is one of these many things
     ;;; TODO: Add Paper, Pen, Line, Set and Repeat
     [(PAPER expr)                                           (paper-expr $2 (numeric-expr DEFAULT-PAPER-SIZE) (numeric-expr DEFAULT-PAPER-SIZE))]
     [(PAPER expr expr expr)                                 (paper-expr $2 $3 $4)]
     [(PEN expr)                                             (pen-expr $2)]
     [(SET l-value expr)                                     (assignment-expr $2 $3)]
     [(LINE expr expr expr expr)                             (line-expr $2 $3 $4 $5)]
     [(REPEAT IDENTIFIER expr expr maybe-newlines block)     (repeat-expr $2 $3 $4 $6)]

     ; print, simply prints to the console
     [(PRINT expr ) (print-expr $2)]
     [(VALUE expr) (value-expr $2)]
    
     [(FOREVER maybe-newlines block) (forever-expr $3)]
     [(ANTIALIAS expr) (antialias-expr $2)]
    
     ; now for the predicates and comparisons
     [(SAME expr expr maybe-newlines block) (same-expr $2 $3 $5)]
     [(NOTSAME expr expr maybe-newlines  block) (not-same-expr $2 $3 $5)]
     [(SMALLER expr expr maybe-newlines block) (smaller-expr $2 $3 $5)]
     [(NOTSMALLER expr expr maybe-newlines block) (not-smaller-expr $2 $3 $5)]

     ; and then functions and command definitions
     [(COMMAND IDENTIFIER maybe-newlines block) (command-fun $2 null $4)]
     [(NUMBER IDENTIFIER maybe-newlines block) (number-fun $2 null $4)]
     [(COMMAND IDENTIFIER parameters maybe-newlines block) (command-fun $2 $3 $5)]
     [(NUMBER IDENTIFIER parameters maybe-newlines block) (number-fun $2 $3 $5)]

     ; !!!!!!!!! change this part
     ; loads, but we'll ignore this for now
     [(LOAD FILENAME) (load-expr $2)]
     
     ; function application, well really command application, these MUST
     ; end with a newline, or you'll have a ton of shift/reduce conflicts--at
     ; least you can't do this unless you make two passes so you can figure
     ; out which identifiers are actually functions and not variables!
     [(IDENTIFIER exprs) (apply-expr $1 $2)]
     [(IDENTIFIER) (apply-expr $1 null)]

     [(block) $1]
     )
    
 
    
    ; a block will simply return a list of statements, not a special struct
    (block
     [(LEFTBRACE maybe-newlines statements RIGHTBRACE)
      (flatten (filter (λ (el) (not (null? el))) $3))])

    
    ; parameters are names used for a function
    (parameters
     [(IDENTIFIER) (list $1)]
     [(IDENTIFIER parameters) (cons $1 $2)])

    ; legal l-values in the language
    (l-value
     ;;; TODO: Add variables (they are l-values)
     [(IDENTIFIER)                         (var-expr $1)]
     [(LEFTBRACKET expr expr RIGHTBRACKET) (get-paper-loc $2 $3)])
    
    ; lists of expressions? Really only useful for fun calls
    (exprs
     [(expr) (list $1)]
     [(expr exprs) (cons $1 $2)])
        
    (expr
     [(expr ADDITION term) (add-expr $1 $3)]
     [(expr SUBTRACTION term) (sub-expr $1 $3)]
     [(term) $1])

    (term
     [(term MULTIPLICATION factor) (mult-expr $1 $3)]
     [(term DIVISION factor) (div-expr $1 $3)]
     [(factor) $1])

    (factor
     [(rvalues) $1]
     [(LEFTPARENTHESIS expr RIGHTPARENTHESIS) $2])

    (rvalues
     [(NUMERICVALUE) (numeric-expr $1)]
     ;;; TODO: the second place you need to add variables, because they can also be r-values
     [(IDENTIFIER)                         (var-expr $1)]
     [(LEFTBRACKET expr expr RIGHTBRACKET) (get-paper-loc $2 $3)]
     [(LESSTHAN TIME expr GREATERTHAN) (time-expr $3)]
     [(LESSTHAN MOUSE expr GREATERTHAN) (mouse-expr $3)]
     [(LESSTHAN KEY expr GREATERTHAN) (key-expr $3)]
     [(LESSTHAN IDENTIFIER exprs GREATERTHAN) (apply-expr $2 $3)]))))

;; this function will parse from any input port and
;; return an ast (a program struct)
(define (parse in)
  (port-count-lines! in)
  (dbn-parser (get-tokenizer in)))

;; this function parses a string and returns a program struct
(define (parse-str str)
  (let ([in (open-input-string str)])
    (parse in)))

;; this function opens a file, parses it, and returns a program struct
(define (parse-file filename)
  (let ([in (open-input-file filename)])
    (parse in)))

;; converts an AST to a list, this turns all the structs into lists that
;; contains the name of the struct and its contents
(define (ast->list ast)
  (map (λ (element)
         (cond
           [(struct? element) (ast->list element)]
           [(list? element) (map (λ (x) (ast->list x)) element)]
           [else element]))
       (vector->list (struct->vector ast))))




