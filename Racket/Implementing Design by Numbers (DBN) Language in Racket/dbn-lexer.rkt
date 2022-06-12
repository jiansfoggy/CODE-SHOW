#lang racket

; needed for parsing stuff
(require parser-tools/lex
         
         ; this last gives us prettier names for common regular expression stuff,
         ; and also renames it so they're all prefixed with ':' in their names
         (prefix-in : parser-tools/lex-sre)
         "dbn-errors.rkt")

; just output everything
(provide (all-defined-out))

; now, define the tokens that the parser will use
(define-tokens names-and-values (NUMERICVALUE
                                 IDENTIFIER
                                 FILENAME
                                 ))

; this is for the end of file marker
(define-empty-tokens end-of-file (EOF))


; these are all the keywords
(define-empty-tokens keywords (PAPER
                               PEN
                               LINE
                               SET
                               REPEAT
                               FOREVER
                               SAME
                               NOTSAME
                               SMALLER
                               NOTSMALLER
                               MOUSE
                               LOAD
                               NUMBER
                               VALUE
                               KEY
                               NET
                               TIME
                               PRINT
                               COMMAND
                               ANTIALIAS
                               NEWLINE))

; these are the math operators
(define-empty-tokens math-operators (ADDITION
                                     SUBTRACTION
                                     MULTIPLICATION
                                     DIVISION))


; these are the parentheses variations
(define-empty-tokens parentheses (LEFTPARENTHESIS
                                  RIGHTPARENTHESIS
                                  LESSTHAN
                                  GREATERTHAN
                                  LEFTBRACE
                                  RIGHTBRACE
                                  LEFTBRACKET
                                  RIGHTBRACKET))

; and finally the lexer, this returns a function that takes an input port
(define dbnlexer
  (lexer-src-pos
   [#\+                                             (token-ADDITION)]
   [#\-                                             (token-SUBTRACTION)]
   [#\*                                             (token-MULTIPLICATION)]
   [#\/                                             (token-DIVISION)]
   [#\(                                             (token-LEFTPARENTHESIS)]
   [#\{                                             (token-LEFTBRACE)]
   [#\[                                             (token-LEFTBRACKET)]
   [#\<                                             (token-LESSTHAN)]
   [#\)                                             (token-RIGHTPARENTHESIS)]
   [#\}                                             (token-RIGHTBRACE)]
   [#\]                                             (token-RIGHTBRACKET)]
   [#\>                                             (token-GREATERTHAN)]
   [(:+ #\newline)                                  (token-NEWLINE)]
   ;;; TODO: add Paper, Pen, Line and Set
   [(:or "PAPER" "Paper")                           (token-PAPER)]
   [(:or "PEN" "Pen")                               (token-PEN)]
   [(:or "LINE" "Line")                             (token-LINE)]
   [(:or "SET" "Set")                               (token-SET)]
   [(:or "REPEAT" "Repeat")                         (token-REPEAT)]
   [(:or "FOREVER" "Forever")                       (token-FOREVER)]
   [(:or "SAME?" "Same?")                           (token-SAME)]
   [(:or "NOTSAME?" "Notsame?" "NotSame?")          (token-NOTSAME)]
   [(:or "SMALLER?" "Smaller?")                     (token-SMALLER)]
   [(:or "NOTSMALLER?" "Notsmaller?" "NotSmaller?") (token-NOTSMALLER)]
   [(:or "MOUSE" "Mouse")                           (token-MOUSE)]
   [(:or "LOAD" "Load")                             (token-LOAD)]
   [(:or "NUMBER" "Number")                         (token-NUMBER)]
   [(:or "VALUE"  "Value")                          (token-VALUE)]
   [(:or "KEY" "Key")                               (token-KEY)]
   [(:or "NET" "Net")                               (token-NET)]
   [(:or "TIME" "Time")                             (token-TIME)]
   [(:or "PRINT" "Print")                           (token-PRINT)]
   [(:or "COMMAND" "Command")                       (token-COMMAND)]
   [(:or "NUMBER" "Number")                         (token-NUMBER)]
   [(:or "ANTIALIAS" "Antialias")                   (token-ANTIALIAS)]
   
   ;;; TODO: Add numbers, which should be a token-NUMERICVALUE and contain an actual number, not a string
   [(:or (:or (:or #\0 (char-range #\1 #\9)) (:: (char-range #\1 #\9) (:+ (:or #\0 (char-range #\1 #\9)))))
         (:or "" (:: (:* (:or #\0 (char-range #\1 #\9))) #\. (:+ (:or #\0 (char-range #\1 #\9)))))
         (:or "" (:: (:* (:or #\0 (char-range #\1 #\9))) (:* (:: #\. (:+ (:or #\0 (char-range #\1 #\9))))) "E" (:or "" #\+ #\-) (:+ (:or #\0 (char-range #\1 #\9))))
                 (:: (:* (:or #\0 (char-range #\1 #\9))) (:* (:: #\. (:+ (:or #\0 (char-range #\1 #\9))))) "e" (:or "" #\+ #\-) (:+ (:or #\0 (char-range #\1 #\9))))))
                                                    (token-NUMERICVALUE (string->number lexeme))]

   [(:: (:+ (:or upper-case lower-case #\_ #\- #\/ numeric)) ".dbn")
                                                    (token-FILENAME lexeme)]

   ; identifiers
   ;;; TODO: Add identifiers, which should be a token-IDENTIFIER and contain the lexeme
   ; [(:: (:+ alphabetic) (:* (:or numeric alphabetic)))
   [(:: (:+ alphabetic) (:* (:or numeric alphabetic)))
                                                    (token-IDENTIFIER (string->symbol lexeme))]
   ; comments
   [(:: "//" (:* (char-complement (:or #\newline #\linefeed)))
        (:+ (:or #\newline #\linefeed))) (return-without-pos (dbnlexer input-port))]

   
   ; handle a lang line so we ignore it
   [(:: "#lang" (:+ (union #\space #\tab)) (union "dbn" "dbn-lang")) (return-without-pos (dbnlexer input-port))]

   ; ignore whitespace
   [whitespace (return-without-pos (dbnlexer input-port))]

   ; good, ole eof
   [(eof) (token-EOF)]

   ; anything else is a syntax error, so report it as such
   [any-char (raise-lex-error start-pos lexeme)]))



; position -> string -> error
; raises a lexing error
(define (raise-lex-error pos lexeme)
  (let* ([linenums? (not (eq? (position-line pos) #f))]
         [loc (if linenums? (position-line pos) (position-offset pos))]
         [col (position-col pos)]
         [partial-msg (string-append (if linenums? "syntax error at line "
                                         "syntax error at offset ") (number->string loc))]
         [msg (string-append partial-msg (if linenums? (string-append ", col " (number->string col)) "")
                             ": '" lexeme "'")])
    (lexer-error #t)
    (raise-syntax-error 'dbnlexer msg)))


; input port -> thunk
; creates a thunk that when called will return the next token from the input stream
(define (get-tokenizer in)
  (λ () (dbnlexer in)))


; input port -> list of tokens
; this function takes an input port and returns a list of
; tokens read from it (until it hits eof)
(define (lex in)
  (port-count-lines! in)
  (let ([tokenize (get-tokenizer in)])
    (define (lexfun)
      (let ([tok (tokenize)])
        (cond
          ; test to see if we hit eof as the base case
          [(eq? (position-token-token tok) (token-EOF)) null]
          [else (cons (position-token-token tok) (lexfun))])))
    (lexfun)))


; string -> list of tokens
; this function takes a string and returns a list of
; tokens read from it (until it reaches the end)
(define (lex-str str)
  (lex (open-input-string str)))

; filename -> list of tokens
; this function takes a filename, opens it as an input port,
; and then reads tokens until the end is reached
(define (lex-file filename)
  (lex (open-input-file filename)))
