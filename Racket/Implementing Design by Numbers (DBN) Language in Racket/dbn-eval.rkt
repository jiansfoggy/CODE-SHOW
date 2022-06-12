#lang racket

(require "dbn-ast.rkt" "dbn-parser.rkt" "papersim.rkt" "dbn-env.rkt" "dbn-errors.rkt")


; generic evaluate from an input port
(define (eval-dbn in [env #f] [run-sim #t] [clear? #t])
  ; reset errors before we move on
  (reset-errors!)
  ; this creates an ast from in
  (let ([prog (parse in)])
    (if (not (parser-error))
        ; see if we should run the sim
        (begin
          (when (eq? run-sim #t)
            (run-paper-sim))
          (set-antialias 'aligned)
          ; evaluate the statements
          (eval-statements (empty-env) (program-statements prog))
          ; refresh to make sure things draw
          (refresh))
        (error "quitting due to parse error"))))

; evaluate a file
(define (eval-file filename [env #f] [run-sim #t] [clear? #t])
  (let ([in (open-input-file filename)])
    (run-paper-sim)
    (eval-dbn in env run-sim clear?)
    (refresh)))

; evaluate a string
(define (eval-str str)
  (let ([in (open-input-string str)])
    (eval-dbn in)))

; evaluates all the files under a directory
(define (eval-dir dirname sleeplen [env #f] [run-sim #t] [clear? #t])
  (run-paper-sim)
  (fold-files (λ (path kind acc)
                (cond
                  [(eq? kind 'file)
                   (printf "Evaluating ~a~n" path)
                   (eval-file path env #t clear?)
                   (refresh)
                   (printf "Enter for next one:")
                   (read (current-input-port))])) #f dirname #f))

; evaluate the list of statements
(define (eval-statements env statements [slow? #f])
  ; this is simply a fold, but we pass the environment along
  ; in the accumulator so we can update it from statement to statement
  ; as needed, this means sometimes we have to keep it! notice that
  ; foldl takes the arguments opposite of foldl in Haskell (accumulator
  ; comes last, not first
  (unless (not slow?) (sleep slow?))
  (foldl-and-exit (λ (s e) (eval-statement e s #f)) env statements))
  

; this is like foldl, but you can exit
(define (foldl-and-exit fun acc lst)
  ; if the list is empty, return the accumulator
  (if (null? lst)
      acc
      ; otherwise, recurse
      (let* ([el (first lst)]
             [result (fun el acc)])
        (match result
          [(cons 'exit v) v]
          [else (foldl-and-exit fun result (rest lst))]))))

; This function evaluates statements, but it also accumulates the
; environment, meaning that it will pass the environment from fun
; to fun
(define (eval-statement env statement [slow? #f])
  (unless (not slow?) (sleep slow?))
  (match statement
    ; Paper 
    [(paper-expr exp xs ys) 
     (clear-paper (dbncolor (eval-expr env exp))) env]    
    ; Pen
    [(pen-expr exp)
     (set-pen-color! (dbncolor (eval-expr env exp))) env]
    ; Print
    [(print-expr exp) (printf "~a~n" (eval-expr env exp)) env]

    ; TODO: Add Line expressions
    [(line-expr x1 y1 x2 y2)  (draw-line (eval-expr env x1) (eval-expr env y1)
                                         (eval-expr env x2) (eval-expr env y2)) env]

    [(load-expr filename) (eval-file (~a "./ExtractPrograms/" filename)) env]
    
    ; Assignment to a paper location, this is a special case
    [(assignment-expr (get-paper-loc x y) color)
     (let ([xcoord (eval-expr env x)]
           [ycoord (eval-expr env y)]
           [col (eval-expr env color)])
       (draw-point xcoord ycoord (dbncolor col)))
     env]
       

    ; Assignment to a variable name, need to see if it's there first
    ;;; TODO: Add variable assignment, this requires using the environment
    ;;;       to see if it's there and creating it if it's not

    [(assignment-expr (var-expr sym) expr) (let ([var (apply-env env sym)]
                                                 [val (eval-expr env expr)])
                                             (if var
                                                 (begin (setref! var val) env)
                                                 (extend-env env sym val))
                                             )]
    ; the antialias expression, for setting up antialias stuff
    [(antialias-expr expr)
     (let ([val (eval-expr env expr)])
       (cond [(= val 0) (set-antialias 'aligned)]
             [(< val 0) (set-antialias 'unaligned)]
             [else (set-antialias 'smoothed)]))]

    ; Repeat!
    [(repeat-expr sym from to body)
     (let* ([start (eval-expr env from)] ; evaluate the start value
            [end (eval-expr env to)]     ; then the ending value
            [newenv (extend-env env sym start)]
            [ref (apply-env newenv sym)])
       (letrec ([loop (λ () (cond [(<= (deref ref) end)                                   
                                   (eval-statements newenv body slow?)
                                   (setref! ref (add1 (deref ref)))
                                   ;(printf "repeat from ~a to ~a~n" (deref ref) end)
                                   (refresh)
                                   (loop)]))])
         (loop)
         env))]

    ; Forever loops
    [(forever-expr body)
     ; just loop forever, reuse the env from when we entered the loop
     (letrec ([loop (λ ()
                      (eval-statements env body slow?)
                      (refresh)
                      (loop))])
       (loop)
       env)]

    ; boolean-like expressions--we execute the body depending on whether or not they return true
    ; Same? 
    [(same-expr expr1 expr2 body)
     (let ([val1 (eval-expr env expr1)]
           [val2 (eval-expr env expr2)])       
       (when (equal? val1 val2) (eval-statements env body slow?))
       env)]
    
    ; NotSame?
    [(not-same-expr expr1 expr2 body)
     (let ([val1 (eval-expr env expr1)]
           [val2 (eval-expr env expr2)])
       (when (not (equal? val1 val2)) (eval-statements env body slow?))
       env)]

    ; Smaller?
    [(smaller-expr expr1 expr2 body)
     (let ([val1 (eval-expr env expr1)]
           [val2 (eval-expr env expr2)])
       (when (< val1 val2) (eval-statements env body slow?))
       env)]

    ; NotSmaller?
    [(not-smaller-expr expr1 expr2 body)
     (let ([val1 (eval-expr env expr1)]
           [val2 (eval-expr env expr2)])
       (when (not (< val1 val2)) (eval-statements env body))
       env)]

    ; Value statement, this is like a return, so we don't need to
    ; pass on the current environment since it should really be the
    ; last thing done in a list of statements
    [(value-expr expr) (cons 'exit (eval-expr env expr))]
    
    
    ; to create a function, we need to create a closure and store it in the
    ; current environment, so a new environment will be passed on here
    ;;; TODO
    [(command-fun sym params body)      (let ([commend-v (closure sym params body env)])
                                          (extend-env env sym commend-v))]
    
    ; and we do the same thing for the numbers
    ;;; TODO (Achievement) [(number-fun sym params body)
    [(number-fun sym params body)       (let ([number-v (closure sym params body env)])
                                          (extend-env env sym number-v))]

    ;;; TODO: now for expressions as statements, these we ignore the return value of
    ; application as statements, I've left some comments to help you along
    ; [(apply-expr sym exprs)  
     ; evaluate all the arugments, then call the function
         ; make sure we found it, or return an error otherwise
         ; return the previous environment to be carried along
    [(apply-expr sym exprs)    (letrec ([gene_env (lambda (para_names rslt_argu old_env)
                                                (if (or (null? para_names) (null? rslt_argu))
                                                    old_env
                                                    (gene_env (cdr para_names) (cdr rslt_argu)
                                                          (extend-env old_env (car para_names) (eval-expr old_env (car rslt_argu))))
                                                    ))]
                                        )
                                 (begin (if (apply-env env sym)
                                            (eval-statements (gene_env (closure-params (deref (apply-env env sym)))
                                                                   exprs env)
                                                             (closure-body (deref (apply-env env sym))))
                                            (error "We don't have this function, " sym ", in current environment."))
                                        env)
                                 )]
    ))


(define (eval-expr env expr)
  (match expr
    ; literal numbers
    [(numeric-expr a) a]

    ; variable lookups
    [(var-expr sym) (let [(val (apply-env env sym))]
                      (if val
                          (deref val)
                          (error "undefined variable " sym)))]
    
    ; used as an expression, the paper location returns the color
    [(get-paper-loc x y)
     (let* ([xcoord (eval-expr env x)]
            [ycoord (eval-expr env y)]
            [color (get-pixel-color xcoord ycoord)])
       color)]

    ; math operations
    [(add-expr a b) (+ (eval-expr env a) (eval-expr env b))]
    [(sub-expr a b) (- (eval-expr env a) (eval-expr env b))]
    [(div-expr a b) (/ (eval-expr env a) (eval-expr env b))]
    [(mult-expr a b) (* (eval-expr env a) (eval-expr env b))]

    ; read mouse info
    [(mouse-expr expr)
     (let ([val (eval-expr env expr)])
       (cond
         [(= val 1) (get-mouse-x)]
         [(= val 2) (get-mouse-y)]
         [(= val 3) (get-mouse-button)]
         [else (error "Expected a mouse value from 1 to 3 inclusive, got " val " instead.")]))]

    ; read key info
    [(key-expr expr)
     (let ([val (eval-expr env expr)])
       ; uh, this seems a little limiting, we can only read 26 keys?
       (cond [(and (>= val) (<= val 26)) (get-key val)]
             [else (error "Expected a key range from 1 to 26 inclusive, got " val " instead.")]))]
    

    ; time expressions
    [(time-expr expr)
     (let ([val (eval-expr env expr)])
       (cond [(= val 1) (get-time 'hour)]
             [(= val 2) (get-time 'minutes)]
             [(= val 3) (get-time 'seconds)]
             [(= val 4) (get-time 'milliseconds)]
             [else (error "Expected a Time range from 1 to 4 inclusive, got " val " instead")]))]
                        

    ; handle function application as an expression, these we care about the return value
    ;;; TODO: function application as an expression (not a statement)--you should return
    ; the result of the evaluation of all the statements in the body
    ; [(apply-expr sym exprs)
     ; evaluate all the arugments, then call the function
         ; make sure we found it, or return an error otherwise
                  ; grab the closure from the environment, which has parameters
                     ; then evaluate all the statements and return the result
    [(apply-expr sym exprs)    (letrec ([gene_env (lambda (para_names rslt_argu old_env)
                                                (if (or (null? para_names) (null? rslt_argu))
                                                    old_env
                                                    (gene_env (cdr para_names) (cdr rslt_argu)
                                                          (extend-env old_env (car para_names) (eval-expr old_env (car rslt_argu))))
                                                    ))]
                                        )
                                 (begin (if (apply-env env sym)
                                            (eval-statements (gene_env (closure-params (deref (apply-env env sym)))
                                                                   exprs env)
                                                             (closure-body (deref (apply-env env sym))))
                                            (error "We don't have this function, " sym ", in current environment."))
                                        env)
                                 )]
    
    ))

; error file
; (eval-file "./EP/Program-ch12-p147-1.dbn")
; (eval-file "./ExamplePrograms/  ")
; (eval-dir "./ExamplePrograms/" 10)
; (eval-dir "./EP/" 10)
; (eval-dir "./EP1/" 10)
; (eval-dir "/Users/sunjian/Public/Document/3351/Week10/dbn-assignment/EP2/" 2)
; (eval-file "./EP2/Program-ch14-p170-1.dbn")
; Program-ch12-p147-1.dbn
; (eval-file "/Users/sunjian/Public/Document/3351/Week10/dbn-assignment/ExamplePrograms/Program-ch12-p147-1.dbn")
; (eval-file "./EP1/Program-ch13-p160-1.dbn")
; Program-ch13-p161-1.dbn

; (pathlist-closure '("/Users/sunjian/Public/Document/3351/Week10/dbn-assignment/ExamplePrograms/"))
; (define Chart_List (pathlist-closure '("/Users/sunjian/Public/Document/3351/Week10/dbn-assignment/ExamplePrograms/")))
; (list-ref Chart_List 9)
; (make-directory* "/Users/sunjian/Public/Document/3351/Week10/dbn-assignment/ExamplePrograms/")