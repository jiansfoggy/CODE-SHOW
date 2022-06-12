#lang racket

;(require racket/gui)
(require racket/gui/dynamic)

; bunch of things to make drawing work
(define gui:make-screen-bitmap (gui-dynamic-require 'make-screen-bitmap))
(define gui:bitmap-dc% (gui-dynamic-require 'bitmap-dc%))
(define gui:frame% (gui-dynamic-require 'frame%))
(define gui:canvas% (gui-dynamic-require 'canvas%))
(define gui:panel% (gui-dynamic-require 'panel%))
(define gui:color% (gui-dynamic-require 'color%))
(define gui:make-color (gui-dynamic-require 'make-color))
(define gui:make-pen (gui-dynamic-require 'make-pen))

#;(require test-engine/racket-tests)

; I think this is all we need to provide?
(provide dbncolor dbncolor-grey run-paper-sim set-pen-color! clear-paper
         draw-point draw-line get-pixel-color get-pen-color
         get-mouse-x get-mouse-y get-mouse-button get-key get-time refresh
         set-antialias dbncolor->greyscale)
         


; simple structure to represent colors from design by numbers
; The parameter, grey, is rounded to the nearest integer between 0 and 100.
(struct dbncolor (grey) #:transparent
  #:guard (lambda (grey type-name)
            ; I clamp this between 0 and 100
            (if (number? grey) (max 0 (min 100 grey))
                (raise-argument-error 'dbncolor "a number between 0 - 100" 0 grey))))

#;(check-error (dbncolor 1000))


(define PAPER-WIDTH (make-parameter 100))
(define PAPER-HEIGHT (make-parameter 100))

; defines the default bitmap we will modify 
(define current-paper% (gui:make-screen-bitmap (PAPER-WIDTH) (PAPER-HEIGHT)))

; define a new bitmap dc, using the bitmap initialization constructor,
; this allows us to draw onto the canvas (so it sticks) and into the bitmap
(define current-dc% (new gui:bitmap-dc% (bitmap current-paper%)))

; this defines the actual window 
(define paper-frame% (new gui:frame%
                   [label "DBN"]
                   [width (PAPER-WIDTH)]
                   [height (PAPER-HEIGHT)]
                   [style '(float hide-menu-bar)]))


(define (get-mouse-button)
  mouse-button)

(define (get-mouse-x)
  mouse-x)

(define (get-mouse-y)
  mouse-y)
  
(define mouse-x 0)
(define mouse-y 0)
(define mouse-button 0)

; the beginning of adding mouse recording
(define (handle-mouse-event event)
  (set! mouse-x (send event get-x))
  (set! mouse-y (send event get-y))
  ;(printf "Mouse (~a, ~a)" mouse-x mouse-y)
  (set! mouse-button (if (eq? (send event button-down?) #t) 100 0)))

(define keys (make-vector 27 0))

(define (set-key! loc val)
  (printf "set key ~a to ~a\n" loc val)
  (vector-set! keys loc val))

(define (get-key val)
  (vector-ref keys val))

(define (get-time sym)
  (let* ([the-seconds (current-seconds)]
         [the-date (seconds->date the-seconds)])
    (cond
      [(eq? sym 'hour) (date-hour the-date)]
      [(eq? sym 'minutes) (date-minute the-date)]
      [(eq? sym 'seconds) (date-second the-date)]
      [(eq? sym 'milliseconds) (modulo (current-milliseconds) 1000)]
      [else (raise-argument-error 'getTime "argument must be 'hour, 'minutes, 'seconds, or 'milliseconds" 0 sym)])))
      
; the beginning of adding keyboard recording
(define (handle-key-event event)
  (let ([press (send event get-key-code)]
        [release (send event get-key-release-code)])
    (cond
      [(char? press) (let ([val (- (char->integer (char-upcase press)) 64)])
                         (cond
                           [(and (>= val 0) (<= val 26)) (set-key! val 100)]))

                     ]
      [(char? release) (let ([val (- (char->integer (char-upcase release)) 64)])
                           (cond
                             [(and (>= val 0) (<= val 26)) (set-key! val 0)]))])))
      

; this creates a class that inherits from canvas but passes the handling
; of mouse end keyboard events to a couple of helper functions--we'll create
; the canvas from this instead of from the canvas% object
(define event-handling-canvas%
  (class gui:canvas% ; base class
    (define/override (on-event event)
      (handle-mouse-event event))
    (define/override (on-char event)
      (handle-key-event event))
    (super-new)))

; create a panel so we can keep the minimum size of the window to the paper size
(define main-panel% (new gui:panel% [parent paper-frame%]
     [style '(border)]
     [min-width (PAPER-WIDTH)]
     [min-height (PAPER-HEIGHT)]))

; defines the canvas we draw on--this sets up the callback to
; call draw-bitmap on the dc whenever the canvas needs to be painted
(define paper-canvas% (new event-handling-canvas% [parent main-panel%]
     [paint-callback
      (lambda (canvas dc)
        (send dc draw-bitmap current-paper% 0 0))]))

; int int dbncolor -> void
; draws a point in a specific dbncolor on the given coordinate
(define (draw-point x y col)
  (cond
    [(dbncolor? col)
     (let ([last-pen (send current-dc% get-pen)])
       (send current-dc% set-pen (color%->pen% (dbncolor->color% col)))
       #;(printf "last-pen: ~a (color: ~a)~n" last-pen col)
       (send current-dc% draw-point x (- (PAPER-HEIGHT) y))
       (send current-dc% set-pen last-pen)
       #;(printf "last-pen: ~a (color: ~a)~n" (send current-dc% get-pen) col)
       #;(send current-dc% flush))]
    [else raise-argument-error 'draw-point "dbncolor?" 2 x y col]))

(define (refresh)
  (send paper-canvas% refresh-now))

; int, int, int, int -> void
; draws a line with the current pen color from x, y to x1, y1.
(define (draw-line x y x1 y1)
  (send current-dc% draw-line x (- (PAPER-HEIGHT) y) x1 (- (PAPER-HEIGHT) y1))
  #;(send current-dc% flush))


; turns on or off antialiasing
(define (set-antialias val)
  (send current-dc% set-smoothing val))

; erases the current paper with the given color
(define (clear-paper col)
  (cond
    [(dbncolor? col) (let ([background-color (dbncolor->color% col)])
                       (send current-dc% set-background background-color)
                       (send current-dc% clear)
                       (send paper-canvas% refresh-now))]
    [else raise-argument-error 'clear-paper "dbncolor?" 0 col]))

; sets the current pen color for drawing things
; dbncolor -> void
(define (set-pen-color! col)
  (cond
    [(dbncolor? col) (send current-dc% set-pen (color%->pen% (dbncolor->color% col)))]
    [else raise-argument-error 'set-pen-color! "dbncolor?" 0 col]))

; returns the current pen color for drawing things on the canvas
; -> dbncolor
(define (get-pen-color)
  (color%->dbncolor (send (send current-dc% get-pen) get-color)))

; launches a window with the backing bitmap
(define (run-paper-sim)
  (send paper-frame% show #t))

; returns the pixel at a given point
; int, int -> dbncolor
(define (get-pixel-color x y)
  (let ([col% (make-object gui:color%)])
    (if (send current-dc% get-pixel x (- (PAPER-HEIGHT) y) col%)
        (let ([red (send col% red)]
              [green (send col% green)]
              [blue (send col% blue)])
          #;(printf "rbg: (~a, ~a, ~a)~n" red green blue)
          (dbncolor-grey (color%->dbncolor col%)))
        (raise-argument-error 'get-pixel-color "x and y do not refer to a valid pixel" 0 x y))))
    

; dbncolor -> color%
; creates a dbncolor from the range 0-100 to be a greyscale
; color% used by racket gui
(define (dbncolor->color% col)
  (let ([x (- 255 (min 255 (max 0 (exact-round (* (dbncolor-grey col) 2.55)))))])
    (gui:make-color x x x 1)))

; color -> dbncolor
; convert an racket gui color% into a dbncolor
(define (color%->dbncolor col)
  (let ([red (send col red)]
        [green (send col green)]
        [blue (send col blue)])
    ; luminosity conversion for greyscale: 0.21 R + 0.72 G + 0.07 B.
    (dbncolor (- 100 (min 100 (max 0 (exact-round
              (* (+ (* .21 (/ red 255)) (* .72 (/ green 255)) (* .07 (/ blue 255))) 100))))))))

; converts a color% to a pen%, which is needed for drawing in the gui
(define (color%->pen% col)
  (gui:make-pen #:color col #:width 1))

; convert a dbncolor to the grey scale value in the interval [0, 100]
(define (dbncolor->greyscale col)
  (dbncolor-grey col))
              

#;(test)