;; gorilla-repl.fileformat = 1

;; **
;;; # Gorilla REPL
;;; 
;;; Welcome to gorilla :-)
;;; 
;;; Shift + enter evaluates code. Hit ctrl+g twice in quick succession or click the menu icon (upper-right corner) for more commands ...
;;; 
;;; It's a good habit to run each worksheet in its own namespace: feel free to use the declaration we've provided below if you'd like.
;; **

;; @@
(ns ppp
  (:require [gorilla-plot.core :as plot])
  (:use clojure.repl
        [anglican 
          core runtime emit ]))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; **
;;; 1. Problem Formulation
;;; ============
;;; Goal
;;; ----
;;; Given a function $$ f:[0,1] \to \mathbb{R} $$ we want to find the element @@ x \in \mathbb{R}@@ that maximizes @@f@@. 
;;; In other words, we want to solve the following problem:
;;; $$ \max_{x \in [0,1]}	f(x) $$
;;; 
;;; If we assume that @@ f @@ is differentiable, then one possible way to solve this is by using the known method called **Hill Climbing**.
;;; 
;;; Reminder: Hill-Climbing
;;; ---------
;;; Let @@\frac{df}{dx} @@ be the derivative of @@ f @@. Given a learning rate @@ \alpha @@ and initial value @@ x_0@@ we use the following update rule:
;;; $$ x_1 \leftarrow x_0 + \alpha \frac{df}{dx} $$
;;; $$ x_2 \leftarrow x_1 + \alpha \frac{df}{dx} $$
;;; $$	\vdots	$$
;;; 
;;; However, we want to examine the optimization problem with functions that are not necessarily differentiable. 
;;; We will use some techniques known from variational optimization to go around this problem of @@ f @@ being non-differentiable. With these techniques, we will solve the optimization problem of an alternative function @@ F @@. This function will be differentiable and we can apply hill-climbing onto the optimization problem regarding @@ F @@.
;;; 
;;; 
;; **

;; **
;;; 2. Method of resolution
;;; =======================
;;; Reformulation of the problem:
;;; -----------------------------
;;; Let's define $$ F(m):= \mathbb{E}_{x \sim q_m(x)}[f(x)] $$ with @@ q @@ being some suitable distribution.
;;; Since @@ F @@ is just like some average of @@ f @@ over @@ x@@, the following inequality holds:
;;; 
;;; $$ F \leq \max_{x \in [0,1]}	f(x)  $$
;;; 
;;; Intuitively speaking, the distribution @@ q @@ gives different values of x different weights. Hence, by choosing a "good" distribution that gives the argument @@ x @@ maximizing our target function, we can approximately solve this problem. Hence our new optimization problem is:
;;; 
;;; $$ \max_{m } F(m) $$
;;; 
;;; 
;; **

;; **
;;; Explanation with an example:
;;; ----------------------------
;;; Let us look at an example to enhance our understanding. Remember the Dirac-Distribution @@q_m := \delta_m @@. We know that
;;; 
;;; $$ \mathbb{E}_{x \sim \delta_m (x) }[f(x)]	= f(m) $$
;;; 
;;; Hence:
;;; 
;;; $$ \underset{y}{\operatorname{argmax}} \mathbb{E}_{x \sim \delta_y (x) }[f(x)] = \underset{y}{\operatorname{argmax}}  f(y)	$$
;;; 
;;; Since we want to choose @@ q @@ such that @@ F @@ is differentiable and such that the derivative is easily computable, we will choose the normal distribution @@ x \sim \mathcal{N}(\mu,\,\sigma^{2})@@. 
;;; To simplify this, we will fix @@ \sigma  @@ and optimize w.r.t. @@ \mu @@.
;; **

;; **
;;; Reminder: Normal Distribution
;;; --------
;;; Remember, the probability density function of the normal distribution is given by:
;;; 
;;; $$ p_{ \mu, \sigma } (x ) = \frac{1}{\sigma \sqrt {2\pi }} e^ { \frac{ -  (x - \mu ) ^2 }{ 2\sigma ^2 }} $$
;;; 
;;; Since the mean of the normal distribution gives us the peak of the pdf, we can deduce that finding @@ \mu @@ that maximizes @@ F @@ is approximately the argument @@ x @@ maximizing @@ f @@.
;; **

;; @@
;; TODO Plot.
;; Function: Gaussian Probability Density  Function ~> GPDF
(defn gpdf [m sigma x]
  (* (/ 1 (* (sqrt (* 2 Math/PI)) sigma ))
     (exp (/ (* -1 (pow (- x m) 2) )
             (* 2 (pow sigma 2))  ))))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;ppp/gpdf</span>","value":"#'ppp/gpdf"}
;; <=

;; **
;;; Summing up:
;;; ----------
;;; To sum it up, we have the following optimization problem:
;;; 
;;; $$ \underset{m}{\operatorname{argmax}} \mathbb{E}_{x \sim \mathcal{N}(m,\,\sigma^{2}) }[f(x)]	$$
;;; 
;;; Since we fix @@ \sigma @@, we denote the corresponding pdf with @@ p_m (x) @@ and our objective function  with @@ F(m) @@.
;;; 
;;; 
;;; Luckily, @@ F @@  is differentiable w.r.t @@ m @@ and we can use Hill-climbing to find the optimal @@ m @@.
;;; Let us note first, that for the derivative the following holds:
;;; 
;;; 
;;; $$	\frac{d}{dm} F(m) = \frac{d}{dm} \int f(x) p_m (x) dx = \int \frac{d}{dm} f(x)p_m (x) dx = \int f(x) \frac{d}{dm} p_m (x) dx = \int f(x) \left(\frac{d}{dm} \log p_m (x)  \right) p_m (x) dx	$$
;;; 
;;; 
;;; 
;;; 
;;; 
;;; Hence:
;;; 
;;; 
;;; $$	\frac{d}{dm} F(m) = \mathbb{E}_{x \sim p_m (x) }\left[f(x) \frac{d}{dm} \log p_m (x) 	\right]	 $$ 
;; **

;; **
;;; The Derivation of the Logarithm-Term is given by:
;;; $$ 		\frac{d}{dm} \log p_m (x)  = 	\frac{d}{dm} \left( - \frac{(x-m)^2}{2\sigma^2}	\right) = \frac{x-m}{\sigma ^2}	$$
;; **

;; @@
; Partial Derivative of gpdf wrt mu: d/dm log(p_m (x))
(defn dm_loggdpf [m sigma x]
  (/ (- x m) (pow sigma 2)) )
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;ppp/dm_loggdpf</span>","value":"#'ppp/dm_loggdpf"}
;; <=

;; **
;;; 3. Application:
;;; ===============
;; **

;; **
;;; 1 . Defining the target function @@ f: \mathbb{R} \to \mathbb{R} @@
;; **

;; @@
;; Defining target function that we want to minimize
(defn target [x] (* (* (pow (- x 2) 2 ) -1)))

;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;ppp/target</span>","value":"#'ppp/target"}
;; <=

;; **
;;; 2 . Defining our probabilistic model:
;;; 
;;; For hill-climbing, we will use the derivative:
;;; 
;;; $$	\frac{d}{dm} F(m) = \mathbb{E}_{x \sim p_m (x) }\left[f(x) \frac{d}{dm} \log p_m (x) 	\right]	 $$ 
;;; 
;;; Since the calculation of the integral is intractable, we approximate the right-hand-side by sampling @@ x \sim \mathcal{N}(m, \sigma ^2)@@ and plugging our samples into the function @@f(x) \frac{d}{dm} \log p_m (x)  @@.
;;; 
;;; 
;; **

;; @@
; Query to sample f(x)d/dm log pm(x) with x~pm(x), pm normal(m,sigma)
(with-primitive-procedures [dm_loggdpf
                            target]
  (defquery obj-func [m sigma]
    (let [x (sample (normal m sigma))]
      (* (target x) (dm_loggdpf m sigma x)))) )


;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;ppp/obj-func</span>","value":"#'ppp/obj-func"}
;; <=

;; **
;;; 3 . Defining a sampling function:
;;; We use the Lightweight-Metropolis-Hastings Algorithm to do the inference.
;;; 
;;; 
;; **

;; @@
; FUNCTION that executes the query and returns the expectation of the sampling approximation
; Input: m .. mean, sigma .. standard deviation, n .. size of samples
(defn sampling [m sigma n]
  (let [lazy-samples (doquery :lmh obj-func [m sigma])
        samples (map :result (take n  (take 10000 (drop 1000 lazy-samples))))]
    (/ (reduce + samples) n)))



;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;ppp/sampling</span>","value":"#'ppp/sampling"}
;; <=

;; **
;;; 4 . Hill-Climbing:
;;; 
;;; Initialize: @@ \sigma @@, @@ m_0 @@ @@ \alpha @@
;; **

;; @@

;; Initial Values
(def sigma 0.0001)
(def m0 2)
(def alpha 0.3)

(def number-of-iterations 100)

;; @@
;; =>
;;; {"type":"list-like","open":"","close":"","separator":"</pre><pre>","items":[{"type":"list-like","open":"","close":"","separator":"</pre><pre>","items":[{"type":"list-like","open":"","close":"","separator":"</pre><pre>","items":[{"type":"html","content":"<span class='clj-var'>#&#x27;ppp/sigma</span>","value":"#'ppp/sigma"},{"type":"html","content":"<span class='clj-var'>#&#x27;ppp/m0</span>","value":"#'ppp/m0"}],"value":"[#'ppp/sigma,#'ppp/m0]"},{"type":"html","content":"<span class='clj-var'>#&#x27;ppp/alpha</span>","value":"#'ppp/alpha"}],"value":"[[#'ppp/sigma,#'ppp/m0],#'ppp/alpha]"},{"type":"html","content":"<span class='clj-var'>#&#x27;ppp/number-of-iterations</span>","value":"#'ppp/number-of-iterations"}],"value":"[[[#'ppp/sigma,#'ppp/m0],#'ppp/alpha],#'ppp/number-of-iterations]"}
;; <=

;; **
;;; Repeat:
;;; 
;;; $$ m_1 \leftarrow m_0 + \alpha \left( \frac{d}{dm} F \right) (m_0) $$
;;; $$ m_2 \leftarrow m_1 + \alpha \left( \frac{d}{dm} F \right) (m_1) $$
;;; $$	\vdots	$$
;;; 
;; **

;; @@
;; FUNCTION: Hillclimbing
;; Input: m0 .. initial value, number-of-iterations .. number of iterations
(defn hill-climbing
  [m0 number-of-iterations]
  (loop [m m0 k number-of-iterations]
    (if (= k 0)
      m
      (recur (+ m (* alpha (sampling m sigma 100))) (- k 1))))) ;; sampling size: 100

;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;ppp/hill-climbing</span>","value":"#'ppp/hill-climbing"}
;; <=

;; **
;;; 5 . Execution of hillclimbing
;; **

;; @@
;; Just for comparison, we execute hill-climbing several times.
(take 5 (repeatedly #(hill-climbing m0 number-of-iterations)))
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>2.000002139488291</span>","value":"2.000002139488291"},{"type":"html","content":"<span class='clj-double'>2.000009412273153</span>","value":"2.000009412273153"},{"type":"html","content":"<span class='clj-double'>2.0000174849358836</span>","value":"2.0000174849358836"},{"type":"html","content":"<span class='clj-double'>2.0000195852386446</span>","value":"2.0000195852386446"},{"type":"html","content":"<span class='clj-double'>1.9999866665709252</span>","value":"1.9999866665709252"}],"value":"(2.000002139488291 2.000009412273153 2.0000174849358836 2.0000195852386446 1.9999866665709252)"}
;; <=

;; @@
;;(with-primitive-procedures [target]
;;(defquery mean-func [m sigma]
;;    (let [x (sample (normal m sigma))]
;;      (target x)) ))
;; @@

;; **
;;; THE END
;;; ======
;;; Extensions of our approach:
;;; ----------
;;; * In Hill-Climbing: 
;;; 
;;; Check, if the updating procedure of @@ m @@ is correct, i.e. is @@ F ( m_0)  \leq F ( m_1) \leq F ( m_2) \leq ... @@ ensured?
;;; 
;;; * Distribution:
;;; 
;;; Including @@ \sigma @@ as second parameter.
;;; 
;;; * Target function:
;;; 
;;; Multi-dimensional case.
;;; 
;; **

;; @@

;; @@
