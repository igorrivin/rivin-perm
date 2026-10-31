# Low-rank permanents: four worked examples

A permanent sums the weights of every bijection between two sets of size n:

\[
 \operatorname{per}(A)=\sum_{\sigma\in S_n}\prod_i A_{i,\sigma(i)}.
\]

There are n! such assignments, but a small number of shared features can let
us aggregate their contributions without listing them. These examples show
what that aggregation computes in matching and probability. All inputs are
deterministic synthetic models, not measurements from an application dataset.

Run the three exact examples from this checkout or an unpacked source archive:

```sh
python showcase_low_rank.py
```

To include the full-rank research example and retain exact values and raw times:

```sh
python showcase_low_rank.py --include-approximation --output /tmp/showcases.json
```

The optional fourth example needs `notes/experiments/near_low_rank_permanent.py`
from the checkout. The three exact examples use the public `matrix_permanent`
API and have no dependency on the private notes tree. The script is a source
tool included in the sdist, rather than a new installed wheel API.

## 1. Matching through polynomial features

Suppose each item on the left has an attribute x_i and each item on the right
has an attribute y_j. Give their pairing the weight

\[
 A_{ij}=(1+x_i y_j)^2=1+2x_i y_j+x_i^2y_j^2.
\]

The permanent is the weighted assignment partition function: the total weight
of all one-to-one pairings. Dividing by n! gives the mean product weight under
a uniformly random bijection. Dividing a particular assignment's product
weight by this partition function gives its probability in the corresponding
weighted matching model. Such a model has three feature channels, so its
matrix has rank at most three even when every row and column is different.

Our example takes x_i=y_i=i, for i=1,...,n. Rank is exactly three for n>=3.
Entries are positive integers, so the existing exact API applies directly:

```python
from perm import matrix_permanent

n = 40
A = [[(1 + i*j)**2 for j in range(1, n+1)] for i in range(1, n+1)]
partition_function = matrix_permanent(A, parallel=False)
```

At order 40 there are only binom(42,2)=861 final homogeneous coefficient
states per side. This example demonstrates rank compression without repeated
rows or columns. A model A_ij=sum_a u_ia v_aj has the same interpretation:
each pairing's compatibility is a sum over a few shared channels.

## 2. Gaussian intensity moments from three shared modes

Let g_1,g_2,g_3 be independent standard **proper complex** Gaussian variables,
with E|g_a|^2=1. A sensor or field location observes

\[
 z_i=\sum_{a=1}^3F_{ia}g_a,\qquad G=FF^*.
\]

Complex Wick's theorem gives

\[
 \mathbb E\left[\prod_i |z_i|^2\right]=\operatorname{per}(G).
\]

Thus the permanent computes a joint moment of all sensor intensities, and
G has rank at most three regardless of the number of sensors. We choose
nonnegative integer loadings F_i=(1,i+1,1+((i^2+3i) mod 7)), with i starting
at zero. This makes G an integer matrix of rank exactly three, with every
row distinct. No complex arithmetic is needed to evaluate this example.

The script also reports per(G)/prod_i G_ii, the joint moment after normalizing
each sensor to mean intensity one. Independent sensors have normalized moment
one; perfectly shared single-mode signals have normalized moment n!. This
separates overall signal strength from intensity correlations. The formula
computes a moment, which can be much greater than one. It
uses proper complex Gaussians: a real Gaussian product moment has different
pairing combinatorics. See [Fassino, Pistone and Rogantin, Section 2.3](https://www.giannidiorestino.it/PDFs/2019-computing-the-moments-of-the-complex-gaussian-full-and-sparse-covariance-matrix.pdf).

## 3. Probability of an unlabeled frequency profile

There are n known categories with candidate probabilities p_i=w_i/S, where
S=sum_i w_i. After N observations, retain only the multiset of counts f_j,
including zeros, rather than which category had each count. Define

\[
 A_{ij}=w_i^{f_j},\quad N=\sum_j f_j,\quad
 m_k=\#\{j:f_j=k\}.
\]

The probability of that unlabeled profile under the candidate distribution is

\[
 \Pr(\text{profile})=
 \frac{N!\operatorname{per}(A)}
 {S^N\left(\prod_j f_j!\right)\left(\prod_k m_k!\right)}.
\]

The permanent sums over assignments of counts to category labels. The m_k!
factors remove duplicate assignments of equal counts, including zero counts.
The other factors are the usual multinomial normalization. The script uses
w_i=i and f_j=j mod 3 (zero-based j), giving three distinct count values and
rank exactly three. It returns the probability as an exact rational number.
This evaluates a given candidate distribution; it does not optimize the
profile maximum likelihood objective or infer an unknown alphabet size.

Permanents of this form appear in [profile maximum likelihood](https://proceedings.mlr.press/v134/anari21a.html).
Repeated columns offer an additional shortcut here. The script independently
groups identical columns, extracts their multiplicity coefficient, and times
that calculation too. It can beat the general low-rank implementation; all
results and timings are retained.

## 4. Independent noise makes the covariance full rank

For a related sensor model, take two shared Gaussian modes with loadings
F_i=(1,i+1), plus independent proper complex sensor noise of variance 1/1024.
The covariance becomes

\[
 A=FF^T+D,\qquad D=I/1024.
\]

It is positive definite and has rank n. Its permanent is still the joint
intensity moment, but the public low-rank specialization does not
apply. The optional example belongs to a separate research checkout, not this public
repository. It uses research tools to compute:

1. an exact diagonal-plus-low-rank permanent, as a structured reference;
2. a first-order approximation about B=FF^T, with a rigorous rational interval;
3. an exact generic reference as well, when requested at a permitted order.

The first-order estimate is per(B)+sum_ij E_ij per(B_-i,-j), with E=A-B.
If |E_ij|<=epsilon*B_ij, its error is at most
per(B)*((1+epsilon)^n-1-n*epsilon). The implementation tightens this with
rowwise/columnwise errors, and reports a point inside its certified interval.
The printed relative error is a **guarantee**; the separately recorded actual
error comes from comparison with the exact structured reference. A small
matrix perturbation alone is not a relative permanent guarantee. All values
in this example are exact rationals, so no measurement or floating-point
uncertainty is hidden in the certificate.

The default accuracy target is 0.001. `--relative-tolerance 1e-8` changes the
target; an unmet target is reported as false rather than silently accepted.
See the research assessment (maintainer archive; not included)
for the derivations. The optional research code and assessment are omitted
from the sdist; the first three examples remain runnable there.

## Comparing methods and reproducing the captures

```sh
# Includes a native exact generic comparator; complete calls, three samples.
python showcase_low_rank.py --order 22 --compare-generic \
  --include-approximation --output /tmp/showcases-22.json

# Larger structured examples; no exponential generic comparison.
python showcase_low_rank.py --order 40 --allow-large \
  --include-approximation --output /tmp/showcases-40.json
```

Generic comparisons above order 14 require the native extension. This script
rejects generic comparisons above order 22 even with `--allow-large`.
Output files must be new, to preserve earlier captures. Exact results, matrices,
model parameters, interpretation values, method order, raw times and source
hashes are recorded in JSON; rational values have decimal-string numerator
and denominator fields. The command line prints shorter scientific values.

Timings include each complete permanent calculation, validation, rank discovery
and CRT. Input construction and the final application normalization are outside
the timings. There is one warmup per method, followed by three samples with
rotated method order. Calls are serial; generic comparisons use the available
scalar backend. These are development examples on synthetic models, not
application holdouts or a claim that every structured method is slower.

Fresh measurements used an Apple M2 Pro, Python 3.11.17 and the existing
native scalar extension, with `OMP_NUM_THREADS=4` in the environment and
`parallel=False` on these calls. All exact methods agreed, including the
generic references at order 22. Times below are medians of three measurements
after one warmup; input construction and interpretation are excluded.

| Order-22 example | Auto exact | Generic exact | Other method |
|---|---:|---:|---:|
| Polynomial-feature matching | 43.9 ms | 1,866 ms | Explicit low rank: 44.4 ms |
| Three-mode Gaussian moment | 29.2 ms | 1,202 ms | Explicit low rank: 28.8 ms |
| Profile likelihood | 13.6 ms | 930 ms | Grouped columns: 0.549 ms |
| Noisy covariance, full rank | Inapplicable | 2,155 ms | Structured exact: 38.7 ms; certified approximation: 4.32 ms |

The first two show about 42x and 41x reductions against generic exact
computation. Profile likelihood improves against generic computation too,
but grouped columns are the best measured method for that input. For the
full-rank covariance the guaranteed relative error is below 3.62e-7; the
observed error against the exact structured reference is recorded separately.

The order-22 profile has eight unseen categories, seven seen once, and seven
seen twice, for 21 observations in total. Under p_i=i/253, the probability
of exactly that count multiset is about 0.011468, or 1.15 percent. The category
labels have been forgotten, but the alphabet still has 22 categories.

| Order-40 example | Auto exact | Other method or interpretation |
|---|---:|---|
| Polynomial-feature matching | 548 ms | 240-digit exact partition function; 861 coefficient states per side |
| Three-mode Gaussian moment | 341 ms | Normalized intensity moment about 1.9941e46 |
| Profile likelihood | 220 ms | Grouped columns: 2.84 ms; profile probability about 0.000256423 |
| Noisy covariance, full rank | Inapplicable | Structured exact: 407 ms; certified approximation: 13.7 ms |

No generic order-40 calculation was run. The noisy covariance certificate
guarantees relative error below 3.82e-7, comfortably within the requested
0.001 target. This demonstrates a bounded approximation rather than an exact
automatic low-rank calculation on a full-rank matrix.

The order-22 capture (maintainer archive; not included)
and order-40 capture (maintainer archive; not included)
retain all results and timings, including the grouped-column advantage and
differences between automatic and explicit low-rank timings. They are
checkout evidence omitted from the source distribution. The 16 application
tests check small permutation sums, Gaussian latent-variable expansion, and
profile normalization by enumerating every length-four sample sequence on a
three-category alphabet. Source-distribution checks run the exact examples
again with no research helper present.

The final native-enabled suite passed **2,221 tests, 70 skipped**, in 58.54 s.
The validation record (maintainer archive; not included)
retains source hashes and distinguishes fresh correctness checks from the
timing captures.
