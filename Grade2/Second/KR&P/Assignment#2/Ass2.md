## Assignment 2
左之睿 191300087 1710670843@qq.com
### Q1、
#### (1) Translate:
(a) There is nothing in the domain.
(b) Everybody who teaches a course is a teacher.
(c) teaches course
(d) 不存在这种表达，$\exist$后面不能放concept name
(e) Everything that is taught is a teacher or a school
(f) Every teacher teaches something.
(g) Every teacher teaches nothing.
(h) If something teaches at least 3 things, then it is a teacher.
(i) If something teaches at least 4 courses, then it is a teacher.
(j) Everything teaches a course.
(k) Everybody teaching something at least teaches 2 things.
(l) Everybody teaching at least 2 things teaches something.
#### (2) State whether it is:
    在此仅列出属于的部分，若没提到则表示该expression并不属于那些类
    (如(d)仅属于none of the above)
(a) DL-Lite and $\mathcal{ALC}$&nbsp;concept inclusion
(b) $\mathcal{EL}$&nbsp;and $\mathcal{ALC}$&nbsp;concept inclusion
(c) $\mathcal{ALC}$&nbsp;concept 
(d) none of the above
(e) DL-Lite with "or" concept inclusion
(f) $\mathcal{EL}$, DL-Lite and $\mathcal{ALC}$&nbsp;concept inclusion
(g) $\mathcal{ALC}$&nbsp;concept inclusion
(h) DL-Lite concept inclusion
(i) none of the above,但是如果对$\mathcal{ALC}$稍加扩展，则它是$\mathcal{ALC}$&nbsp;concept inclusion
(j) $\mathcal{ALC}$&nbsp;concpet inclusion
(k) DL_Lite concept inclusion
(l) DL-Lite concept inclusion
#### (3) Define an  interpretation when it not follows from the empty TBox
(a) not follows from.
对任意的Interpretation $\mathcal{I}$，都有$\Delta^{\mathcal{I}}\not ={\emptyset}$，

(b) not follows from.
Interpretation $\mathcal{I}$:
$\Delta^{\mathcal{I}}=\{a\}$
$teaches^{\mathcal{I}}=\{(a,a)\}$
$Course^{\mathcal{I}}=\{a\}$
$Teacher^{\mathcal{I}}=\emptyset$
故$\exist (teaches.Course)^{\mathcal{I}}=\{a\}\not\subseteq Teacher^{\mathcal{I}}$

(e) not follows from.
Interpretation $\mathcal{I}$:
$\Delta^{\mathcal{I}}=\{a,b\}$
$teaches^{\mathcal{I}}=\{(a,b)\}$
$Teacher^{\mathcal{I}}=\emptyset$
$School^{\mathcal{I}}=\emptyset$
故$(\exist teaches^-.T)^{\mathcal{I}}=\{b\}\not\subseteq(Teacher)^{\mathcal{I}}\sqcup(Department)^{\mathcal{I}}$

(f) not follows from.
Interpretation $\mathcal{I}$:
$\Delta^{\mathcal{I}}=\{a\}$
$Teacher^{\mathcal{I}}=\{a\}$
$teaches^{\mathcal{I}}=\emptyset$
此时$Teacher^{\mathcal{I}}=\{a\}\not\subseteq(\exist teaches.T)^{\mathcal{I}}$

(g) not follows from.
Interpretation $\mathcal{I}$:
$\Delta^{\mathcal{I}}=\{a\}$
$Teacher^{\mathcal{I}}=\{a\}$
$teaches^{\mathcal{I}}=\{(a,a)\}$
此时$Teacher^{\mathcal{I}}=\{a\}\not\subseteq(\exist teaches.\bot)^{\mathcal{I}}$

(h) not follows from.
Interpretation $\mathcal{I}$:
$\Delta^{\mathcal{I}}=\{a,a_1,a_2,a_3\}$
$Teacher^{\mathcal{I}}=\emptyset$
$teaches^{\mathcal{I}}=\{(a,a_1),(a,a_1),(a,a_3)\}$
此时$(\geq3\ teaches.T)^{\mathcal{I}}=\{a\}\not\subseteq\emptyset$

(i) not follows from.
Interpretation $\mathcal{I}$:
$\Delta^{\mathcal{I}}=\{a,b_1,b_2,b_3,b_4\}$
$Teacher^{\mathcal{I}}=\emptyset$
$Course^{\mathcal{I}}=\{b_1,b_2,b_3,b_4\}$
$teaches^{\mathcal{I}}=\{(a,b_1),(a,b_2),(a,b_3),(a,b_4)\}$
此时$(\geq4 teaches.Course)^{\mathcal{I}}=\{a\}\not\subseteq\emptyset$

(j) not follows from.
Interpretation $\mathcal{I}$:
$\Delta^{\mathcal{I}}=\{a\}$
$teaches^{\mathcal{I}}=\emptyset$
$Course^{\mathcal{I}}=\emptyset$
此时$\forall (teaches.T)^{\mathcal{I}}=\{a\}\not\subseteq(\exist teaches.Course)^{\mathcal{I}}$


(k) not follows from.
Interpretation $\mathcal{I}$:
$\Delta^{\mathcal{I}}=\{a,b\}$
$teaches^{\mathcal{I}}=\{(a,b)\}$
我们有$a\isin(\exist teaches.T)^{\mathcal{I}}$但$a\notin(\geq2\ teaches.T)^{\mathcal{I}}$

(l) It follows from the empty TBox.


#### (4) Check satisfiable
(c) satisfiable
Interpretation $\mathcal{I}$:
$\Delta^{\mathcal{I}}=\{a\}$
$teaches^{\mathcal{I}}=\{(a,a)\}$
$Course^{\mathcal{I}}=\{a\}$
此时$(\forall teaches.Course)^{\mathcal{I}}=\{a\}$
### Q2、
##### create:
Mammals$\sqsubseteq$Animals
Lions$\sqsubseteq$Mammals$\sqcap$Carnivore
Giraffe$\sqsubseteq$Mammals$\sqcap$Herbicore
Carnivore$\sqsubseteq\exist$eat.meat
Vertebrate$\equiv$Animal$\sqcap\exist$has.backbone

No. Because $\exist eat.Meat$ is not a concept name 
(a) Lion is an animal lives in Savannah
(b) Animals which eat meat are carnivores.
(c) The vertebrate which has wing,leg and lays egg is a bird.
(d) Reptiles are vertebrates which lay egg.

### Q3、
$A\sqcap{B}=\emptyset$
$\exist r.B=\{1,2\}$
$\exist r.(A\sqcap{B})=\emptyset$
$\Tau=\{1,2,3,4,5,6\}$
$A\sqcap(\exist r.B)=\{1,2\}$

#### True:
$\mathcal{I}\models{A\equiv}\exist r.B$
$\mathcal{I}\models{A\sqcap}B\sqsubseteq\Tau$
$\mathcal{I}\models\exist r.A\sqsubseteq A\sqcap B$


### Q4、
记Parent为A，hasChild为r，Mother为B，Person为C
令$r^{\mathcal{I}}=\{(a,b),(b,c)\}$

$A^{\mathcal{I}}=\{a,b\}$

$B^{\mathcal{I}}=\{b\}$

$C^{\mathcal{I}}=\{a,b,c\}$
此时$\mathcal{I}\models\mathcal{T}但是\mathcal{I}\not\models Parent\sqsubseteq Mother$

### Q5、
(a) No, 因为$\equiv$不能出现在normal form中
(b) 
$X$ is a fresh concept name
$Bird\sqsubseteq Vertebrate$
$Bird\sqsubseteq\exist has\_part.Wing$
$Reptile \sqsubseteq Vertebrate$
$Reptile \sqsubseteq \exist lays.Egg$
$X\sqsubseteq\exist has\_part.Wing$
$\exist has\_part.Wing\sqsubseteq X$
$Vertebrate\sqcap X\sqsubseteq Bird$
(c)首先初始化$S(A)=\{A\},R(r)=\emptyset$ for $A$ and $r$ in $\mathcal{T'}$
然后使用如下四个规则进行处理
$simpleR:{\ \ }if\ A'\isin S(A)\ and\  A'\sqsubseteq B\isin\mathcal{T'}\ and\ B \not\isin S(A)$
$then\ S(A):=S(A)\cup \{B\}$

$conjR:{\ \ }if\ A_1,A_2\isin S(A)\ and\ A_1\sqcap A_2\sqsubseteq B\isin\mathcal{T'}\ and\ B\not\isin S(A)$
$then\ S(A):=S(A)\cup \{B\}$

$rightR:\ \ if\ A'\isin S(A)\ and\ A'\sqsubseteq\exist r.B\isin\mathcal{T'}\ and\ (A,B)\not\isin R(r)$
$then\ R(r):=R(r)\cup{(A,B)}$

$leftR:\ \ if\ (A,B)\isin R(r)\ and\ B'\isin S(B)\ and\ \exist r.B'\sqsubseteq A'\isin\mathcal{T'}\ and\ A'\not\isin S(A)$
$then\ S(A):=S(A)\cup\{A'\}$
最后，$\mathcal{T'}\models{A\sqsubseteq B}\ iff\ B\ in\ S(A)$
(d)
$Reptile\sqsubseteq_{\mathcal{T'}}Vertebrate$: Yes
$Vertebrate\sqsubseteq_{\mathcal{T'}}Bird$: No

### Q6、
(a) No,因为&nbsp;$X\sqcap Y\sqsubseteq\exist r.B$不是normal form
(b) 题目中并未给出Z的具体来源，在此视为对$X\sqcap Y\sqsubseteq\exist r.B$处理得到的产物，即$X\sqcap Y\sqsubseteq Z,Z\sqsubseteq\exist r.B$
$A\sqsubseteq_{\mathcal{T}}Z:$ Yes
$B\sqsubseteq_{\mathcal{T}}Z:$ No
$X\sqsubseteq_{\mathcal{T}}Y:$ No
$A\sqsubseteq_{\mathcal{T}}A':$ Yes
$B\sqsubseteq_{\mathcal{T}}B':$ Yes

### Q7、
设$\mathcal{T}$是一个$\mathcal{EL}$-TBox，定义如下的解释$\mathcal{I}$:
$\Delta^{\mathcal{I}}=\{a\}$
$A^{\mathcal{I}}=\{a\}$（对所有的concept name A都是如此）
$r^{\mathcal{I}}=\{(a,a)\}$（对所有的role name r都是如此）
此时，$for\ all\ \mathcal{EL}-concepts\ A,有A^{\mathcal{I}}=\{a\}$
显然$\mathcal{I}\models A\sqsubseteq B$对所有$\mathcal{EL}$-concept inclusions $A\sqsubseteq B$成立，因此$\mathcal{I}$是$\mathcal{T}$的一个model，证明完毕