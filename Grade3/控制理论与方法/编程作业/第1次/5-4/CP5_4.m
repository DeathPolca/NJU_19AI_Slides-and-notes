%(a)
num=[21];
den=[1 2 0];
sys1=tf(num,den);
sys=feedback(sys1,[1]);%解出闭环传递函数得到w和zeta
w=21^0.5;
zeta=1/w;
PO=100*exp(-zeta*pi/(1-zeta^2)^0.5);%算出来49.54%
%(b)
t=[0:0.01:5];
step(sys,t);